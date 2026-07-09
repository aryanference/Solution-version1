"""HGA²C Policy = wires GAT encoder + both actors + critic """
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.gat_encoder import GATEncoder
from models.allocation_actor import AllocationActor
from models.routing_actor import RoutingActor
from models.critic import Critic
from env.masks import compute_swap_mask, compute_relocation_mask


class HGA2CPolicy(nn.Module):
    """Hierarchical Graph-A2C policy."""

    def __init__(
        self,
        node_feature_dim: int = 7,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        ff_dim: int = 512,
        max_swap: int = 7,
        max_reloc: int = 5,
        vehicle_context_dim: int = 6,
        lstm_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        clip_logits: float = 10.0,
        dropout: float = 0.1,
        use_travel_time_bias: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_swap = max_swap
        self.max_reloc = max_reloc

        # shared backbone
        self.encoder = GATEncoder(
            node_feature_dim=node_feature_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_travel_time_bias=use_travel_time_bias,
        )

        # allocation head (level-1)
        self.allocation_actor = AllocationActor(
            d_model=d_model,
            max_swap=max_swap,
            max_reloc=max_reloc,
        )

        # routing head (level-2)
        self.routing_actor = RoutingActor(
            d_model=d_model,
            vehicle_context_dim=vehicle_context_dim,
            hidden_dim=lstm_hidden_dim,
            clip_logits=clip_logits,
        )

        self.critic = Critic(
            d_model=d_model,
            n_global_scalars=2,
            hidden_dim=critic_hidden_dim,
        )

    def encode(self, node_features: torch.Tensor, travel_times: torch.Tensor) -> torch.Tensor:
        return self.encoder(node_features, travel_times)

    def get_value(self, node_embeddings: torch.Tensor, global_scalars: torch.Tensor) -> torch.Tensor:
        return self.critic(node_embeddings, global_scalars)

    def allocate_swaps(
        self,
        node_embeddings: torch.Tensor,
        s_check: list[int],
        a_remaining: int,
        greedy: bool = False,
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        n_regions = len(s_check)
        x_values: list[int] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        budget = a_remaining

        for r in range(n_regions):
            # node r+1 because node 0 is hub
            region_emb = node_embeddings[:, r + 1, :]

            raw_mask = compute_swap_mask(s_check[r], budget)[:self.max_swap + 1]
            padded = raw_mask + [False] * (self.max_swap + 1 - len(raw_mask))
            mask_t = torch.tensor([padded], dtype=torch.bool, device=node_embeddings.device)

            action, lp, ent = self.allocation_actor.forward_swap(region_emb, mask_t, greedy=greedy)

            x_val = action.item()
            x_values.append(x_val)
            log_probs.append(lp)
            entropies.append(ent)
            budget -= x_val

        return x_values, log_probs, entropies

    def allocate_relocations(
        self,
        node_embeddings: torch.Tensor,
        x_values: list[int],
        s_hat: list[int],
        scooter_cap: int,
        greedy: bool = False,
    ) -> tuple[list[list[int]], list[torch.Tensor], list[torch.Tensor]]:
        n_regions = len(x_values)
        p_matrix = [[0] * n_regions for _ in range(n_regions)]
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for r in range(n_regions):
            committed = 0
            for l in range(n_regions):
                if r == l:
                    continue

                src_emb = node_embeddings[:, r + 1, :]
                dst_emb = node_embeddings[:, l + 1, :]

                raw_mask = compute_relocation_mask(x_values[r], s_hat[r], committed, scooter_cap)
                padded = raw_mask + [False] * (self.max_reloc + 1 - len(raw_mask))
                mask_t = torch.tensor([padded], dtype=torch.bool, device=node_embeddings.device)

                action, lp, ent = self.allocation_actor.forward_relocation(
                    src_emb, dst_emb, mask_t, greedy=greedy
                )

                p_val = action.item()
                p_matrix[r][l] = p_val
                committed += p_val
                log_probs.append(lp)
                entropies.append(ent)

        return p_matrix, log_probs, entropies

    def route_vehicles(
        self,
        node_embeddings: torch.Tensor,
        x_values: list[int],
        p_matrix: list[list[int]],
        instance: dict[str, Any],
        economics: dict[str, Any],
        greedy: bool = False,
    ) -> tuple[list[list[int]], list[torch.Tensor], list[torch.Tensor]]:
        from env.simulator import build_travel_time_matrix, compute_processing_time

        n_regions = instance["region_count"]
        n_vehicles = instance["vehicle_count"]
        C_b = instance["battery_carrying_capacity"]
        C_s = instance["scooter_carrying_capacity"]

        pdp_pairs = [
            (r, l)
            for r in range(n_regions)
            for l in range(n_regions)
            if r != l and p_matrix[r][l] > 0
        ]

        active_regions: set[int] = set()
        for r in range(n_regions):
            if x_values[r] > 0:
                active_regions.add(r)
            for l in range(n_regions):
                if p_matrix[r][l] > 0:
                    active_regions.add(r)
                    active_regions.add(l)

        graph_emb = self.encoder.get_graph_embedding(node_embeddings)
        h, c = self.routing_actor.init_state(graph_emb)

        vehicle_routes: list[list[int]] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        claimed: set[int] = set()

        hub_emb = node_embeddings[:, 0, :]

        for v in range(n_vehicles):
            route: list[int] = []
            bat_load = 0
            scoot_load = 0
            prev_emb = hub_emb

            for step in range(n_regions + 1):
                remaining = active_regions - claimed
                if not remaining:
                    break

                n_nodes = node_embeddings.size(1)
                mask = torch.zeros(1, n_nodes, dtype=torch.bool, device=node_embeddings.device)

                for r in remaining:
                    nidx = r + 1

                    if bat_load + x_values[r] > C_b:
                        continue

                    net_sc = 0
                    for pr, pl in pdp_pairs:
                        if pr == r and pr not in set(route):
                            net_sc += p_matrix[pr][pl]
                        elif pl == r and pr in set(route):
                            net_sc -= p_matrix[pr][pl]
                    if scoot_load + net_sc > C_s:
                        continue

                    # delivery blocked until pickup visited
                    blocked = False
                    for pr, pl in pdp_pairs:
                        if pl == r and pr not in set(route):
                            blocked = True
                            break
                    if blocked:
                        continue

                    mask[0, nidx] = True

                if not mask.any():
                    break

                vctx = torch.tensor(
                    [[scoot_load, bat_load, 0.0,
                      C_s - scoot_load, C_b - bat_load, float(len(route))]],
                    dtype=torch.float32, device=node_embeddings.device
                )

                action, lp, ent, h, c = self.routing_actor.step(
                    node_embeddings, prev_emb, vctx, h, c, mask, greedy=greedy
                )

                nidx = action.item()
                if nidx == 0:
                    break

                region = nidx - 1
                route.append(region)
                claimed.add(region)
                log_probs.append(lp)
                entropies.append(ent)

                bat_load += x_values[region]
                for pr, pl in pdp_pairs:
                    if pr == region and pr not in set(route[:-1]):
                        scoot_load += p_matrix[pr][pl]
                    elif pl == region and pr in set(route):
                        scoot_load -= p_matrix[pr][pl]

                prev_emb = node_embeddings[:, nidx, :]

            vehicle_routes.append(route)

        return vehicle_routes, log_probs, entropies

    def forward(
        self,
        obs: dict[str, Any],
        instance: dict[str, Any],
        economics: dict[str, Any],
        greedy: bool = False,
        x_labels: list[int] | None = None,
        p_labels: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device

        node_features = torch.tensor(
            obs["node_features"], dtype=torch.float32, device=device
        ).unsqueeze(0)

        travel_times = torch.tensor(
            obs["travel_time_matrix"], dtype=torch.float32, device=device
        ).unsqueeze(0)

        node_embs = self.encode(node_features, travel_times)

        s_check = [r["non_functional"] for r in instance["regions"]]
        s_hat   = [r["functional"]     for r in instance["regions"]]
        a_left  = instance["extra_batteries"]
        sc_cap  = instance["scooter_carrying_capacity"]

        if x_labels is not None and p_labels is not None:
            x_values = x_labels
            p_matrix = p_labels
            swap_lps, swap_ents = [], []
            reloc_lps, reloc_ents = [], []
        else:
            x_values, swap_lps, swap_ents = self.allocate_swaps(
                node_embs, s_check, a_left, greedy=greedy
            )
            p_matrix, reloc_lps, reloc_ents = self.allocate_relocations(
                node_embs, x_values, s_hat, sc_cap, greedy=greedy
            )

        routes, route_lps, route_ents = self.route_vehicles(
            node_embs, x_values, p_matrix, instance, economics, greedy=greedy
        )

        scalars = torch.tensor(
            [[float(a_left - sum(x_values)), float(sum(1 for r in routes if r))]],
            dtype=torch.float32, device=device,
        )
        value = self.get_value(node_embs, scalars)

        return {
            "x": x_values,
            "p": p_matrix,
            "vehicle_routes": routes,
            "log_probs": swap_lps + reloc_lps + route_lps,
            "entropies": swap_ents + reloc_ents + route_ents,
            "value": value,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str | Path, device: str = "cpu") -> None:
        sd = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(sd)


def build_policy_from_config(
    hyperparams: dict[str, Any],
    node_feature_dim: int = 7,
) -> HGA2CPolicy:
    return HGA2CPolicy(
        node_feature_dim=node_feature_dim,
        d_model=hyperparams.get("d_model", 128),
        n_heads=hyperparams.get("n_heads", 8),
        n_encoder_layers=hyperparams.get("n_encoder_layers", 3),
        ff_dim=hyperparams.get("ff_dim", 512),
        max_swap=hyperparams.get("max_swap_per_region", 7),
        max_reloc=hyperparams.get("max_relocation_per_pair", 5),
        vehicle_context_dim=6,
        lstm_hidden_dim=hyperparams.get("lstm_hidden_dim", 128),
        critic_hidden_dim=hyperparams.get("critic_hidden_dim", 128),
        clip_logits=hyperparams.get("pointer_clip", 10.0),
        use_travel_time_bias=hyperparams.get("use_travel_time_bias", True),
    )

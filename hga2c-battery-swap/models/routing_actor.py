"""Level-2 Routing Actor — PDP-aware Pointer Network decoder (§5.3)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RoutingActor(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        vehicle_context_dim: int = 6,
        hidden_dim: int = 128,
        clip_logits: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.clip_logits = clip_logits

        self.context_proj = nn.Linear(vehicle_context_dim, d_model)

        # LSTM input = prev node embedding + vehicle context
        self.lstm = nn.LSTMCell(d_model * 2, hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)

        self.init_h = nn.Linear(d_model, hidden_dim)
        self.init_c = nn.Linear(d_model, hidden_dim)

    def init_state(self, graph_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tanh(self.init_h(graph_embedding)), torch.tanh(self.init_c(graph_embedding))

    def step(
        self,
        node_embeddings: torch.Tensor,
        prev_node_embedding: torch.Tensor,
        vehicle_context: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        routing_mask: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.context_proj(vehicle_context)
        lstm_input = torch.cat([prev_node_embedding, ctx], dim=-1)
        h_new, c_new = self.lstm(lstm_input, (h, c))

        query = self.query_proj(h_new)
        keys  = self.key_proj(node_embeddings)

        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_model)
        scores = self.clip_logits * torch.tanh(scores)
        scores = scores.masked_fill(~routing_mask, float("-inf"))

        # safety: if everything masked, force hub
        all_masked = (~routing_mask).all(dim=-1)
        if all_masked.any():
            scores[all_masked, 0] = 0.0

        dist = Categorical(logits=scores)
        action = scores.argmax(dim=-1) if greedy else dist.sample()

        return action, dist.log_prob(action), dist.entropy(), h_new, c_new

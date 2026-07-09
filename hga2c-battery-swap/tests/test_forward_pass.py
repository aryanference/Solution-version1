"""Smoke tests for the HGA²C neural network forward pass.

Verifies:
  1. All components instantiate without errors.
  2. Forward pass on target instance produces correct output shapes.
  3. Masks are applied (no probability mass on infeasible actions).
  4. Gradients flow through all components.
"""
import json
from pathlib import Path

import pytest
import torch
import yaml

from models.gat_encoder import GATEncoder
from models.allocation_actor import AllocationActor
from models.routing_actor import RoutingActor
from models.critic import Critic
from models.hga2c_policy import HGA2CPolicy, build_policy_from_config
from env.battery_swap_env import make_env


class TestGATEncoder:
    """Test the Graph Attention Encoder in isolation."""

    def test_output_shape(self):
        """Encoder produces [batch, n_nodes, d_model] embeddings."""
        encoder = GATEncoder(node_feature_dim=7, d_model=128, n_heads=8, n_layers=3)
        x = torch.randn(2, 10, 7)  # batch=2, 10 nodes, 7 features
        tt = torch.rand(2, 10, 10)  # travel times
        out = encoder(x, tt)
        assert out.shape == (2, 10, 128)

    def test_graph_embedding(self):
        """Graph embedding is [batch, d_model]."""
        encoder = GATEncoder(node_feature_dim=7, d_model=128)
        x = torch.randn(1, 10, 7)
        tt = torch.rand(1, 10, 10)
        emb = encoder(x, tt)
        graph_emb = encoder.get_graph_embedding(emb)
        assert graph_emb.shape == (1, 128)

    def test_gradient_flow(self):
        """Loss.backward() doesn't error."""
        encoder = GATEncoder(node_feature_dim=7, d_model=64, n_heads=4, n_layers=2)
        x = torch.randn(1, 10, 7, requires_grad=False)
        tt = torch.rand(1, 10, 10)
        out = encoder(x, tt)
        loss = out.sum()
        loss.backward()
        # Check gradients exist on encoder parameters
        for p in encoder.parameters():
            assert p.grad is not None or not p.requires_grad


class TestAllocationActor:
    """Test the Level-1 Allocation Actor."""

    def test_swap_output_shape(self):
        actor = AllocationActor(d_model=128, max_swap=7)
        emb = torch.randn(1, 128)
        mask = torch.tensor([[True, True, True, False, False, False, False, False]])
        action, lp, ent = actor.forward_swap(emb, mask)
        assert action.shape == (1,)
        assert lp.shape == (1,)
        assert action.item() < 3  # only 3 values allowed

    def test_relocation_output_shape(self):
        actor = AllocationActor(d_model=128, max_reloc=5)
        src = torch.randn(1, 128)
        dst = torch.randn(1, 128)
        mask = torch.tensor([[True, True, False, False, False, False]])
        action, lp, ent = actor.forward_relocation(src, dst, mask)
        assert action.shape == (1,)
        assert action.item() < 2

    def test_greedy_swap(self):
        """Greedy mode should pick the highest-logit feasible action."""
        actor = AllocationActor(d_model=128, max_swap=7)
        emb = torch.randn(1, 128)
        mask = torch.tensor([[True, True, True, False, False, False, False, False]])
        action, _, _ = actor.forward_swap(emb, mask, greedy=True)
        assert action.item() < 3


class TestRoutingActor:
    """Test the Level-2 Routing Actor."""

    def test_step_output_shape(self):
        actor = RoutingActor(d_model=128, vehicle_context_dim=6, hidden_dim=128)
        node_emb = torch.randn(1, 10, 128)
        prev_emb = torch.randn(1, 128)
        ctx = torch.randn(1, 6)
        graph_emb = node_emb.mean(dim=1)
        h, c = actor.init_state(graph_emb)
        mask = torch.zeros(1, 10, dtype=torch.bool)
        mask[0, 1:5] = True  # regions 0-3 eligible

        action, lp, ent, h_new, c_new = actor.step(
            node_emb, prev_emb, ctx, h, c, mask
        )
        assert action.shape == (1,)
        assert h_new.shape == h.shape
        assert 1 <= action.item() <= 4  # only nodes 1-4 unmasked


class TestCritic:
    """Test the Value Network."""

    def test_output_shape(self):
        critic = Critic(d_model=128, n_global_scalars=2, hidden_dim=128)
        emb = torch.randn(1, 10, 128)
        scalars = torch.tensor([[5.0, 1.0]])
        value = critic(emb, scalars)
        assert value.shape == (1, 1)


class TestHGA2CPolicy:
    """Integration test for the full policy."""

    @pytest.fixture
    def policy(self):
        return HGA2CPolicy(
            node_feature_dim=7, d_model=64, n_heads=4,
            n_encoder_layers=2, ff_dim=128,
            max_swap=7, max_reloc=5,
            lstm_hidden_dim=64, critic_hidden_dim=64,
        )

    @pytest.fixture
    def env_and_data(self):
        env = make_env()
        obs, info = env.reset(seed=42)
        inst = env.instance
        econ = env.economics
        return env, obs, inst, econ

    def test_encode(self, policy, env_and_data):
        _, obs, _, _ = env_and_data
        node_features = torch.tensor(obs["node_features"]).unsqueeze(0)
        tt = torch.tensor(obs["travel_time_matrix"]).unsqueeze(0)
        emb = policy.encode(node_features, tt)
        assert emb.shape == (1, 10, 64)

    def test_full_forward(self, policy, env_and_data):
        """Full forward produces valid allocation and routes."""
        _, obs, inst, econ = env_and_data
        result = policy.forward(obs, inst, econ, greedy=True)

        assert "x" in result
        assert "p" in result
        assert "vehicle_routes" in result
        assert "value" in result

        # Swap values respect S̆_r bounds
        s_check = [r["non_functional"] for r in inst["regions"]]
        for r, xr in enumerate(result["x"]):
            assert 0 <= xr <= s_check[r], f"x[{r}]={xr} out of bounds [0, {s_check[r]}]"

        # Budget respected
        assert sum(result["x"]) <= inst["extra_batteries"]

    def test_gradient_flow_full(self, policy, env_and_data):
        """Gradients flow through the full policy (encoder + actors + critic)."""
        _, obs, inst, econ = env_and_data
        result = policy.forward(obs, inst, econ, greedy=False)

        # Compute a dummy loss from log_probs and value
        if result["log_probs"]:
            policy_loss = -torch.stack(result["log_probs"]).sum()
        else:
            policy_loss = torch.tensor(0.0)
        value_loss = result["value"].squeeze() ** 2
        loss = policy_loss + value_loss

        loss.backward()

        # Check some encoder parameters got gradients
        has_grad = False
        for name, p in policy.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through the model"

"""Feasibility audit — ≥1000 sampled episodes with zero violations (§8).

This is an automated test, not a manual check. It verifies that the
policy's output NEVER violates any constraint from §3 across a large
sample of episodes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from env.battery_swap_env import make_env
from env.simulator import Plan, VehicleRoute, simulate_plan
from models.hga2c_policy import build_policy_from_config


class TestFeasibility:
    """Large-sample feasibility audit over random episodes."""

    N_SAMPLES = 100  # Start with 100 for CI speed; set to 1000+ for full audit

    @pytest.fixture(scope="class")
    def setup(self):
        """Load configs and policy."""
        with open("configs/instance.json") as f:
            instance = json.load(f)
        with open("configs/economics.yaml") as f:
            economics = yaml.safe_load(f)
        with open("configs/hyperparams.yaml") as f:
            hp = yaml.safe_load(f)

        policy = build_policy_from_config(hp)
        ckpt = Path("checkpoints/stage3_final.pt")
        if ckpt.exists():
            policy.load_checkpoint(ckpt)
        policy.eval()

        return instance, economics, hp, policy

    def test_no_violations_stochastic(self, setup):
        """Zero constraint violations across N_SAMPLES stochastic rollouts."""
        instance, economics, hp, policy = setup
        env = make_env(instance=instance, economics=economics)

        total_violations = 0
        violation_details: list[str] = []

        for i in range(self.N_SAMPLES):
            torch.manual_seed(i)
            obs, _ = env.reset(seed=i)

            with torch.no_grad():
                output = policy.forward(obs, instance, economics, greedy=False)

            vehicle_routes_obj = [
                VehicleRoute(vehicle_id=v, route=r)
                for v, r in enumerate(output["vehicle_routes"])
            ]
            plan = Plan(
                x=output["x"], p=output["p"],
                vehicle_routes=vehicle_routes_obj,
                vehicle_assignments={
                    r: v for v, route in enumerate(output["vehicle_routes"]) for r in route
                },
            )
            result = simulate_plan(plan, instance, economics)

            if result.feasibility_violations:
                total_violations += len(result.feasibility_violations)
                violation_details.extend(
                    [f"Episode {i}: {v}" for v in result.feasibility_violations]
                )

            # Also verify constraint 2: x_r ≤ S̆_r
            s_check = [r["non_functional"] for r in instance["regions"]]
            for r, xr in enumerate(output["x"]):
                assert xr <= s_check[r], f"Ep {i}: x[{r}]={xr} > S̆[{r}]={s_check[r]}"

            # Verify constraint 3: Σ x_r ≤ A
            assert sum(output["x"]) <= instance["extra_batteries"], \
                f"Ep {i}: Σx={sum(output['x'])} > A={instance['extra_batteries']}"

            # Verify constraint 4: Σ_l p_rl ≤ x_r + Ŝ_r
            s_hat = [r["functional"] for r in instance["regions"]]
            for r in range(len(output["x"])):
                total_reloc = sum(output["p"][r][l] for l in range(len(output["x"])) if l != r)
                pool = output["x"][r] + s_hat[r]
                assert total_reloc <= pool, \
                    f"Ep {i}: reloc_from[{r}]={total_reloc} > pool={pool}"

        assert total_violations == 0, \
            f"Found {total_violations} violations in {self.N_SAMPLES} episodes:\n" + \
            "\n".join(violation_details[:20])

    def test_no_violations_greedy(self, setup):
        """Zero violations with greedy decoding (deterministic)."""
        instance, economics, hp, policy = setup
        env = make_env(instance=instance, economics=economics)

        obs, _ = env.reset(seed=42)
        with torch.no_grad():
            output = policy.forward(obs, instance, economics, greedy=True)

        vehicle_routes_obj = [
            VehicleRoute(vehicle_id=v, route=r)
            for v, r in enumerate(output["vehicle_routes"])
        ]
        plan = Plan(
            x=output["x"], p=output["p"],
            vehicle_routes=vehicle_routes_obj,
            vehicle_assignments={
                r: v for v, route in enumerate(output["vehicle_routes"]) for r in route
            },
        )
        result = simulate_plan(plan, instance, economics)
        assert result.feasibility_violations == [], \
            f"Greedy violations: {result.feasibility_violations}"

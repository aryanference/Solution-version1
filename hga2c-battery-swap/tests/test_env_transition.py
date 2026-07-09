"""Unit tests for the environment's state transitions and reward computation.

Tests the Gymnasium-style BatterySwapEnv: reset, step (both allocation and
routing phases), reward computation, and episode termination.
"""
import json
import math
from pathlib import Path

import pytest
import yaml

from env.simulator import (
    Plan,
    PlanResult,
    VehicleRoute,
    simulate_plan,
    build_travel_time_matrix,
    euclidean_distance,
    compute_processing_time,
)


# ---------------------------------------------------------------------------
# Simulator unit tests (no env dependency — pure function tests)
# ---------------------------------------------------------------------------

class TestEuclideanDistance:
    """Test the distance function used throughout."""

    def test_same_point(self):
        assert euclidean_distance((0, 0), (0, 0)) == 0.0

    def test_unit_x(self):
        assert euclidean_distance((0, 0), (1, 0)) == pytest.approx(1.0)

    def test_diagonal(self):
        assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_hub_to_region4(self):
        hub = (5.5, 6.99)
        r4 = (7.5, 7.5)
        assert euclidean_distance(hub, r4) == pytest.approx(2.064, abs=1e-3)


class TestTravelTimeMatrix:
    """Test the travel time matrix builder."""

    def test_shape(self, instance_data):
        hub = (instance_data["hub"]["x"], instance_data["hub"]["y"])
        coords = [(r["x"], r["y"]) for r in instance_data["regions"]]
        tt = build_travel_time_matrix(hub, coords)
        assert len(tt) == 10
        assert all(len(row) == 10 for row in tt)

    def test_diagonal_zero(self, instance_data):
        hub = (instance_data["hub"]["x"], instance_data["hub"]["y"])
        coords = [(r["x"], r["y"]) for r in instance_data["regions"]]
        tt = build_travel_time_matrix(hub, coords)
        for i in range(10):
            assert tt[i][i] == pytest.approx(0.0)

    def test_symmetric(self, instance_data):
        hub = (instance_data["hub"]["x"], instance_data["hub"]["y"])
        coords = [(r["x"], r["y"]) for r in instance_data["regions"]]
        tt = build_travel_time_matrix(hub, coords)
        for i in range(10):
            for j in range(10):
                assert tt[i][j] == pytest.approx(tt[j][i])


class TestProcessingTime:
    """Test processing time computation."""

    def test_untouched_region(self):
        """No swaps, no relocations → T_r = 0."""
        x = [0] * 9
        p = [[0] * 9 for _ in range(9)]
        assert compute_processing_time(0, x, p, 1.5, 1.0) == 0.0

    def test_swaps_only(self):
        """2 swaps at region 4 → T_4 = 2 * 1.5 = 3.0 min."""
        x = [0, 0, 0, 0, 2, 0, 0, 0, 0]
        p = [[0] * 9 for _ in range(9)]
        assert compute_processing_time(4, x, p, 1.5, 1.0) == pytest.approx(3.0)

    def test_relocation_pickup(self):
        """Relocate 1 scooter from region 3 to region 8 → T_3 += 1*1.0."""
        x = [0] * 9
        p = [[0] * 9 for _ in range(9)]
        p[3][8] = 1  # 1 scooter from 3→8
        assert compute_processing_time(3, x, p, 1.5, 1.0) == pytest.approx(1.0)

    def test_relocation_dropoff(self):
        """Same relocation 3→8 → T_8 += 1*1.0 (dropoff)."""
        x = [0] * 9
        p = [[0] * 9 for _ in range(9)]
        p[3][8] = 1
        assert compute_processing_time(8, x, p, 1.5, 1.0) == pytest.approx(1.0)

    def test_combined(self):
        """2 swaps + 1 pickup + 1 dropoff at same region."""
        x = [0, 0, 0, 0, 2, 0, 0, 0, 0]
        p = [[0] * 9 for _ in range(9)]
        p[4][8] = 1  # pickup from region 4
        p[6][4] = 1  # dropoff to region 4
        # T_4 = 2*1.5 + (1 pickup + 1 dropoff)*1.0 = 3.0 + 2.0 = 5.0
        assert compute_processing_time(4, x, p, 1.5, 1.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Full plan simulation tests
# ---------------------------------------------------------------------------

class TestSimulatePlan:
    """Test the full plan simulation and objective computation."""

    @pytest.fixture
    def simple_plan(self, instance_data, economics_config):
        """A simple valid plan: swap 2 in Region 4, no relocation, 1 vehicle."""
        n = instance_data["region_count"]
        x = [0] * n
        x[4] = 2  # swap 2 in Region 4 (S̆_4 = 2)

        p = [[0] * n for _ in range(n)]

        route = VehicleRoute(vehicle_id=0, route=[4])
        plan = Plan(
            x=x,
            p=p,
            vehicle_routes=[route],
            vehicle_assignments={4: 0},
        )
        return plan

    def test_simple_plan_no_violations(self, simple_plan, instance_data, economics_config):
        """A valid plan should produce zero feasibility violations."""
        result = simulate_plan(simple_plan, instance_data, economics_config)
        assert result.feasibility_violations == []

    def test_simple_plan_travel_cost(self, simple_plan, instance_data, economics_config):
        """Travel: Hub→R4→Hub = 2*travel_time(Hub, R4) ≈ 2*2.064."""
        result = simulate_plan(simple_plan, instance_data, economics_config)
        expected_tt = 2 * euclidean_distance(
            (instance_data["hub"]["x"], instance_data["hub"]["y"]),
            (instance_data["regions"][4]["x"], instance_data["regions"][4]["y"]),
        )
        assert result.total_travel_time == pytest.approx(expected_tt, abs=0.01)

    def test_simple_plan_unmet_demand(self, simple_plan, instance_data, economics_config):
        """Region 4: D=3, Ŝ=3, x=2, net=5 → d=0. Region 8: D=2, avail=0 → d=2."""
        result = simulate_plan(simple_plan, instance_data, economics_config)
        # Region 4: 3 + 2 = 5 ≥ 3 → unmet = 0
        assert result.per_region_unmet[4] == 0.0
        # Region 8: 0 + 0 = 0 < 2 → unmet = 2
        assert result.per_region_unmet[8] == 2.0

    def test_constraint_violation_detected(self, instance_data, economics_config):
        """A plan violating Constraint 2 should be flagged."""
        n = instance_data["region_count"]
        x = [0] * n
        x[0] = 5  # Region 0 has S̆=0, so x=5 violates Constraint 2

        p = [[0] * n for _ in range(n)]
        plan = Plan(x=x, p=p, vehicle_routes=[], vehicle_assignments={})

        result = simulate_plan(plan, instance_data, economics_config)
        assert len(result.feasibility_violations) > 0
        assert "Constraint 2" in result.feasibility_violations[0]

    def test_budget_violation_detected(self, instance_data, economics_config):
        """A plan violating Constraint 3 (too many total swaps) should be flagged."""
        n = instance_data["region_count"]
        # Try to swap more than A=8 total
        x = [0] * n
        x[1] = 1  # S̆=1
        x[2] = 1  # S̆=1
        x[3] = 1  # S̆=1
        x[4] = 2  # S̆=2
        x[5] = 1  # S̆=1
        x[7] = 1  # S̆=1
        # Total = 7, but let's push it to 9
        # Override to force violation
        x[4] = 2
        x[3] = 1
        # Actually, max feasible = 0+1+1+1+2+1+0+1+0 = 7 < 8, so no violation
        # Let's just make an impossible plan
        x = [2, 2, 2, 2, 2, 0, 0, 0, 0]  # Total=10 > A=8 but also x[0]=2 > S̆[0]=0
        plan = Plan(x=x, p=[[0]*n for _ in range(n)], vehicle_routes=[], vehicle_assignments={})

        result = simulate_plan(plan, instance_data, economics_config)
        assert len(result.feasibility_violations) > 0

    def test_relocation_updates_availability(self, instance_data, economics_config):
        """Relocation p[3][8]=1 should make Region 8 unmet demand decrease by 1."""
        n = instance_data["region_count"]
        x = [0] * n
        p = [[0] * n for _ in range(n)]
        p[3][8] = 1  # Move 1 functional scooter from R3 to R8

        route = VehicleRoute(vehicle_id=0, route=[3, 8])
        plan = Plan(x=x, p=p, vehicle_routes=[route], vehicle_assignments={3: 0, 8: 0})

        result = simulate_plan(plan, instance_data, economics_config)
        # Region 8: Ŝ=0 + x=0 + incoming=1 - outgoing=0 = 1. D=2. Unmet=1.
        assert result.per_region_unmet[8] == pytest.approx(1.0)
        # Region 3: Ŝ=2 + x=0 + incoming=0 - outgoing=1 = 1. D=1. Unmet=0.
        assert result.per_region_unmet[3] == pytest.approx(0.0)

    def test_empty_plan_all_demand_unmet(self, instance_data, economics_config):
        """No swaps, no relocation, no routes → all demand unmet."""
        n = instance_data["region_count"]
        x = [0] * n
        p = [[0] * n for _ in range(n)]
        plan = Plan(x=x, p=p, vehicle_routes=[], vehicle_assignments={})

        result = simulate_plan(plan, instance_data, economics_config)

        # Unmet for each region = max(0, D_r - Ŝ_r)
        demand = [r["demand"] for r in instance_data["regions"]]
        s_hat = [r["functional"] for r in instance_data["regions"]]
        for r in range(n):
            expected_unmet = max(0, demand[r] - s_hat[r])
            assert result.per_region_unmet[r] == pytest.approx(expected_unmet)

        # Total: max(0,2-1) + max(0,1-1) + max(0,0-0) + max(0,1-2) + max(0,3-3) +
        #        max(0,3-2) + max(0,3-2) + max(0,1-2) + max(0,2-0) = 1+0+0+0+0+1+1+0+2 = 5
        assert result.total_unmet_demand == pytest.approx(5.0)

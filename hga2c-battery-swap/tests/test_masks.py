"""Unit tests for feasibility masks (§3 MILP constraints).

TDD: Written BEFORE env/masks.py. Each test encodes one MILP constraint
from §3's table and verifies the mask enforces it correctly.
"""
import pytest


# ---------------------------------------------------------------------------
# Import the masks module (will be implemented in env/masks.py)
# ---------------------------------------------------------------------------
from env.masks import (
    compute_swap_mask,
    compute_relocation_mask,
    compute_budget_mask,
    compute_routing_mask,
)


# ===================================================================
# Constraint 2: x_r ≤ S̆_r (can't swap more than are broken)
# ===================================================================

class TestSwapMask:
    """Constraint 2: swap count limited by non-functional scooters in region."""

    def test_region_with_zero_nonfunctional(self):
        """Region 0 has S̆=0 → only x_r=0 is allowed."""
        mask = compute_swap_mask(s_check_r=0, a_remaining=8)
        assert mask == [True]  # only value 0

    def test_region_with_two_nonfunctional(self):
        """Region 4 has S̆=2 → x_r ∈ {0,1,2} all allowed if budget permits."""
        mask = compute_swap_mask(s_check_r=2, a_remaining=8)
        assert mask == [True, True, True]  # values 0, 1, 2

    def test_region_with_one_nonfunctional(self):
        """Region 1 has S̆=1 → x_r ∈ {0,1}."""
        mask = compute_swap_mask(s_check_r=1, a_remaining=8)
        assert mask == [True, True]

    def test_budget_limits_swap(self):
        """If only 1 battery remains, S̆=2 still limits to {0,1}."""
        mask = compute_swap_mask(s_check_r=2, a_remaining=1)
        assert mask == [True, True, False]  # 0 ok, 1 ok, 2 exceeds budget

    def test_zero_budget_only_zero(self):
        """If no batteries remain, only x_r=0 is feasible."""
        mask = compute_swap_mask(s_check_r=3, a_remaining=0)
        assert mask == [True, False, False, False]

    def test_region_8_always_zero(self):
        """Region 8: S̆=0, demand=2 → can NEVER swap locally (must use relocation)."""
        mask = compute_swap_mask(s_check_r=0, a_remaining=8)
        assert mask == [True]  # only 0


# ===================================================================
# Constraint 3: Σ x_r ≤ A (global battery budget)
# ===================================================================

class TestBudgetMask:
    """Constraint 3: running battery budget across all regions."""

    def test_full_budget(self):
        """With A=8 and no swaps yet, all regions can use up to their S̆."""
        # Region 4: S̆=2, A_remaining=8 → {0,1,2} all ok
        mask = compute_budget_mask(a_remaining=8, s_check_r=2)
        assert mask == [True, True, True]

    def test_budget_exhausted(self):
        """After using all 8 batteries, no more swaps possible."""
        mask = compute_budget_mask(a_remaining=0, s_check_r=2)
        assert mask == [True, False, False]

    def test_partial_budget(self):
        """A_remaining=1, S̆=2 → only {0,1} feasible."""
        mask = compute_budget_mask(a_remaining=1, s_check_r=2)
        assert mask == [True, True, False]

    def test_budget_exceeds_nonfunctional(self):
        """A_remaining=5 but S̆=1 → still only {0,1}."""
        mask = compute_budget_mask(a_remaining=5, s_check_r=1)
        assert mask == [True, True]


# ===================================================================
# Constraint 4: Σ_l p_rl ≤ x_r + Ŝ_r (relocation pool)
# ===================================================================

class TestRelocationMask:
    """Constraint 4: can't relocate more scooters than are available after swaps."""

    def test_no_available_pool(self):
        """Region 8: x_r=0, Ŝ=0 → can't relocate anything FROM Region 8."""
        mask = compute_relocation_mask(
            x_r=0, s_hat_r=0, already_relocated=0, max_possible=5
        )
        # Only p_rl=0 is feasible
        assert mask[0] is True
        assert all(not m for m in mask[1:])

    def test_full_pool_available(self):
        """Region 4: x_r=2, Ŝ=3 → pool=5, can relocate up to 5."""
        mask = compute_relocation_mask(
            x_r=2, s_hat_r=3, already_relocated=0, max_possible=5
        )
        assert all(mask)  # {0,1,2,3,4,5} all feasible

    def test_partial_already_relocated(self):
        """If 3 already relocated from pool of 5, only {0,1,2} more allowed."""
        mask = compute_relocation_mask(
            x_r=2, s_hat_r=3, already_relocated=3, max_possible=5
        )
        remaining = 5 - 3  # 2 more allowed
        assert mask == [True, True, True, False, False, False]

    def test_swap_only_no_functional(self):
        """Region 2: x_r=1, Ŝ=0 → pool=1, can relocate at most 1."""
        mask = compute_relocation_mask(
            x_r=1, s_hat_r=0, already_relocated=0, max_possible=5
        )
        assert mask[0] is True
        assert mask[1] is True
        assert all(not m for m in mask[2:])

    def test_no_swap_functional_only(self):
        """Region 6: x_r=0, Ŝ=2 → pool=2, can relocate up to 2."""
        mask = compute_relocation_mask(
            x_r=0, s_hat_r=2, already_relocated=0, max_possible=5
        )
        assert mask == [True, True, True, False, False, False]


# ===================================================================
# Constraints 5, 13-14, 15-17, 18: Routing masks
# ===================================================================

class TestRoutingMask:
    """Routing feasibility masks enforcing capacity, PDP, and exclusivity."""

    def test_capacity_mask_scooter(self):
        """Constraint 17: Can't visit a pickup node if scooter load would exceed C_v^s."""
        mask = compute_routing_mask(
            eligible_nodes=[1, 2, 3],
            node_scooter_delta={1: 2, 2: 1, 3: -1},  # pickup adds, delivery removes
            node_battery_delta={1: 0, 2: 0, 3: 0},
            current_scooter_load=4,
            current_battery_load=0,
            scooter_capacity=5,
            battery_capacity=5,
            visited_pickups_by_vehicle=set(),
            pdp_pairs=[],  # (pickup, delivery) pairs
            claimed_by_other_vehicle=set(),
            allow_hub_return=False,
        )
        # Node 1: load 4+2=6 > 5 → MASKED
        # Node 2: load 4+1=5 ≤ 5 → ok
        # Node 3: load 4-1=3 ≤ 5 → ok
        assert mask == {1: False, 2: True, 3: True}

    def test_capacity_mask_battery(self):
        """Constraint 18: Can't visit node if battery load would exceed C_v^b."""
        mask = compute_routing_mask(
            eligible_nodes=[1, 2],
            node_scooter_delta={1: 0, 2: 0},
            node_battery_delta={1: 3, 2: 1},
            current_scooter_load=0,
            current_battery_load=4,
            scooter_capacity=5,
            battery_capacity=5,
            visited_pickups_by_vehicle=set(),
            pdp_pairs=[],
            claimed_by_other_vehicle=set(),
            allow_hub_return=False,
        )
        # Node 1: 4+3=7 > 5 → MASKED
        # Node 2: 4+1=5 ≤ 5 → ok
        assert mask == {1: False, 2: True}

    def test_pdp_delivery_masked_until_pickup(self):
        """Constraints 13-14: Delivery node masked until paired pickup visited."""
        mask = compute_routing_mask(
            eligible_nodes=[1, 2, 3],  # 1=pickup, 2=delivery for pair, 3=independent
            node_scooter_delta={1: 1, 2: -1, 3: 0},
            node_battery_delta={1: 0, 2: 0, 3: 1},
            current_scooter_load=0,
            current_battery_load=0,
            scooter_capacity=5,
            battery_capacity=5,
            visited_pickups_by_vehicle=set(),  # pickup NOT yet visited
            pdp_pairs=[(1, 2)],  # node 1 is pickup, node 2 is delivery
            claimed_by_other_vehicle=set(),
            allow_hub_return=False,
        )
        # Node 2 (delivery) should be masked because pickup (1) not visited yet
        assert mask[1] is True   # pickup ok
        assert mask[2] is False  # delivery MASKED
        assert mask[3] is True   # independent ok

    def test_pdp_delivery_unmasked_after_pickup(self):
        """After pickup visited, delivery becomes eligible."""
        mask = compute_routing_mask(
            eligible_nodes=[2, 3],  # pickup 1 already visited
            node_scooter_delta={2: -1, 3: 0},
            node_battery_delta={2: 0, 3: 1},
            current_scooter_load=1,  # carrying the picked-up scooter
            current_battery_load=0,
            scooter_capacity=5,
            battery_capacity=5,
            visited_pickups_by_vehicle={1},  # pickup WAS visited by this vehicle
            pdp_pairs=[(1, 2)],
            claimed_by_other_vehicle=set(),
            allow_hub_return=False,
        )
        assert mask[2] is True  # delivery NOW ok
        assert mask[3] is True

    def test_region_exclusivity(self):
        """Constraint 5: Region claimed by another vehicle → masked."""
        mask = compute_routing_mask(
            eligible_nodes=[1, 2, 3],
            node_scooter_delta={1: 0, 2: 0, 3: 0},
            node_battery_delta={1: 1, 2: 1, 3: 1},
            current_scooter_load=0,
            current_battery_load=0,
            scooter_capacity=5,
            battery_capacity=5,
            visited_pickups_by_vehicle=set(),
            pdp_pairs=[],
            claimed_by_other_vehicle={2},  # Region 2 claimed
            allow_hub_return=False,
        )
        assert mask[1] is True
        assert mask[2] is False  # CLAIMED → masked
        assert mask[3] is True

    def test_hub_return_only_when_allowed(self):
        """Hub return (node 0) only available when explicitly allowed."""
        mask = compute_routing_mask(
            eligible_nodes=[0, 1],  # 0 = hub
            node_scooter_delta={0: 0, 1: 0},
            node_battery_delta={0: 0, 1: 1},
            current_scooter_load=0,
            current_battery_load=0,
            scooter_capacity=5,
            battery_capacity=5,
            visited_pickups_by_vehicle=set(),
            pdp_pairs=[],
            claimed_by_other_vehicle=set(),
            allow_hub_return=False,
        )
        assert mask[0] is False  # hub NOT allowed
        assert mask[1] is True

    def test_hub_return_when_no_eligible(self):
        """When allow_hub_return=True, hub becomes available."""
        mask = compute_routing_mask(
            eligible_nodes=[0],
            node_scooter_delta={0: 0},
            node_battery_delta={0: 0},
            current_scooter_load=0,
            current_battery_load=0,
            scooter_capacity=5,
            battery_capacity=5,
            visited_pickups_by_vehicle=set(),
            pdp_pairs=[],
            claimed_by_other_vehicle=set(),
            allow_hub_return=True,
        )
        assert mask[0] is True

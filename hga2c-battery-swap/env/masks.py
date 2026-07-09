"""Feasibility masks for the Battery Swap & Relocation problem (§3)."""
from __future__ import annotations

from typing import Any


def compute_swap_mask(s_check_r: int, a_remaining: int) -> list[bool]:
    """Compute feasible values for x_r (swaps in region r).

    Constraints enforced:
        - (2) x_r ≤ S̆_r: can't swap more than non-functional count.
        - (3) x_r ≤ A_remaining: can't exceed remaining battery budget.
    """
    max_swaps = s_check_r  # Constraint 2
    mask: list[bool] = []
    for v in range(max_swaps + 1):
        mask.append(v <= a_remaining)  # Constraint 3
    return mask


def compute_budget_mask(a_remaining: int, s_check_r: int) -> list[bool]:
    """Compute swap mask purely from budget perspective (Constraint 3).

    Equivalent to compute_swap_mask but with clearer naming for callers
    that already handle Constraint 2 separately.
    """
    return compute_swap_mask(s_check_r=s_check_r, a_remaining=a_remaining)


def compute_relocation_mask(
    x_r: int,
    s_hat_r: int,
    already_relocated: int,
    max_possible: int,
) -> list[bool]:
    """Compute feasible values for p_rl (relocate from region r to region l).

    Constraint 4: Σ_l p_rl ≤ x_r + Ŝ_r (available pool after swaps).

    The total relocations FROM region r across all destinations l must not
    exceed the available pool. This mask is for one specific (r, l) pair,
    given how many have already been allocated to other destinations.
    """
    pool = x_r + s_hat_r  # Total available for relocation from region r
    remaining_pool = pool - already_relocated
    remaining_pool = max(remaining_pool, 0)

    mask: list[bool] = []
    for v in range(max_possible + 1):
        mask.append(v <= remaining_pool)
    return mask


def compute_routing_mask(
    eligible_nodes: list[int],
    node_scooter_delta: dict[int, int],
    node_battery_delta: dict[int, int],
    current_scooter_load: int,
    current_battery_load: int,
    scooter_capacity: int,
    battery_capacity: int,
    visited_pickups_by_vehicle: set[int],
    pdp_pairs: list[tuple[int, int]],
    claimed_by_other_vehicle: set[int],
    allow_hub_return: bool,
) -> dict[int, bool]:
    """Compute routing mask for the next node selection by a vehicle.

    Constraints enforced:
        - (5)  Region assigned to at most one vehicle (exclusivity).
        - (13) Pickup before delivery in PDP pairs (precedence).
        - (14) Same vehicle serves both pickup and delivery.
        - (17) Scooter carrying capacity: q_vi ≤ C_v^s.
        - (18) Battery carrying capacity.
    """
    # Build a lookup: delivery_node → pickup_node for PDP precedence
    delivery_to_pickup: dict[int, int] = {}
    for pickup, delivery in pdp_pairs:
        delivery_to_pickup[delivery] = pickup

    mask: dict[int, bool] = {}

    for node in eligible_nodes:
        if node == 0:
            mask[node] = allow_hub_return
            continue

        if node in claimed_by_other_vehicle:
            mask[node] = False
            continue

        if node in delivery_to_pickup:
            required_pickup = delivery_to_pickup[node]
            if required_pickup not in visited_pickups_by_vehicle:
                mask[node] = False
                continue

        new_scooter_load = current_scooter_load + node_scooter_delta.get(node, 0)
        if new_scooter_load > scooter_capacity:
            mask[node] = False
            continue

        new_battery_load = current_battery_load + node_battery_delta.get(node, 0)
        if new_battery_load > battery_capacity:
            mask[node] = False
            continue

        # All constraints pass
        mask[node] = True

    return mask

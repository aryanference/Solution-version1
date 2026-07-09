"""Nearest-Neighbor greedy heuristic baseline -NNG"""
from __future__ import annotations

import logging
import time
from typing import Any

from env.masks import compute_swap_mask, compute_relocation_mask
from env.simulator import (
    Plan,
    PlanResult,
    VehicleRoute,
    build_travel_time_matrix,
    simulate_plan,
)

logger = logging.getLogger(__name__)


def solve_nearest_neighbor(       # NNG defined 
    instance: dict[str, Any],
    economics: dict[str, Any],
) -> tuple[Plan, dict[str, Any]]:
    start_time = time.time()

    n_regions = instance["region_count"]
    n_vehicles = instance["vehicle_count"]
    demand = [r["demand"] for r in instance["regions"]]
    s_hat = [r["functional"] for r in instance["regions"]]
    s_check = [r["non_functional"] for r in instance["regions"]]
    A = instance["extra_batteries"]
    C_b = instance["battery_carrying_capacity"]
    C_s = instance["scooter_carrying_capacity"]

    hub_xy = (instance["hub"]["x"], instance["hub"]["y"])
    region_coords = [(r["x"], r["y"]) for r in instance["regions"]]
    tt = build_travel_time_matrix(hub_xy, region_coords)

    x = [0] * n_regions
    a_remaining = A

    deficits = [(max(0, demand[r] - s_hat[r]), r) for r in range(n_regions)]
    deficits.sort(reverse=True)

    for deficit, r in deficits:
        if deficit <= 0 or a_remaining <= 0:
            continue
        swaps = min(s_check[r], deficit, a_remaining)
        x[r] = swaps
        a_remaining -= swaps

    p = [[0] * n_regions for _ in range(n_regions)]

    surplus = []
    deficit_regions = []
    for r in range(n_regions):
        available = s_hat[r] + x[r]
        need = demand[r]
        if available > need:
            surplus.append((available - need, r))
        elif available < need:
            deficit_regions.append((need - available, r))

    surplus.sort(reverse=True)
    deficit_regions.sort(reverse=True)

    for d_amount, d_region in deficit_regions:
        remaining_need = d_amount
        for s_idx in range(len(surplus)):
            s_amount, s_region = surplus[s_idx]
            if s_amount <= 0 or remaining_need <= 0:
                continue
            move = min(s_amount, remaining_need)
            p[s_region][d_region] = move
            surplus[s_idx] = (s_amount - move, s_region)
            remaining_need -= move

    active_regions: set[int] = set()
    for r in range(n_regions):
        if x[r] > 0:
            active_regions.add(r)
        for l in range(n_regions):
            if p[r][l] > 0:
                active_regions.add(r)
                active_regions.add(l)

    vehicle_routes: list[VehicleRoute] = []
    assignments: dict[int, int] = {}
    remaining = set(active_regions)

    for v in range(n_vehicles):
        if not remaining:
            vehicle_routes.append(VehicleRoute(vehicle_id=v))
            continue

        route: list[int] = []
        current_node = 0  # hub
        battery_load = 0
        scooter_load = 0

        while remaining:
            best_r = None
            best_dist = float("inf")
            for r in remaining:
                node = r + 1
                dist = tt[current_node][node]
                new_battery = battery_load + x[r]
                net_scooter = sum(p[r][l] for l in range(n_regions) if l != r) - \
                              sum(p[l][r] for l in range(n_regions) if l != r)
                new_scooter = scooter_load + net_scooter
                if new_battery > C_b or new_scooter > C_s:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_r = r

            if best_r is None:
                break

            route.append(best_r)
            remaining.discard(best_r)
            assignments[best_r] = v
            current_node = best_r + 1
            battery_load += x[best_r]
            scooter_load += sum(p[best_r][l] for l in range(n_regions) if l != best_r) - \
                            sum(p[l][best_r] for l in range(n_regions) if l != best_r)

        vehicle_routes.append(VehicleRoute(vehicle_id=v, route=route))

    while len(vehicle_routes) < n_vehicles:
        vehicle_routes.append(VehicleRoute(vehicle_id=len(vehicle_routes)))

    plan = Plan(
        x=x, p=p,
        vehicle_routes=vehicle_routes,
        vehicle_assignments=assignments,
    )

    solve_time = time.time() - start_time
    solver_info = {
        "status": "Heuristic",
        "solve_time_seconds": solve_time,
        "solver": "NearestNeighbor",
    }

    return plan, solver_info

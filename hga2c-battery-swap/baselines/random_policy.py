"""here its a Random feasible policy baseline (§7) ."""
from __future__ import annotations

import logging
import random
import time
from typing import Any

from env.masks import compute_swap_mask, compute_relocation_mask # see the swap mask and relocation 
from env.simulator import (
    Plan,
    PlanResult,
    VehicleRoute,
    build_travel_time_matrix,
    simulate_plan,
)

logger = logging.getLogger(__name__)


def solve_random(
    instance: dict[str, Any],
    economics: dict[str, Any],
    seed: int = 42,
) -> tuple[Plan, dict[str, Any]]:
    start_time = time.time()
    rng = random.Random(seed)

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

    for r in range(n_regions):
        mask = compute_swap_mask(s_check[r], a_remaining)
        feasible = [v for v, ok in enumerate(mask) if ok]
        chosen = rng.choice(feasible)
        x[r] = chosen
        a_remaining -= chosen

    p = [[0] * n_regions for _ in range(n_regions)]
    reloc_committed = [0] * n_regions

    for r in range(n_regions):
        for l in range(n_regions):
            if r == l:
                continue
            mask = compute_relocation_mask(x[r], s_hat[r], reloc_committed[r], C_s)
            feasible = [v for v, ok in enumerate(mask) if ok]
            chosen = rng.choice(feasible)
            p[r][l] = chosen
            reloc_committed[r] += chosen

    active_regions: set[int] = set()
    for r in range(n_regions):
        if x[r] > 0:
            active_regions.add(r)
        for l in range(n_regions):
            if p[r][l] > 0:
                active_regions.add(r)
                active_regions.add(l)

    remaining = list(active_regions)
    rng.shuffle(remaining)

    vehicle_routes: list[VehicleRoute] = []
    assignments: dict[int, int] = {}
    idx = 0

    for v in range(n_vehicles):
        route: list[int] = []
        battery_load = 0
        scooter_load = 0

        while idx < len(remaining):
            r = remaining[idx]
            new_battery = battery_load + x[r]
            net_scooter = sum(p[r][l_] for l_ in range(n_regions) if l_ != r) - \
                          sum(p[l_][r] for l_ in range(n_regions) if l_ != r)
            new_scooter = scooter_load + net_scooter

            if new_battery > C_b or new_scooter > C_s:
                break

            route.append(r)
            assignments[r] = v
            battery_load = new_battery
            scooter_load += net_scooter
            idx += 1

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
        "status": "RandomFeasible",
        "solve_time_seconds": solve_time,
        "solver": "Random",
    }

    return plan, solver_info

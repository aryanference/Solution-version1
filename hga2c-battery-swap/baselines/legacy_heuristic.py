"""Legacy heuristic baseline: Nearest-unserved-region dispatch.

When vehicles depart from the hub and travel to the nearest unserved region with
either unmet demand or non-functional scooters. They perform swaps and
relocations until capacity is reached, then return to the hub.
"""
from __future__ import annotations

import logging
from typing import Any

from env.simulator import Plan, VehicleRoute, build_travel_time_matrix, simulate_plan

logger = logging.getLogger(__name__)


def solve_legacy_heuristic(
    instance: dict[str, Any],
    economics: dict[str, Any],
) -> tuple[Plan, dict[str, Any]]:
    """Solve using nearest-unserved-region heuristic."""
    n_regions = instance["region_count"]
    n_vehicles = instance["vehicle_count"]
    C_b = instance["battery_carrying_capacity"]
    C_s = instance["scooter_carrying_capacity"]
    extra_batteries = instance["extra_batteries"]
    
    hub_xy = (instance["hub"]["x"], instance["hub"]["y"])
    region_coords = [(r["x"], r["y"]) for r in instance["regions"]]
    tt = build_travel_time_matrix(hub_xy, region_coords)

    demand = [r["demand"] for r in instance["regions"]]
    s_hat = [r["functional"] for r in instance["regions"]]
    s_check = [r["non_functional"] for r in instance["regions"]]
    
    x = [0] * n_regions
    p = [[0] * n_regions for _ in range(n_regions)]
    
    remaining_batteries = extra_batteries
    unserved = set(range(n_regions))
    
    vehicle_routes = []
    
    for v in range(n_vehicles):
        route = []
        battery_load = 0
        scooter_load = 0
        current_node = 0  # hub
        
        while True:
            # Finding candidate regions that need service (demand or non-functional)
            candidates = []
            for r in unserved:
                needs_swap = s_check[r] > 0
                net_scooters = s_hat[r] + x[r]
                needs_reloc = demand[r] > net_scooters
                if needs_swap or needs_reloc:
                    candidates.append(r)
            
            if not candidates:
                break
                
            # Nearest unserved
            best_r = min(candidates, key=lambda r: tt[current_node][r + 1])
            route.append(best_r)
            unserved.remove(best_r)
            current_node = best_r + 1
            
            # Perform Swaps
            swaps_needed = s_check[best_r]
            swaps_possible = min(swaps_needed, C_b - battery_load, remaining_batteries)
            if swaps_possible > 0:
                x[best_r] = swaps_possible
                battery_load += swaps_possible
                remaining_batteries -= swaps_possible
                

            if battery_load >= C_b:
                break # Return to hub
                
        vehicle_routes.append(VehicleRoute(vehicle_id=v, route=route))

    plan = Plan(x=x, p=p, vehicle_routes=vehicle_routes, vehicle_assignments={})
    
    sim_result = simulate_plan(plan, instance, economics)
    
    metrics = {
        "objective_z": sim_result.objective_z,
        "travel_cost": sim_result.travel_cost,
        "unmet_demand_penalty": sim_result.unmet_demand_penalty,
        "delay_penalty": sim_result.delay_penalty,
        "total_unmet_demand": sim_result.total_unmet_demand,
        "demand_fulfillment_rate": sim_result.demand_fulfillment_rate,
        "violations": len(sim_result.feasibility_violations)
    }
    
    return plan, metrics


if __name__ == "__main__":
    import json
    import yaml
    
    with open("configs/instance.json") as f:
        inst = json.load(f)
    with open("configs/economics.yaml") as f:
        econ = yaml.safe_load(f)
        
    plan, metrics = solve_legacy_heuristic(inst, econ)
    print("Legacy Heuristic Plan:")
    print("x:", plan.x)
    for vr in plan.vehicle_routes:
        print(f"Vehicle {vr.vehicle_id}: {vr.route}")
    print("Metrics:", metrics)

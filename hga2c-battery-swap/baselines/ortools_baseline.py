"""OR-Tools CP-SAT can be termed Routing baseline """
from __future__ import annotations

import logging
import time
from typing import Any

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from env.simulator import (
    Plan,
    PlanResult,
    VehicleRoute,
    build_travel_time_matrix,
    simulate_plan,
)

logger = logging.getLogger(__name__)


def solve_ortools(
    instance: dict[str, Any],
    economics: dict[str, Any],
    time_limit_seconds: int = 60,
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

    active_regions: list[int] = []
    for r in range(n_regions):
        if x[r] > 0 or any(p[r][l] > 0 for l in range(n_regions) if l != r) or \
           any(p[l][r] > 0 for l in range(n_regions) if l != r):
            active_regions.append(r)

    if not active_regions:
        vehicle_routes = [VehicleRoute(vehicle_id=v) for v in range(n_vehicles)]
        plan = Plan(x=x, p=p, vehicle_routes=vehicle_routes, vehicle_assignments={})
        solve_time = time.time() - start_time
        return plan, {"status": "NoWork", "solve_time_seconds": solve_time, "solver": "OR-Tools"}

    n_ortools_nodes = 1 + len(active_regions)
    region_to_ortools = {r: i + 1 for i, r in enumerate(active_regions)}
    ortools_to_region = {i + 1: r for i, r in enumerate(active_regions)}

    SCALE = 1000
    dist_matrix: list[list[int]] = []
    for i in range(n_ortools_nodes):
        row: list[int] = []
        for j in range(n_ortools_nodes):
            if i == 0:
                ni = 0
            else:
                ni = active_regions[i - 1] + 1
            if j == 0:
                nj = 0
            else:
                nj = active_regions[j - 1] + 1
            row.append(int(tt[ni][nj] * SCALE))
        dist_matrix.append(row)

    manager = pywrapcp.RoutingIndexManager(
        n_ortools_nodes, n_vehicles, 0  # depot = node 0
    )
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Battery capacity dimension
    def battery_demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        if node == 0:
            return 0
        region = ortools_to_region[node]
        return x[region]

    battery_callback_index = routing.RegisterUnaryTransitCallback(
        battery_demand_callback
    )
    routing.AddDimensionWithVehicleCapacity(
        battery_callback_index, 0, [C_b] * n_vehicles, True, "BatteryLoad"
    )

    # Scooter capacity dimension
    def scooter_demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        if node == 0:
            return 0
        region = ortools_to_region[node]
        pickup = sum(p[region][l] for l in range(n_regions) if l != region)
        dropoff = sum(p[l][region] for l in range(n_regions) if l != region)
        return pickup - dropoff

    scooter_callback_index = routing.RegisterUnaryTransitCallback(
        scooter_demand_callback
    )
    routing.AddDimensionWithVehicleCapacity(
        scooter_callback_index, 0, [C_s] * n_vehicles, True, "ScooterLoad"
    )

    # Pickup-delivery pairs
    for r in range(n_regions):
        for l in range(n_regions):
            if r != l and p[r][l] > 0:
                if r in region_to_ortools and l in region_to_ortools:
                    pickup_index = manager.NodeToIndex(region_to_ortools[r])
                    delivery_index = manager.NodeToIndex(region_to_ortools[l])
                    routing.AddPickupAndDelivery(pickup_index, delivery_index)
                    routing.solver().Add(
                        routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
                    )

    # Allow dropping nodes if infeasible (penalty for unserved)
    penalty = int(economics.get("lambda_unmet", 50.0) * SCALE * 10)
    for i in range(1, n_ortools_nodes):
        routing.AddDisjunction([manager.NodeToIndex(i)], penalty)

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit_seconds

    # Solve
    solution = routing.SolveWithParameters(search_params)

    vehicle_routes_list: list[VehicleRoute] = []
    assignments: dict[int, int] = {}

    if solution:
        for v in range(n_vehicles):
            route: list[int] = []
            index = routing.Start(v)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0 and node in ortools_to_region:
                    region = ortools_to_region[node]
                    route.append(region)
                    assignments[region] = v
                index = solution.Value(routing.NextVar(index))
            vehicle_routes_list.append(VehicleRoute(vehicle_id=v, route=route))
    else:
        # Fallback: no solution found, return empty routes
        logger.warning("OR-Tools found no solution, returning empty routes")
        vehicle_routes_list = [VehicleRoute(vehicle_id=v) for v in range(n_vehicles)]

    plan = Plan(
        x=x, p=p,
        vehicle_routes=vehicle_routes_list,
        vehicle_assignments=assignments,
    )

    solve_time = time.time() - start_time
    solver_info = {
        "status": "Optimal" if solution else "NoSolution",
        "solve_time_seconds": solve_time,
        "solver": "OR-Tools",
    }

    return plan, solver_info

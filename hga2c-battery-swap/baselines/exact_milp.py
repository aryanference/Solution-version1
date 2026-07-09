"""Exact MILP baseline using PuLP + CBC (just for a baseline compariasion)"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pulp

from env.simulator import (
    Plan,
    PlanResult,
    VehicleRoute,
    build_travel_time_matrix,
    simulate_plan,
)

logger = logging.getLogger(__name__)


def solve_milp(
    instance: dict[str, Any],
    economics: dict[str, Any],
    time_limit: int = 600,
    solver_name: str = "CBC",
) -> tuple[Plan, dict[str, Any]]:
    start_time = time.time()

    n_regions = instance["region_count"]
    n_vehicles = instance["vehicle_count"]
    R = list(range(n_regions))                  # region indices
    V = list(range(n_vehicles))                 # vehicle indices
    N_nodes = 1 + n_regions                     # hub (0) + regions (1..R)
    N = list(range(N_nodes))                    # all node indices in TT matrix

    hub_xy = (instance["hub"]["x"], instance["hub"]["y"])
    region_coords = [(r["x"], r["y"]) for r in instance["regions"]]
    demand = [r["demand"] for r in instance["regions"]]
    s_hat = [r["functional"] for r in instance["regions"]]
    s_check = [r["non_functional"] for r in instance["regions"]]
    A = instance["extra_batteries"]
    C_b = instance["battery_carrying_capacity"]
    C_s = instance["scooter_carrying_capacity"]

    lam = economics.get("lambda_travel", 1.0)
    lam_prime = economics.get("lambda_unmet", 50.0)
    period = economics.get("period_length", 60)
    lam_double_prime = lam_prime / period
    swap_time_min = economics.get("swap_time_min", 1.5)
    reloc_time_min = economics.get("reloc_time_min", 1.0)
    urgency = economics.get("urgency_weighting", "demand_proportional")

    h = [float(demand[r]) if urgency == "demand_proportional" else 1.0 for r in R]

    tt = build_travel_time_matrix(hub_xy, region_coords)

    max_tt = max(tt[i][j] for i in N for j in N)
    max_proc = swap_time_min * max(s_check) + reloc_time_min * 2 * max(
        sum(s_hat) + A, 1
    )
    M = (max_tt + max_proc) * N_nodes * 2 + 1000

    # Region node indices in the TT matrix: region r → node r+1
    def rn(r: int) -> int:
        return r + 1

    prob = pulp.LpProblem("BatterySwapRelocation", pulp.LpMinimize)

    x = {r: pulp.LpVariable(f"x_{r}", 0, s_check[r], cat="Integer") for r in R}

    p = {
        (r, l): pulp.LpVariable(f"p_{r}_{l}", 0, cat="Integer")
        for r in R for l in R if r != l
    }

    y = {
        (r, l): pulp.LpVariable(f"y_{r}_{l}", cat="Binary")
        for r in R for l in R if r != l
    }

    u = {v: pulp.LpVariable(f"u_{v}", cat="Binary") for v in V}

    z = {
        (r, v): pulp.LpVariable(f"z_{r}_{v}", cat="Binary")
        for r in R for v in V
    }

    w = {
        (v, i, j): pulp.LpVariable(f"w_{v}_{i}_{j}", cat="Binary")
        for v in V for i in N for j in N if i != j
    }

    q = {
        (v, i): pulp.LpVariable(f"q_{v}_{i}", 0, C_s)
        for v in V for i in N
    }

    m = {
        (v, i): pulp.LpVariable(f"m_{v}_{i}", 0)
        for v in V for i in N
    }

    d = {r: pulp.LpVariable(f"d_{r}", 0) for r in R}

    h_var = {r: pulp.LpVariable(f"h_var_{r}", 0) for r in R}

    T = {r: pulp.LpVariable(f"T_{r}", 0) for r in R}
    for r in R:
        pickups = pulp.lpSum(p[(r, l)] for l in R if l != r)
        dropoffs = pulp.lpSum(p[(l, r)] for l in R if l != r)
        prob += T[r] == swap_time_min * x[r] + reloc_time_min * (pickups + dropoffs), \
            f"ProcessingTime_{r}"

    travel_cost = pulp.lpSum(
        lam * tt[i][j] * w[(v, i, j)]
        for v in V for i in N for j in N if i != j
    )
    unmet_penalty = pulp.lpSum(lam_prime * d[r] for r in R)
    delay_penalty = pulp.lpSum(
        lam_double_prime * h[r] * m[(v, rn(r))]
        for v in V for r in R
    )
    prob += travel_cost + unmet_penalty + delay_penalty, "Objective"


    # Constraint 3: Σ x_r ≤ A
    prob += pulp.lpSum(x[r] for r in R) <= A, "BatteryBudget"

    # Constraint 4: Σ_l p_rl ≤ x_r + Ŝ_r for each r
    for r in R:
        prob += (
            pulp.lpSum(p[(r, l)] for l in R if l != r) <= x[r] + s_hat[r],
            f"RelocationPool_{r}",
        )

    # Constraint 5: Σ_v z_rv ≤ 1 for each r
    for r in R:
        prob += (
            pulp.lpSum(z[(r, v)] for v in V) <= 1,
            f"OneVehiclePerRegion_{r}",
        )

    # Constraint 6: Σ_r z_rv ≤ M * u_v for each v
    for v in V:
        prob += (
            pulp.lpSum(z[(r, v)] for r in R) <= M * u[v],
            f"VehicleUsed_{v}",
        )

    # Constraint 7: Σ_{i∈R} w_{v,i,0} = u_v (return to hub)
    for v in V:
        prob += (
            pulp.lpSum(w[(v, rn(r), 0)] for r in R) == u[v],
            f"ReturnToHub_{v}",
        )

    # Constraint 8: Σ_{j∈R} w_{v,0,j} = u_v (depart from hub)
    for v in V:
        prob += (
            pulp.lpSum(w[(v, 0, rn(r))] for r in R) == u[v],
            f"DepartFromHub_{v}",
        )

    # Constraint 9: Σ_j w_{v,i,j} = z_{i,v} for i ∈ R (flow out)
    for v in V:
        for r in R:
            prob += (
                pulp.lpSum(w[(v, rn(r), j)] for j in N if j != rn(r)) == z[(r, v)],
                f"FlowOut_{v}_{r}",
            )

    # Constraint 10: Σ_i w_{v,i,j} = z_{j,v} for j ∈ R (flow in)
    for v in V:
        for r in R:
            prob += (
                pulp.lpSum(w[(v, i, rn(r))] for i in N if i != rn(r)) == z[(r, v)],
                f"FlowIn_{v}_{r}",
            )

    # Constraint 11: Sub-tour elimination + time tracking
    # m_vj ≥ m_vi + t_ij + T_r - M*(1 - w_vij) for all v, i∈N, j∈R
    for v in V:
        for i in N:
            for r in R:
                j = rn(r)
                if i != j:
                    prob += (
                        m[(v, j)] >= m[(v, i)] + tt[i][j] + T[r] - M * (1 - w[(v, i, j)]),
                        f"SubTour_{v}_{i}_{j}",
                    )

    # Constraint 12: p_rl ≤ M * y_rl (link relocation to indicator)
    for r in R:
        for l in R:
            if r != l:
                prob += (
                    p[(r, l)] <= M * y[(r, l)],
                    f"RelocIndicator_{r}_{l}",
                )

    # Constraint 13: Same vehicle for pickup and delivery
    # z_rv ≥ y_rl + z_lv - 1 (if relocation r→l, same vehicle serves both)
    for r in R:
        for l in R:
            if r != l:
                for v in V:
                    prob += (
                        z[(r, v)] >= y[(r, l)] + z[(l, v)] - 1,
                        f"SameVehicle_{r}_{l}_{v}",
                    )

    # Constraint 14: Pickup before delivery (time ordering)
    # m_v,l ≥ m_v,r + T_r + t_rl - M*(1 - y_rl) for same vehicle
    for r in R:
        for l in R:
            if r != l:
                for v in V:
                    prob += (
                        m[(v, rn(l))] >= m[(v, rn(r))] + tt[rn(r)][rn(l)]
                        - M * (2 - y[(r, l)] - z[(r, v)]),
                        f"PickupBeforeDelivery_{r}_{l}_{v}",
                    )

    # Constraints 15-16: Scooter load tracking
    # q_v,hub = 0 (start with no scooters)
    for v in V:
        prob += q[(v, 0)] == 0, f"InitLoad_{v}"

    # Load change at each region node
    for v in V:
        for r in R:
            j = rn(r)
            # Net scooter change at region r: pickups - dropoffs
            net_pickup = pulp.lpSum(p[(r, l)] for l in R if l != r)
            net_dropoff = pulp.lpSum(p[(l, r)] for l in R if l != r)
            for i in N:
                if i != j:
                    prob += (
                        q[(v, j)] >= q[(v, i)] + net_pickup - net_dropoff
                        - M * (1 - w[(v, i, j)]),
                        f"LoadTrack_{v}_{i}_{j}_lb",
                    )
                    prob += (
                        q[(v, j)] <= q[(v, i)] + net_pickup - net_dropoff
                        + M * (1 - w[(v, i, j)]),
                        f"LoadTrack_{v}_{i}_{j}_ub",
                    )

    # Constraint 17: q_vi ≤ C_s (already in variable bounds)

    # Constraint 18: Battery capacity — Σ_r z_rv * x_r ≤ C_b
    # This is non-linear (z * x), linearize with auxiliary variables
    for v in V:
        # Use: battery_load_v = Σ_r z_rv * x_r
        # Linearize: create aux_rv ≤ x_r, aux_rv ≤ M*z_rv, aux_rv ≥ x_r - M*(1-z_rv)
        aux = {}
        for r in R:
            aux[r] = pulp.LpVariable(f"aux_bat_{v}_{r}", 0, s_check[r])
            prob += aux[r] <= x[r], f"AuxBat_ub1_{v}_{r}"
            prob += aux[r] <= M * z[(r, v)], f"AuxBat_ub2_{v}_{r}"
            prob += aux[r] >= x[r] - M * (1 - z[(r, v)]), f"AuxBat_lb_{v}_{r}"

        prob += (
            pulp.lpSum(aux[r] for r in R) <= C_b,
            f"BatteryCapacity_{v}",
        )

    # Constraint 19: Delayed readiness — h_var_r ≥ x_r + Σ_l p_lr - Σ_l p_rl
    for r in R:
        incoming = pulp.lpSum(p[(l, r)] for l in R if l != r)
        outgoing = pulp.lpSum(p[(r, l)] for l in R if l != r)
        prob += (
            h_var[r] >= x[r] + incoming - outgoing,
            f"DelayBasis_{r}",
        )

    # Constraint 20: Demand balance
    # Ŝ_r + x_r + Σ_l p_lr - Σ_l p_rl + d_r ≥ D_r
    for r in R:
        incoming = pulp.lpSum(p[(l, r)] for l in R if l != r)
        outgoing = pulp.lpSum(p[(r, l)] for l in R if l != r)
        prob += (
            s_hat[r] + x[r] + incoming - outgoing + d[r] >= demand[r],
            f"DemandBalance_{r}",
        )

    if solver_name.upper() == "GUROBI":
        try:
            solver = pulp.GUROBI(msg=1, timeLimit=time_limit)
        except Exception:
            logger.warning("Gurobi not available, falling back to CBC")
            solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit)
    else:
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit)

    logger.info("Solving MILP with %s (time limit: %ds)...", solver_name, time_limit)
    prob.solve(solver)

    solve_time = time.time() - start_time
    status = pulp.LpStatus[prob.status]
    obj_value = pulp.value(prob.objective) if prob.status == 1 else None

    logger.info("MILP status: %s, objective: %s, solve time: %.1fs",
                status, obj_value, solve_time)

    x_sol = [int(round(x[r].varValue or 0)) for r in R]
    p_sol = [[0] * n_regions for _ in range(n_regions)]
    for r in R:
        for l in R:
            if r != l:
                p_sol[r][l] = int(round(p[(r, l)].varValue or 0))

    # Extract vehicle routes from w variables
    vehicle_routes: list[VehicleRoute] = []
    assignments: dict[int, int] = {}

    for v in V:
        route: list[int] = []
        if (u[v].varValue or 0) < 0.5:
            vehicle_routes.append(VehicleRoute(vehicle_id=v))
            continue

        current = 0
        visited_in_route: set[int] = set()
        for _ in range(N_nodes + 1):
            next_node = None
            for j in N:
                if j != current and (w[(v, current, j)].varValue or 0) > 0.5:
                    next_node = j
                    break
            if next_node is None or next_node == 0:
                break
            region_idx = next_node - 1
            if region_idx in visited_in_route:
                break  # safety
            route.append(region_idx)
            visited_in_route.add(region_idx)
            assignments[region_idx] = v
            current = next_node

        vehicle_routes.append(VehicleRoute(vehicle_id=v, route=route))

    plan = Plan(
        x=x_sol,
        p=p_sol,
        vehicle_routes=vehicle_routes,
        vehicle_assignments=assignments,
    )

    solver_info = {
        "status": status,
        "objective_value": obj_value,
        "solve_time_seconds": solve_time,
        "solver": solver_name,
        "gap": None,
    }

    return plan, solver_info


# CLI entry point
if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Solve MILP baseline")
    parser.add_argument("--instance", type=str,
                        default="configs/instance.json")
    parser.add_argument("--economics", type=str,
                        default="configs/economics.yaml")
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--solver", type=str, default="CBC",
                        choices=["CBC", "GUROBI"])
    args = parser.parse_args()

    with open(args.instance, "r") as f:
        inst = json.load(f)
    with open(args.economics, "r") as f:
        econ = yaml.safe_load(f)

    plan, info = solve_milp(inst, econ, args.time_limit, args.solver)
    result = simulate_plan(plan, inst, econ)

    print(f"\n{'='*60}")
    print(f"MILP Baseline Results")
    print(f"{'='*60}")
    print(f"Solver status:    {info['status']}")
    print(f"MILP objective:   {info['objective_value']:.4f}")
    print(f"Sim  objective:   {result.objective_z:.4f}")
    print(f"Travel cost:      {result.travel_cost:.4f}")
    print(f"Unmet demand:     {result.total_unmet_demand:.1f}")
    print(f"Delay penalty:    {result.delay_penalty:.4f}")
    print(f"Solve time:       {info['solve_time_seconds']:.1f}s")
    print(f"Violations:       {result.feasibility_violations}")
    print(f"\nSwaps (x):        {plan.x}")
    print(f"Routes:")
    for vr in plan.vehicle_routes:
        print(f"  Vehicle {vr.vehicle_id}: Hub → {' → '.join(f'R{r}' for r in vr.route)} → Hub")
    print(f"{'='*60}")

"""Stateless plan simulator — executes a complete plan and computes objective Z (Eq. 1)."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VehicleRoute:
    vehicle_id: int
    route: list[int] = field(default_factory=list)
    departure_times: dict[int, float] = field(default_factory=dict)
    arrival_times: dict[int, float] = field(default_factory=dict)


@dataclass
class Plan:
    """A complete allocation + routing plan.

    x[r]    = number of battery swaps in region r
    p[r][l] = scooters relocated from r to l
    """
    x: list[int]
    p: list[list[int]]
    vehicle_routes: list[VehicleRoute]
    vehicle_assignments: dict[int, int] = field(default_factory=dict)


@dataclass
class PlanResult:
    objective_z: float
    travel_cost: float
    unmet_demand_penalty: float
    delay_penalty: float
    total_travel_time: float
    per_region_unmet: list[float]
    per_region_delay: list[float]
    total_unmet_demand: float
    demand_fulfillment_rate: float
    vehicle_routes: list[VehicleRoute]
    feasibility_violations: list[str] = field(default_factory=list)


def euclidean_distance(xy_i: tuple[float, float], xy_j: tuple[float, float]) -> float:
    return math.sqrt((xy_i[0] - xy_j[0]) ** 2 + (xy_i[1] - xy_j[1]) ** 2)


def build_travel_time_matrix(
    hub_xy: tuple[float, float],
    region_coords: list[tuple[float, float]],
) -> list[list[float]]:
    """Node 0 = hub, node k = region k-1."""
    all_coords = [hub_xy] + region_coords
    n = len(all_coords)
    return [
        [euclidean_distance(all_coords[i], all_coords[j]) for j in range(n)]
        for i in range(n)
    ]


def compute_processing_time(
    region_idx: int,
    x: list[int],
    p: list[list[int]],
    swap_time: float,
    reloc_time: float,
) -> float:
    """T_r = swap_time * x_r + reloc_time * (pickups + dropoffs at r)"""
    n = len(x)
    pickups  = sum(p[region_idx][l] for l in range(n) if l != region_idx)
    dropoffs = sum(p[l][region_idx] for l in range(n) if l != region_idx)
    return x[region_idx] * swap_time + (pickups + dropoffs) * reloc_time


def simulate_plan(
    plan: Plan,
    instance: dict[str, Any],
    economics: dict[str, Any],
) -> PlanResult:
    """Ground-truth scorer. Returns full Z = λ·travel + λ'·unmet + λ''·delay."""
    violations: list[str] = []

    n_regions    = instance["region_count"]
    regions      = instance["regions"]
    hub_xy       = (instance["hub"]["x"], instance["hub"]["y"])
    region_coords = [(r["x"], r["y"]) for r in regions]
    demand       = [r["demand"]         for r in regions]
    s_hat        = [r["functional"]     for r in regions]
    s_check      = [r["non_functional"] for r in regions]
    extra_batt   = instance["extra_batteries"]
    battery_cap  = instance["battery_carrying_capacity"]
    scooter_cap  = instance["scooter_carrying_capacity"]

    lam      = economics.get("lambda_travel", 1.0)
    lam_un   = economics.get("lambda_unmet", 50.0)
    period   = economics.get("period_length", 60)
    lam_d    = lam_un / period
    swap_t   = economics.get("swap_time_min", 1.5)
    reloc_t  = economics.get("reloc_time_min", 1.0)
    urg_mode = economics.get("urgency_weighting", "demand_proportional")

    tt = build_travel_time_matrix(hub_xy, region_coords)

    x = plan.x
    p = plan.p

    for r in range(n_regions):
        if x[r] > s_check[r]:
            violations.append(f"Constraint 2: x[{r}]={x[r]} > S̆[{r}]={s_check[r]}")
        total_out = sum(p[r][l] for l in range(n_regions) if l != r)
        if total_out > x[r] + s_hat[r]:
            violations.append(
                f"Constraint 4: reloc_from[{r}]={total_out} > x[{r}]+Ŝ[{r}]={x[r]+s_hat[r]}"
            )

    if sum(x) > extra_batt:
        violations.append(f"Constraint 3: Σx={sum(x)} > A={extra_batt}")

    total_travel = 0.0
    sim_routes: list[VehicleRoute] = []
    region_arrival: dict[int, float] = {}

    for vr in plan.vehicle_routes:
        route = vr.route
        if not route:
            sim_routes.append(VehicleRoute(vehicle_id=vr.vehicle_id))
            continue

        arr_times: dict[int, float] = {}
        dep_times: dict[int, float] = {}
        t = 0.0
        prev = 0  # hub

        for rid in route:
            node = rid + 1
            travel = tt[prev][node]
            total_travel += travel
            t += travel

            arr_times[rid] = t
            region_arrival[rid] = min(region_arrival.get(rid, t), t)

            proc = compute_processing_time(rid, x, p, swap_t, reloc_t)
            t += proc
            dep_times[rid] = t
            prev = node

        total_travel += tt[prev][0]  # return to hub
        sim_routes.append(VehicleRoute(
            vehicle_id=vr.vehicle_id,
            route=route,
            arrival_times=arr_times,
            departure_times=dep_times,
        ))

    per_region_unmet: list[float] = []
    for r in range(n_regions):
        incoming = sum(p[l][r] for l in range(n_regions) if l != r)
        outgoing = sum(p[r][l] for l in range(n_regions) if l != r)
        available = s_hat[r] + x[r] + incoming - outgoing
        per_region_unmet.append(max(0.0, demand[r] - available))

    total_unmet = sum(per_region_unmet)

    per_region_delay: list[float] = []
    for r in range(n_regions):
        h_r = float(demand[r]) if urg_mode == "demand_proportional" else 1.0
        m_vr = region_arrival.get(r, 0.0)
        incoming = sum(p[l][r] for l in range(n_regions) if l != r)
        outgoing = sum(p[r][l] for l in range(n_regions) if l != r)
        net_svc = x[r] + incoming - outgoing
        per_region_delay.append(h_r * m_vr if net_svc > 0 and m_vr > 0 else 0.0)

    travel_cost = lam * total_travel
    unmet_pen   = lam_un * total_unmet
    delay_pen   = lam_d  * sum(per_region_delay)
    obj_z       = travel_cost + unmet_pen + delay_pen

    if violations:
        logger.warning("Feasibility violations: %s", violations)

    total_demand = sum(demand)
    fulfillment  = 1.0 - (total_unmet / total_demand) if total_demand > 0 else 1.0

    return PlanResult(
        objective_z=obj_z,
        travel_cost=travel_cost,
        unmet_demand_penalty=unmet_pen,
        delay_penalty=delay_pen,
        total_travel_time=total_travel,
        per_region_unmet=per_region_unmet,
        per_region_delay=per_region_delay,
        total_unmet_demand=total_unmet,
        demand_fulfillment_rate=fulfillment,
        vehicle_routes=sim_routes,
        feasibility_violations=violations,
    )

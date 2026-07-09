"""Training environment for the E-Scooter Battery Swap & Relocation problem """
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from env.masks import (
    compute_swap_mask,
    compute_relocation_mask,
    compute_routing_mask,
)
from env.simulator import (
    Plan,
    PlanResult,
    VehicleRoute,
    build_travel_time_matrix,
    compute_processing_time,
    simulate_plan,
)

logger = logging.getLogger(__name__)


class Phase(Enum):
    """Current phase of the episode."""
    SWAP_ALLOCATION = auto()
    RELOCATION_ALLOCATION = auto()
    ROUTING = auto()
    DONE = auto()


@dataclass
class VehicleState:
    vehicle_id: int
    current_node: int = 0  # hub
    scooter_load: int = 0
    battery_load: int = 0
    departure_time: float = 0.0
    route: list[int] = field(default_factory=list)
    visited_pickups: set[int] = field(default_factory=set)
    active: bool = True


class BatterySwapEnv(gym.Env):
    """environment for E-Scooter Battery Swap & Relocation."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        instance: dict[str, Any],
        economics: dict[str, Any],
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.instance = instance
        self.economics = economics
        self._seed = seed

        self.n_regions: int = instance["region_count"]
        self.n_vehicles: int = instance["vehicle_count"]
        self.extra_batteries: int = instance["extra_batteries"]
        self.battery_cap: int = instance["battery_carrying_capacity"]
        self.scooter_cap: int = instance["scooter_carrying_capacity"]
        self.battery_threshold: int = instance["battery_threshold"]

        self.hub_xy: tuple[float, float] = (instance["hub"]["x"], instance["hub"]["y"])
        self.region_data: list[dict] = instance["regions"]
        self.region_coords: list[tuple[float, float]] = [
            (r["x"], r["y"]) for r in self.region_data
        ]
        self.demand: list[int] = [r["demand"] for r in self.region_data]
        self.s_hat: list[int] = [r["functional"] for r in self.region_data]
        self.s_check: list[int] = [r["non_functional"] for r in self.region_data]

        self.lambda_travel: float = economics.get("lambda_travel", 1.0)
        self.lambda_unmet: float = economics.get("lambda_unmet", 50.0)
        self.period_length: float = economics.get("period_length", 60)
        self.lambda_delay: float = self.lambda_unmet / self.period_length
        self.swap_time: float = economics.get("swap_time_min", 1.5)
        self.reloc_time: float = economics.get("reloc_time_min", 1.0)
        self.reward_mode: str = economics.get("reward_mode", "shaped")

        self.travel_time_matrix = build_travel_time_matrix(
            self.hub_xy, self.region_coords
        )

        self.phase = Phase.DONE
        self.x: list[int] = []
        self.p: list[list[int]] = []
        self.a_remaining: int = 0
        self.current_swap_region: int = 0
        self.current_reloc_src: int = 0
        self.current_reloc_dst: int = 0
        self.reloc_committed: list[int] = []  # per-region total relocated so far

        self.vehicles: list[VehicleState] = []
        self.current_vehicle_idx: int = 0
        self.claimed_regions: set[int] = set()
        self.jobs: list[dict] = []  # job list from allocation
        self.pdp_pairs: list[tuple[int, int]] = []

        # Tracking
        self.step_count: int = 0
        self.cumulative_travel: float = 0.0
        self.episode_log_probs: list[float] = []
        self.episode_values: list[float] = []
        self.episode_rewards: list[float] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        
        super().reset(seed=seed or self._seed)

        # Allow instance override via options
        if options and "instance" in options:
            self._load_instance(options["instance"])

        self.phase = Phase.SWAP_ALLOCATION
        self.x = [0] * self.n_regions
        self.p = [[0] * self.n_regions for _ in range(self.n_regions)]
        self.a_remaining = self.extra_batteries
        self.current_swap_region = 0
        self.current_reloc_src = 0
        self.current_reloc_dst = 0
        self.reloc_committed = [0] * self.n_regions

        self.vehicles = [
            VehicleState(vehicle_id=v) for v in range(self.n_vehicles)
        ]
        self.current_vehicle_idx = 0
        self.claimed_regions = set()
        self.jobs = []
        self.pdp_pairs = []

        self.step_count = 0
        self.cumulative_travel = 0.0
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_rewards = []

        return self._get_obs(), self._get_info()

    def _load_instance(self, instance: dict[str, Any]) -> None:
       # Reloading here
        self.instance = instance
        self.n_regions = instance["region_count"]
        self.n_vehicles = instance["vehicle_count"]
        self.extra_batteries = instance["extra_batteries"]
        self.battery_cap = instance["battery_carrying_capacity"]
        self.scooter_cap = instance["scooter_carrying_capacity"]
        self.hub_xy = (instance["hub"]["x"], instance["hub"]["y"])
        self.region_data = instance["regions"]
        self.region_coords = [(r["x"], r["y"]) for r in self.region_data]
        self.demand = [r["demand"] for r in self.region_data]
        self.s_hat = [r["functional"] for r in self.region_data]
        self.s_check = [r["non_functional"] for r in self.region_data]
        self.travel_time_matrix = build_travel_time_matrix(
            self.hub_xy, self.region_coords
        )

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        truncated = False

        if self.phase == Phase.SWAP_ALLOCATION:
            reward = self._step_swap(action)
        elif self.phase == Phase.RELOCATION_ALLOCATION:
            reward = self._step_relocation(action)
        elif self.phase == Phase.ROUTING:
            reward, terminated = self._step_routing(action)
        else:
            raise RuntimeError("Episode already done; call reset().")

        self.step_count += 1
        if self.reward_mode == "terminal" and not terminated:
            reward = 0.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # Phase 1a: Swap allocation

    def _step_swap(self, action: int) -> float:
        r = self.current_swap_region
        self.x[r] = action
        self.a_remaining -= action

        reward = 0.0

        # Advance to next region
        self.current_swap_region += 1
        if self.current_swap_region >= self.n_regions:
            # All swaps done — move to relocation phase
            self.phase = Phase.RELOCATION_ALLOCATION
            self.current_reloc_src = 0
            self.current_reloc_dst = self._next_reloc_dst(0, -1)

            if self.current_reloc_dst >= self.n_regions:
                # No relocation destinations for src=0, advance src
                self._advance_reloc_src()

            # Shaped reward: bonus for using budget wisely
            if self.reward_mode == "shaped":
                used = self.extra_batteries - self.a_remaining
                max_useful = sum(
                    min(self.s_check[r_], self.demand[r_]) for r_ in range(self.n_regions)
                )
                if max_useful > 0:
                    reward = 0.1 * (used / max_useful)

        return reward

    # Phase 1b: Relocation allocation

    def _step_relocation(self, action: int) -> float:
        src = self.current_reloc_src
        dst = self.current_reloc_dst
        self.p[src][dst] = action
        self.reloc_committed[src] += action

        reward = 0.0
        if self.reward_mode == "shaped" and action > 0:
            # Small bonus for relocating to regions with unmet demand
            surplus_at_src = self.s_hat[src] + self.x[src] - self.demand[src]
            deficit_at_dst = max(0, self.demand[dst] - self.s_hat[dst] - self.x[dst])
            if surplus_at_src > 0 and deficit_at_dst > 0:
                reward = 0.05 * min(action, deficit_at_dst)

        # Advance to next (src, dst) pair
        next_dst = self._next_reloc_dst(src, dst)
        if next_dst >= self.n_regions:
            self._advance_reloc_src()
        else:
            self.current_reloc_dst = next_dst

        return reward

    def _next_reloc_dst(self, src: int, current_dst: int) -> int:
        for d in range(current_dst + 1, self.n_regions):
            if d != src:
                return d
        return self.n_regions

    def _advance_reloc_src(self) -> None:
        for s in range(self.current_reloc_src + 1, self.n_regions):
            first_dst = self._next_reloc_dst(s, -1)
            if first_dst < self.n_regions:
                self.current_reloc_src = s
                self.current_reloc_dst = first_dst
                self.reloc_committed[s] = 0
                return

        # All relocation done — transition to routing
        self._transition_to_routing()

    def _transition_to_routing(self) -> None:
        self.phase = Phase.ROUTING
        self.jobs = []
        self.pdp_pairs = []

        # Swap jobs: one per region with x_r > 0
        for r in range(self.n_regions):
            if self.x[r] > 0:
                self.jobs.append({
                    "type": "swap",
                    "region": r,
                    "batteries": self.x[r],
                })

        # Pickup-delivery jobs: one pair per (r,l) with p_rl > 0
        for r in range(self.n_regions):
            for l in range(self.n_regions):
                if r != l and self.p[r][l] > 0:
                    self.pdp_pairs.append((r, l))
                    self.jobs.append({
                        "type": "pickup",
                        "region": r,
                        "linked_delivery": l,
                        "count": self.p[r][l],
                    })
                    self.jobs.append({
                        "type": "delivery",
                        "region": l,
                        "linked_pickup": r,
                        "count": self.p[r][l],
                    })

        # If no jobs, episode ends immediately
        if not self.jobs:
            self.phase = Phase.DONE

        self.current_vehicle_idx = 0

    # Phase 2: Routing

    def _step_routing(self, action: int) -> tuple[float, bool]:
        vehicle = self.vehicles[self.current_vehicle_idx]

        if action == -1:
            # Return to hub
            if vehicle.route:
                prev_node = vehicle.route[-1] + 1  # 1-indexed in tt matrix
                travel = self.travel_time_matrix[prev_node][0]
                self.cumulative_travel += travel
                vehicle.departure_time += travel

            vehicle.active = False
            return self._advance_vehicle()

        # Visit region `action`
        region = action
        prev_node = 0 if not vehicle.route else (vehicle.route[-1] + 1)
        next_node = region + 1  # 1-indexed in tt matrix

        travel = self.travel_time_matrix[prev_node][next_node]
        self.cumulative_travel += travel
        vehicle.departure_time += travel

        # Processing time
        proc = compute_processing_time(
            region, self.x, self.p, self.swap_time, self.reloc_time
        )
        vehicle.departure_time += proc

        # Update vehicle state
        vehicle.route.append(region)
        self.claimed_regions.add(region)

        # Track battery load
        vehicle.battery_load += self.x[region]

        # Track scooter load for PDP
        for r, l in self.pdp_pairs:
            if r == region and r not in vehicle.visited_pickups:
                vehicle.scooter_load += self.p[r][l]
                vehicle.visited_pickups.add(r)
            elif l == region and r in vehicle.visited_pickups:
                vehicle.scooter_load -= self.p[r][l]

        # Shaped reward: negative travel cost per step
        reward = 0.0
        if self.reward_mode == "shaped":
            reward = -self.lambda_travel * travel

        # Check if current vehicle should continue or we advance
        mask = self.get_routing_mask()
        if not any(v for v in mask.values()):
            # No feasible nodes — return to hub
            if vehicle.route:
                last_node = vehicle.route[-1] + 1
                travel_back = self.travel_time_matrix[last_node][0]
                self.cumulative_travel += travel_back
                vehicle.departure_time += travel_back

            vehicle.active = False
            r, terminated = self._advance_vehicle()
            return reward + r, terminated

        return reward, False

    def _advance_vehicle(self) -> tuple[float, bool]:
        # find next active vehicle
        for v_idx in range(self.current_vehicle_idx + 1, self.n_vehicles):
            self.current_vehicle_idx = v_idx
            vehicle = self.vehicles[v_idx]
            if vehicle.active:
                mask = self.get_routing_mask()
                if any(v for v in mask.values()):
                    return 0.0, False
                else:
                    vehicle.active = False

        # All vehicles done — compute terminal reward
        self.phase = Phase.DONE
        return self._compute_terminal_reward()

    def _compute_terminal_reward(self) -> tuple[float, bool]:
        plan = self._build_plan()
        result = simulate_plan(plan, self.instance, self.economics)

        if self.reward_mode == "terminal":
            reward = -result.objective_z
        else:
            # Shaped mode: terminal part = unmet demand + delay penalties
            # (travel was already given step by step)
            reward = -(result.unmet_demand_penalty + result.delay_penalty)

        self._last_result = result
        return reward, True

    # Observation and mask construction

    def _get_obs(self) -> dict[str, Any]:
        
        # Node features: [D_r, Ŝ_r, S̆_r, x, y, budget_flag, visited_flag]
        node_features = []

        # Hub node
        node_features.append([
            float(self.a_remaining), 0.0, 0.0,
            self.hub_xy[0], self.hub_xy[1], 0.0, 0.0,
        ])

        # Region nodes
        for r in range(self.n_regions):
            visited = 1.0 if r in self.claimed_regions else 0.0
            budget_flag = 1.0 if self.x[r] > 0 else 0.0
            node_features.append([
                float(self.demand[r]),
                float(self.s_hat[r]),
                float(self.s_check[r]),
                self.region_coords[r][0],
                self.region_coords[r][1],
                budget_flag,
                visited,
            ])

        # Vehicle context vectors
        vehicle_contexts = []
        for v in self.vehicles:
            vehicle_contexts.append({
                "vehicle_id": v.vehicle_id,
                "current_node": v.current_node,
                "scooter_load": v.scooter_load,
                "battery_load": v.battery_load,
                "departure_time": v.departure_time,
                "scooter_remaining": self.scooter_cap - v.scooter_load,
                "battery_remaining": self.battery_cap - v.battery_load,
                "active": v.active,
            })

        return {
            "node_features": np.array(node_features, dtype=np.float32),
            "travel_time_matrix": np.array(self.travel_time_matrix, dtype=np.float32),
            "vehicle_contexts": vehicle_contexts,
            "a_remaining": self.a_remaining,
            "phase": self.phase.name,
            "current_swap_region": self.current_swap_region if self.phase == Phase.SWAP_ALLOCATION else -1,
            "current_reloc_src": self.current_reloc_src if self.phase == Phase.RELOCATION_ALLOCATION else -1,
            "current_reloc_dst": self.current_reloc_dst if self.phase == Phase.RELOCATION_ALLOCATION else -1,
            "current_vehicle_idx": self.current_vehicle_idx if self.phase == Phase.ROUTING else -1,
            "x": list(self.x),
            "p": [list(row) for row in self.p],
            "pdp_pairs": list(self.pdp_pairs),
        }

    def _get_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "phase": self.phase.name,
            "step_count": self.step_count,
            "a_remaining": self.a_remaining,
            "cumulative_travel": self.cumulative_travel,
        }
        if self.phase == Phase.DONE and hasattr(self, "_last_result"):
            info["plan_result"] = self._last_result
        return info

    def get_action_mask(self) -> list[bool] | dict[int, bool]:
        if self.phase == Phase.SWAP_ALLOCATION:
            return self.get_swap_mask()
        elif self.phase == Phase.RELOCATION_ALLOCATION:
            return self.get_relocation_mask()
        elif self.phase == Phase.ROUTING:
            return self.get_routing_mask()
        else:
            return []

    def get_swap_mask(self) -> list[bool]:
        r = self.current_swap_region
        if r >= self.n_regions:
            return []
        return compute_swap_mask(
            s_check_r=self.s_check[r],
            a_remaining=self.a_remaining,
        )

    def get_relocation_mask(self) -> list[bool]:
        src = self.current_reloc_src
        if src >= self.n_regions:
            return []
        return compute_relocation_mask(
            x_r=self.x[src],
            s_hat_r=self.s_hat[src],
            already_relocated=self.reloc_committed[src],
            max_possible=self.scooter_cap,
        )

    def get_routing_mask(self) -> dict[int, bool]:
        if self.phase != Phase.ROUTING:
            return {}

        vehicle = self.vehicles[self.current_vehicle_idx]
        if not vehicle.active:
            return {}

        # Build eligible node list (all regions with jobs not yet claimed)
        job_regions = set()
        for job in self.jobs:
            region = job["region"]
            if region not in self.claimed_regions:
                job_regions.add(region)
            elif job["type"] == "delivery" and job["linked_pickup"] in vehicle.visited_pickups:
                # Delivery whose pickup was done by this vehicle
                if region not in self.claimed_regions or region in [
                    j["region"] for j in self.jobs
                    if j.get("linked_pickup") in vehicle.visited_pickups
                    and j["type"] == "delivery"
                ]:
                    job_regions.add(region)

        eligible = list(job_regions)
        if not eligible:
            return {-1: True}  # only hub return

        # Compute node deltas for capacity checks
        node_scooter_delta: dict[int, int] = {}
        node_battery_delta: dict[int, int] = {}

        for region in eligible:
            scooter_delta = 0
            battery_delta = 0

            # Pickup: load scooters
            for r, l in self.pdp_pairs:
                if r == region and r not in vehicle.visited_pickups:
                    scooter_delta += self.p[r][l]
                elif l == region and r in vehicle.visited_pickups:
                    scooter_delta -= self.p[r][l]

            # Battery: swaps assigned to this region
            if self.x[region] > 0 and region not in self.claimed_regions:
                battery_delta = self.x[region]

            node_scooter_delta[region] = scooter_delta
            node_battery_delta[region] = battery_delta

        # Check if all nodes are claimed (no eligible jobs remain for any vehicle)
        no_eligible = len(eligible) == 0

        mask = compute_routing_mask(
            eligible_nodes=eligible,
            node_scooter_delta=node_scooter_delta,
            node_battery_delta=node_battery_delta,
            current_scooter_load=vehicle.scooter_load,
            current_battery_load=vehicle.battery_load,
            scooter_capacity=self.scooter_cap,
            battery_capacity=self.battery_cap,
            visited_pickups_by_vehicle=vehicle.visited_pickups,
            pdp_pairs=self.pdp_pairs,
            claimed_by_other_vehicle={
                r for r in self.claimed_regions
                if r not in {n for v in self.vehicles if v.vehicle_id == vehicle.vehicle_id for n in v.route}
            },
            allow_hub_return=no_eligible,
        )

        return mask

    # Plan construction

    def _build_plan(self) -> Plan:
        vehicle_routes = []
        assignments: dict[int, int] = {}

        for v in self.vehicles:
            vr = VehicleRoute(
                vehicle_id=v.vehicle_id,
                route=list(v.route),
            )
            vehicle_routes.append(vr)
            for r in v.route:
                assignments[r] = v.vehicle_id

        return Plan(
            x=list(self.x),
            p=[list(row) for row in self.p],
            vehicle_routes=vehicle_routes,
            vehicle_assignments=assignments,
        )

    def get_plan_result(self) -> PlanResult | None:
        return getattr(self, "_last_result", None)


def make_env(
    instance_path: str | Path | None = None,
    economics_path: str | Path | None = None,
    instance: dict | None = None,
    economics: dict | None = None,
    seed: int = 42,
) -> BatterySwapEnv:
    if instance is None:
        if instance_path is None:
            instance_path = Path(__file__).parent.parent / "configs" / "instance.json"
        with open(instance_path, "r", encoding="utf-8") as f:
            instance = json.load(f)

    if economics is None:
        if economics_path is None:
            economics_path = Path(__file__).parent.parent / "configs" / "economics.yaml"
        with open(economics_path, "r", encoding="utf-8") as f:
            economics = yaml.safe_load(f)

    return BatterySwapEnv(instance=instance, economics=economics, seed=seed)

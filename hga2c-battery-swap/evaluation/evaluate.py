"""Full evaluation protocol (§8) — runs all methods and writes results to paper/tables."""
from __future__ import annotations

import argparse
import json
import logging
import time
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from baselines.exact_milp import solve_milp
from baselines.nearest_neighbor import solve_nearest_neighbor
from baselines.ortools_baseline import solve_ortools
from baselines.random_policy import solve_random
from baselines.legacy_heuristic import solve_legacy_heuristic
from env.battery_swap_env import make_env
from env.simulator import Plan, VehicleRoute, simulate_plan
from models.hga2c_policy import build_policy_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_hga2c(
    instance: dict[str, Any],
    economics: dict[str, Any],
    checkpoint_path: str,
    hyperparams: dict[str, Any],
    n_rollouts: int = 30,
    greedy: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    import torch
    torch.manual_seed(seed)

    policy = build_policy_from_config(hyperparams)
    if Path(checkpoint_path).exists():
        policy.load_checkpoint(checkpoint_path)
    policy.eval()

    env = make_env(instance=instance, economics=economics, seed=seed)
    results = []

    for i in range(n_rollouts):
        start = time.time()
        obs, _ = env.reset(seed=seed + i)
        with torch.no_grad():
            output = policy.forward(obs, instance, economics, greedy=greedy)

        vehicle_routes_obj = [
            VehicleRoute(vehicle_id=v, route=r)
            for v, r in enumerate(output["vehicle_routes"])
        ]
        plan = Plan(
            x=output["x"], p=output["p"],
            vehicle_routes=vehicle_routes_obj,
            vehicle_assignments={
                r: v for v, route in enumerate(output["vehicle_routes"]) for r in route
            },
        )
        sim_result = simulate_plan(plan, instance, economics)
        elapsed = time.time() - start

        results.append({
            "objective_z": sim_result.objective_z,
            "unmet_demand": sim_result.total_unmet_demand,
            "demand_fulfillment_rate": sim_result.demand_fulfillment_rate,
            "travel_cost": sim_result.travel_cost,
            "delay_penalty": sim_result.delay_penalty,
            "inference_time": elapsed,
            "violations": len(sim_result.feasibility_violations),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all methods")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--milp-time-limit", type=int, default=600)
    args = parser.parse_args()

    with open("configs/instance.json") as f:
        instance = json.load(f)
    with open("configs/economics.yaml") as f:
        economics = yaml.safe_load(f)
    with open("configs/hyperparams.yaml") as f:
        hyperparams = yaml.safe_load(f)

    results_summary: dict[str, dict[str, Any]] = {}
    raw_results = []

    # 1. MILP (Deterministic)
    logger.info("Running MILP baseline...")
    start = time.time()
    plan_milp, milp_info = solve_milp(instance, economics, args.milp_time_limit)
    sim_milp = simulate_plan(plan_milp, instance, economics)
    milp_time = time.time() - start
    milp_z = sim_milp.objective_z
    
    results_summary["MILP"] = {
        "objective_z": sim_milp.objective_z,
        "demand_fulfillment_rate": sim_milp.demand_fulfillment_rate,
        "inference_time": milp_time,
        "optimality_gap": 0.0,
    }
    raw_results.append({
        "Method": "MILP", "Seed": args.seeds[0], "Objective": sim_milp.objective_z,
        "Fulfillment": sim_milp.demand_fulfillment_rate, "Time": milp_time, "Gap": 0.0
    })

    # 2. Nearest Neighbor (Deterministic)
    logger.info("Running Nearest Neighbor...")
    start = time.time()
    plan_nn, _ = solve_nearest_neighbor(instance, economics)
    sim_nn = simulate_plan(plan_nn, instance, economics)
    nn_time = time.time() - start
    nn_gap = (sim_nn.objective_z - milp_z) / milp_z if milp_z > 0 else None
    
    results_summary["NearestNeighbor"] = {
        "objective_z": sim_nn.objective_z,
        "demand_fulfillment_rate": sim_nn.demand_fulfillment_rate,
        "inference_time": nn_time,
        "optimality_gap": nn_gap,
    }
    raw_results.append({
        "Method": "NearestNeighbor", "Seed": args.seeds[0], "Objective": sim_nn.objective_z,
        "Fulfillment": sim_nn.demand_fulfillment_rate, "Time": nn_time, "Gap": nn_gap
    })
    
    # 3. Legacy Heuristic (Deterministic)
    logger.info("Running Legacy Heuristic...")
    start = time.time()
    plan_leg, sim_leg = solve_legacy_heuristic(instance, economics)
    leg_time = time.time() - start
    leg_gap = (sim_leg["objective_z"] - milp_z) / milp_z if milp_z > 0 else None
    
    results_summary["LegacyHeuristic"] = {
        "objective_z": sim_leg["objective_z"],
        "demand_fulfillment_rate": sim_leg["demand_fulfillment_rate"],
        "inference_time": leg_time,
        "optimality_gap": leg_gap,
    }
    raw_results.append({
        "Method": "LegacyHeuristic", "Seed": args.seeds[0], "Objective": sim_leg["objective_z"],
        "Fulfillment": sim_leg["demand_fulfillment_rate"], "Time": leg_time, "Gap": leg_gap
    })

    # 4. OR-Tools (Deterministic)
    logger.info("Running OR-Tools...")
    start = time.time()
    plan_ort, _ = solve_ortools(instance, economics, time_limit_seconds=60)
    sim_ort = simulate_plan(plan_ort, instance, economics)
    ort_time = time.time() - start
    ort_gap = (sim_ort.objective_z - milp_z) / milp_z if milp_z > 0 else None
    
    results_summary["OR-Tools"] = {
        "objective_z": sim_ort.objective_z,
        "demand_fulfillment_rate": sim_ort.demand_fulfillment_rate,
        "inference_time": ort_time,
        "optimality_gap": ort_gap,
    }
    raw_results.append({
        "Method": "OR-Tools", "Seed": args.seeds[0], "Objective": sim_ort.objective_z,
        "Fulfillment": sim_ort.demand_fulfillment_rate, "Time": ort_time, "Gap": ort_gap
    })

    # 5. Random (Stochastic over seeds)
    rnd_zs = []
    rnd_fs = []
    for seed in args.seeds:
        plan_rnd, _ = solve_random(instance, economics, seed=seed)
        sim_rnd = simulate_plan(plan_rnd, instance, economics)
        gap = (sim_rnd.objective_z - milp_z) / milp_z if milp_z > 0 else None
        raw_results.append({
            "Method": "Random", "Seed": seed, "Objective": sim_rnd.objective_z,
            "Fulfillment": sim_rnd.demand_fulfillment_rate, "Time": 0.0, "Gap": gap
        })
        rnd_zs.append(sim_rnd.objective_z)
        rnd_fs.append(sim_rnd.demand_fulfillment_rate)
        
    results_summary["Random"] = {
        "objective_z": np.mean(rnd_zs),
        "objective_z_std": np.std(rnd_zs),
        "demand_fulfillment_rate": np.mean(rnd_fs),
        "optimality_gap": (np.mean(rnd_zs) - milp_z) / milp_z if milp_z > 0 else None,
        "inference_time": 0.0
    }

    # 6. HGA2C (Multi-seed evaluation)
    hga2c_zs = []
    hga2c_fs = []
    hga2c_times = []
    for seed in args.seeds:
        ckpt_path = f"checkpoints/seed_{seed}/stage3_final.pt"
        if not Path(ckpt_path).exists():
            logger.warning(f"Skipping HGA2C evaluation for seed {seed} - Checkpoint not found.")
            continue
            
        # Greedy evaluation for headline results
        res = evaluate_hga2c(
            instance, economics, ckpt_path, hyperparams,
            n_rollouts=1, greedy=True, seed=seed
        )[0]
        
        gap = (res["objective_z"] - milp_z) / milp_z if milp_z > 0 else None
        raw_results.append({
            "Method": "HGA2C", "Seed": seed, "Objective": res["objective_z"],
            "Fulfillment": res["demand_fulfillment_rate"], "Time": res["inference_time"], "Gap": gap
        })
        hga2c_zs.append(res["objective_z"])
        hga2c_fs.append(res["demand_fulfillment_rate"])
        hga2c_times.append(res["inference_time"])
        
    if hga2c_zs:
        results_summary["HGA2C"] = {
            "objective_z": np.mean(hga2c_zs),
            "objective_z_std": np.std(hga2c_zs),
            "demand_fulfillment_rate": np.mean(hga2c_fs),
            "optimality_gap": (np.mean(hga2c_zs) - milp_z) / milp_z if milp_z > 0 else None,
            "inference_time": np.mean(hga2c_times)
        }

    # Print Headline Table
    print(f"\n{'='*95}")
    print(f"{'Method':<20} {'Objective Z':>16} {'Fulfill %':>12} {'Time (s)':>10} {'Gap %':>10}")
    print(f"{'='*95}")
    for method, data in results_summary.items():
        z = data.get("objective_z", float("inf"))
        std = data.get("objective_z_std", 0.0)
        f = data.get("demand_fulfillment_rate", 0.0)
        t = data.get("inference_time", 0.0)
        gap = data.get("optimality_gap", None)
        
        z_str = f"{z:.1f}±{std:.1f}" if std > 0 else f"{z:.1f}"
        f_str = f"{f*100:.1f}%"
        t_str = f"{t:.3f}"
        gap_str = f"{gap*100:.1f}%" if gap is not None else "N/A"
        
        print(f"{method:<20} {z_str:>16} {f_str:>12} {t_str:>10} {gap_str:>10}")
    print(f"{'='*95}")

    # Write raw CSV outputs
    out_dir = Path("paper/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "raw_evaluation_reference.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Seed", "Objective", "Fulfillment", "Time", "Gap"])
        writer.writeheader()
        writer.writerows(raw_results)
        
    logger.info("Raw evaluation metrics saved to %s", csv_file)


if __name__ == "__main__":
    main()

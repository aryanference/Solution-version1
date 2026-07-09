"""Zero-Shot Generalization Sweep.

Loops over varying instance sizes and vehicle counts across 50 held-out instances
to evaluate the scalability and generalization of the HGA²C policy without retraining.
Dumps raw results for statistical testing and plotting.
"""
import argparse
import csv
import logging
from pathlib import Path
import yaml
import numpy as np

from data.instance_generator import generate_instance
from evaluation.evaluate import evaluate_hga2c
from baselines.exact_milp import solve_milp
from baselines.ortools_baseline import solve_ortools
from baselines.nearest_neighbor import solve_nearest_neighbor
from baselines.legacy_heuristic import solve_legacy_heuristic
from env.simulator import simulate_plan

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-instances", type=int, default=50)
    parser.add_argument("--out-csv", type=str, default="paper/tables/raw_generalization_sweep.csv")
    args = parser.parse_args()

    with open("configs/economics.yaml") as f:
        economics = yaml.safe_load(f)
    with open("configs/hyperparams.yaml") as f:
        hyperparams = yaml.safe_load(f)

    # Size sweep configuration
    sizes = [
        {"r": 5, "v": 1},
        {"r": 9, "v": 2},
        {"r": 12, "v": 2},
        {"r": 15, "v": 3},
    ]
    
    raw_results = []
    
    base_seed = 10000  # Offset to ensure no overlap with training instances

    logger.info("Starting Generalization Sweep...")
    
    for size_cfg in sizes:
        r_cnt, v_cnt = size_cfg["r"], size_cfg["v"]
        logger.info(f"=== Evaluating Size: |R|={r_cnt}, |V|={v_cnt} ===")
        
        for i in range(args.n_instances):
            inst_seed = base_seed + (r_cnt * 1000) + i
            instance = generate_instance(
                n_regions=r_cnt,
                n_scooters=r_cnt * 2,
                n_vehicles=v_cnt,
                seed=inst_seed
            )
            
            # Ground truth: MILP for small, OR-Tools for large
            if r_cnt <= 9:
                plan_gt, _ = solve_milp(instance, economics, time_limit=60)
            else:
                plan_gt, _ = solve_ortools(instance, economics, time_limit_seconds=60)
                
            sim_gt = simulate_plan(plan_gt, instance, economics)
            z_gt = sim_gt.objective_z
            
            raw_results.append({
                "InstanceIdx": i, "Regions": r_cnt, "Vehicles": v_cnt,
                "Method": "GroundTruth", "Seed": 0, "Objective": z_gt,
                "Fulfillment": sim_gt.demand_fulfillment_rate, "Gap": 0.0
            })
            
            # Baselines
            plan_nn, _ = solve_nearest_neighbor(instance, economics)
            sim_nn = simulate_plan(plan_nn, instance, economics)
            raw_results.append({
                "InstanceIdx": i, "Regions": r_cnt, "Vehicles": v_cnt,
                "Method": "NearestNeighbor", "Seed": 0, "Objective": sim_nn.objective_z,
                "Fulfillment": sim_nn.demand_fulfillment_rate,
                "Gap": (sim_nn.objective_z - z_gt)/z_gt if z_gt > 0 else 0
            })
            
            plan_leg, sim_leg = solve_legacy_heuristic(instance, economics)
            raw_results.append({
                "InstanceIdx": i, "Regions": r_cnt, "Vehicles": v_cnt,
                "Method": "LegacyHeuristic", "Seed": 0, "Objective": sim_leg["objective_z"],
                "Fulfillment": sim_leg["demand_fulfillment_rate"],
                "Gap": (sim_leg["objective_z"] - z_gt)/z_gt if z_gt > 0 else 0
            })
            
            # HGA2C (all seeds)
            for s_idx, model_seed in enumerate(args.seeds):
                ckpt_path = f"checkpoints/seed_{model_seed}/stage3_final.pt"
                if not Path(ckpt_path).exists():
                    continue
                
                res = evaluate_hga2c(
                    instance, economics, ckpt_path, hyperparams,
                    n_rollouts=1, greedy=True, seed=model_seed
                )[0]
                
                raw_results.append({
                    "InstanceIdx": i, "Regions": r_cnt, "Vehicles": v_cnt,
                    "Method": "HGA2C", "Seed": model_seed, "Objective": res["objective_z"],
                    "Fulfillment": res["demand_fulfillment_rate"],
                    "Gap": (res["objective_z"] - z_gt)/z_gt if z_gt > 0 else 0
                })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "InstanceIdx", "Regions", "Vehicles", "Method", "Seed", "Objective", "Fulfillment", "Gap"
        ])
        writer.writeheader()
        writer.writerows(raw_results)
    
    logger.info("Generalization sweep completed. Wrote %d rows to %s", len(raw_results), out_csv)


if __name__ == "__main__":
    main()

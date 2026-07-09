"""using synthetic dataset whcich is a labeled dataset using exact MILP solver .

Uses multiprocessing to generate instances and solve them in parallel.
Generates `labeled_stage1_dataset.json` used for warm-starting the routing policy.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

from baselines.exact_milp import solve_milp
from data.instance_generator import generate_instance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def generate_and_solve(args: tuple[int, int]) -> dict[str, Any] | None:
    """Worker function for multiprocessing pool."""
    idx, seed = args
    
    instance = generate_instance(
        n_regions=4 + (idx % 6),  # Sweep 4 to 9 regions
        n_scooters=8 + (idx % 13), # Sweep 8 to 20 scooters
        n_vehicles=2,
        seed=seed,
    )
    
    economics = {
        "travel_time_cost_per_hour": 50,
        "unmet_demand_penalty": 100,
        "delay_penalty_per_hour": 10,
        "battery_swap_cost": 2,
        "relocation_cost": 5,
        "lambda_unmet": 100,
        "lambda_delay": 10,
    }
    
    try:
        # Time limit of 30 seconds per instance to avoid hanging workers
        plan, info = solve_milp(instance, economics, time_limit=30)
        
        # Only keep optimal solutions
        if info["status"] != "Optimal":
            return None
            
        return {
            "instance": instance,
            "labels": {
                "x": plan.x,
                "p": plan.p,
            },
            "metrics": info,
        }
    except Exception as e:
        logger.debug("Solver failed for instance %d: %s", idx, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled dataset via parallel MILP.")
    parser.add_argument("--n", type=int, default=1000, help="Number of instances to attempt generating")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="data/labeled_stage1_dataset.json")
    args = parser.parse_args()

    logger.info("Starting parallel MILP generation of %d instances using %d workers...", args.n, args.workers)

    tasks = [(i, args.seed + i) for i in range(args.n)]
    results = []

    # Use a multiprocessing Pool
    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(generate_and_solve, tasks)):
            if res is not None:
                results.append(res)
            
            if (i + 1) % 10 == 0:
                logger.info("Processed %d/%d instances... (Found %d optimal solutions)", i + 1, args.n, len(results))

    if not results:
        logger.error("Failed to generate any valid labeled instances.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved %d successfully labeled instances to %s", len(results), out_path)


if __name__ == "__main__":
    main()

"""Random instance generator for curriculum training 

Produces instances with the same JSON schema as configs/instance.json so the
environment code handles both real and generated instances without branching.

  Features:
  - Configurable region count, scooter count, vehicle count
  - Random scattered coordinates (uniform bounding box)
  - Poisson demand
  - Distance-based scooter assignment to nearest region
  - Enforced validity check (A < total scooters) -  important one when logic used 
  - Deterministic given a seed
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def generate_instance(
    n_regions: int = 9,
    n_scooters: int = 20,
    n_vehicles: int = 2,
    max_battery: int = 5,
    battery_threshold: int = 2,
    battery_carrying_capacity: int = 5,
    scooter_carrying_capacity: int = 5,
    extra_batteries_ratio: float = 0.6,
    extra_batteries_explicit: int | None = None,
    grid_size: float = 15.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a random instance matching the target instance schema."""
    rng = np.random.RandomState(seed)

    while True:
        # --- Generate region coordinates ---
        # Drawn uniformly in a bounding box (0 to grid_size)
        coords_raw = rng.uniform(0.5, grid_size - 0.5, size=(n_regions, 2))
        coords = [(round(float(c[0]), 2), round(float(c[1]), 2)) for c in coords_raw]

        # --- Hub: roughly center of the coordinate space with slight offset ---
        hub_x = round(float(grid_size / 2 + rng.uniform(-1.5, 1.5)), 2)
        hub_y = round(float(grid_size / 2 + rng.uniform(-1.5, 1.5)), 2)

        # --- Generate demand per region ---
        # Drawn from Poisson distribution
        demand_raw = rng.poisson(1.5, size=n_regions)
        demand = [int(d) for d in demand_raw]

        # --- Distribute scooters ---
        scooters: list[dict[str, Any]] = []
        for i in range(n_scooters):
            # Random position within bounding box
            sx = round(float(rng.uniform(0.5, grid_size - 0.5)), 2)
            sy = round(float(rng.uniform(0.5, grid_size - 0.5)), 2)
            
            # Find nearest region by Euclidean distance
            distances = [math.dist((sx, sy), (rx, ry)) for rx, ry in coords]
            nearest_region = int(np.argmin(distances))
            
            # Battery drawn uniformly 0..max_battery
            battery = int(rng.randint(0, max_battery + 1))

            scooters.append({
                "id": f"scooter_{i}",
                "region": nearest_region,
                "x": sx,
                "y": sy,
                "battery": battery,
            })

        # --- Derive Ŝ_r and S̆_r from scooter data ---
        functional: list[int] = [0] * n_regions
        non_functional: list[int] = [0] * n_regions
        for s in scooters:
            r = s["region"]
            if s["battery"] >= battery_threshold:
                functional[r] += 1
            else:
                non_functional[r] += 1

        # --- Extra batteries: proportional to non-functional count ---
        if extra_batteries_explicit is not None:
            extra_batteries = extra_batteries_explicit
        else:
            total_non_func = sum(non_functional)
            extra_batteries = max(1, int(total_non_func * extra_batteries_ratio))

        # --- Validity Check: A < total scooters ---
        if extra_batteries < n_scooters:
            break
        # If A >= scooters, resample instance entirely

    # --- Build regions list ---
    regions: list[dict[str, Any]] = []
    for i in range(n_regions):
        regions.append({
            "id": i,
            "x": coords[i][0],
            "y": coords[i][1],
            "demand": demand[i],
            "functional": functional[i],
            "non_functional": non_functional[i],
        })

    # --- Assemble instance ---
    instance: dict[str, Any] = {
        "region_count": n_regions,
        "scooter_count": n_scooters,
        "max_battery": max_battery,
        "battery_threshold": battery_threshold,
        "extra_batteries": extra_batteries,
        "battery_carrying_capacity": battery_carrying_capacity,
        "scooter_carrying_capacity": scooter_carrying_capacity,
        "vehicle_count": n_vehicles,
        "hub": {"id": "hub", "x": hub_x, "y": hub_y},
        "regions": regions,
        "scooters": scooters,
    }

    return instance


def generate_dataset(
    n_instances: int,
    min_regions: int = 4,
    max_regions: int = 16,
    min_scooters: int = 10,
    max_scooters: int = 30,
    min_vehicles: int = 1,
    max_vehicles: int = 3,
    base_seed: int = 42,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate a pool of random instances for curriculum training."""
    rng = np.random.RandomState(base_seed)
    
    dataset: list[dict[str, Any]] = []
    for i in range(n_instances):
        seed_i = base_seed + i
        n_regions = int(rng.randint(min_regions, max_regions + 1))
        n_scooters = int(rng.randint(min_scooters, max_scooters + 1))
        n_vehicles = int(rng.randint(min_vehicles, max_vehicles + 1))

        instance = generate_instance(
            n_regions=n_regions,
            n_scooters=n_scooters,
            n_vehicles=n_vehicles,
            seed=seed_i,
            **kwargs,
        )
        dataset.append(instance)

    return dataset


def generate_augmented_instance(
    instance: dict[str, Any],
    augmentation_idx: int,
) -> dict[str, Any]:
    """Apply one of 8 grid symmetries to an instance (§6.2 Stage 3)."""
    import copy

    aug = copy.deepcopy(instance)

    # Find center of coordinate system
    all_x = [aug["hub"]["x"]] + [r["x"] for r in aug["regions"]]
    all_y = [aug["hub"]["y"]] + [r["y"] for r in aug["regions"]]
    cx = (min(all_x) + max(all_x)) / 2
    cy = (min(all_y) + max(all_y)) / 2

    def transform(x: float, y: float) -> tuple[float, float]:
        """Apply the augmentation transform centered on (cx, cy)."""
        dx, dy = x - cx, y - cy
        if augmentation_idx >= 4:
            dx = -dx
        rot = (augmentation_idx % 4) * 90
        if rot == 90:
            dx, dy = -dy, dx
        elif rot == 180:
            dx, dy = -dx, -dy
        elif rot == 270:
            dx, dy = dy, -dx
        return round(cx + dx, 4), round(cy + dy, 4)

    aug["hub"]["x"], aug["hub"]["y"] = transform(aug["hub"]["x"], aug["hub"]["y"])
    for r in aug["regions"]:
        r["x"], r["y"] = transform(r["x"], r["y"])
    for s in aug["scooters"]:
        s["x"], s["y"] = transform(s["x"], s["y"])

    return aug


def save_dataset(
    dataset: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


def load_dataset(input_path: str | Path) -> list[dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training datasets")
    parser.add_argument("--n", type=int, default=1000, help="Number of instances")
    parser.add_argument("--min-regions", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--min-scooters", type=int, default=10)
    parser.add_argument("--max-scooters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/training_dataset.json")
    args = parser.parse_args()

    ds = generate_dataset(
        n_instances=args.n,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        min_scooters=args.min_scooters,
        max_scooters=args.max_scooters,
        base_seed=args.seed,
    )
    save_dataset(ds, args.output)
    print(f"Generated {len(ds)} instances -> {args.output}")

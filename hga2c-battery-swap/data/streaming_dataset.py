"""Streaming Dataset for infinite on-the-fly instance generation.

Uses PyTorch IterableDataset to continuously stream uniquely generated
VRP instances via multiprocessing, avoiding memory bloat and overfitting.
"""
from __future__ import annotations

import torch
from torch.utils.data import IterableDataset

from data.instance_generator import generate_instance


def raw_collate_fn(batch):
    """Pass-through collate function to avoid converting dicts to tensors."""
    # Since batch_size=None or 1, we just return the raw python objects.
    # If batch is a list of 1 dict, we just return the dict.
    if isinstance(batch, list) and len(batch) == 1:
        return batch[0]
    return batch


class InstanceDataset(IterableDataset):
    """Infinite iterable dataset for VRP instances."""

    def __init__(
        self,
        min_regions: int = 4,
        max_regions: int = 16,
        min_scooters: int = 10,
        max_scooters: int = 30,
        min_vehicles: int = 1,
        max_vehicles: int = 3,
        base_seed: int = 42,
    ):
        super().__init__()
        self.min_regions = min_regions
        self.max_regions = max_regions
        self.min_scooters = min_scooters
        self.max_scooters = max_scooters
        self.min_vehicles = min_vehicles
        self.max_vehicles = max_vehicles
        self.base_seed = base_seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            seed = self.base_seed
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            seed = self.base_seed + worker_id

        # Use PyTorch's native RNG for sampling configs
        rng = torch.Generator()
        rng.manual_seed(seed)
        
        # We will use this to offset the python random seed in generate_instance
        instance_counter = 0

        while True:
            r = int(torch.randint(self.min_regions, self.max_regions + 1, (1,), generator=rng).item())
            s = int(torch.randint(self.min_scooters, self.max_scooters + 1, (1,), generator=rng).item())
            v = int(torch.randint(self.min_vehicles, self.max_vehicles + 1, (1,), generator=rng).item())
            
            inst_seed = seed + instance_counter * 1000
            instance_counter += 1
            
            instance = generate_instance(
                n_regions=r,
                n_scooters=s,
                n_vehicles=v,
                seed=inst_seed
            )
            yield instance

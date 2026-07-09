"""Stage 2: Joint training (§6.2).

Trains both Level-1 and Level-2 end-to-end via A2C on randomly generated
instances of varying sizes to learn a generalizable policy.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import re
import yaml

from data.streaming_dataset import InstanceDataset, raw_collate_fn
from torch.utils.data import DataLoader
from models.hga2c_policy import HGA2CPolicy, build_policy_from_config
from training.a2c_trainer import A2CTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Joint training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--hyperparams", type=str, default="configs/hyperparams.yaml")
    parser.add_argument("--economics", type=str, default="configs/economics.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Stage 1 checkpoint to load")
    args = parser.parse_args()

    with open(args.hyperparams) as f:
        hp = yaml.safe_load(f)
    with open(args.economics) as f:
        econ = yaml.safe_load(f)

    hp["seed"] = args.seed
    stage_cfg = hp.get("stage2", {})
    n_episodes = args.episodes or stage_cfg.get("episodes", 20000)

    # Generate training instances via Streaming Dataset
    dataset = InstanceDataset(
        min_regions=stage_cfg.get("min_regions", 4),
        max_regions=stage_cfg.get("max_regions", 16),
        min_scooters=stage_cfg.get("min_scooters", 10),
        max_scooters=stage_cfg.get("max_scooters", 30),
        min_vehicles=stage_cfg.get("min_vehicles", 1),
        max_vehicles=stage_cfg.get("max_vehicles", 3),
        base_seed=args.seed + 10000,
    )
    
    # Use DataLoader to spin up 4 background workers for graph generation
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=10, collate_fn=raw_collate_fn)
    logger.info("Streaming instances dynamically using DataLoader for Stage 2...")

    with open("configs/instance.json") as f:
        target_instance = json.load(f)

    # Build policy
    policy = build_policy_from_config(hp)

    # Load Stage 1 checkpoint or mid-stage Stage 2 checkpoint
    start_episode = 0
    
    # Check for mid-stage Stage 2 checkpoints first
    stage2_ckpts = list(Path(args.checkpoint_dir).glob("stage2_ep*.pt"))
    if stage2_ckpts:
        # Find the one with the highest episode number
        def get_ep(p):
            m = re.search(r"stage2_ep(\d+)\.pt", p.name)
            return int(m.group(1)) if m else -1
            
        latest_ckpt = max(stage2_ckpts, key=get_ep)
        ep = get_ep(latest_ckpt)
        if ep > 0:
            start_episode = ep
            policy.load_checkpoint(str(latest_ckpt))
            logger.info("Resuming mid-stage! Loaded Stage 2 checkpoint: %s (Starting at episode %d)", latest_ckpt, start_episode)
    else:
        stage1_ckpt = args.resume or str(Path(args.checkpoint_dir) / "stage1_final.pt")
        if Path(stage1_ckpt).exists():
            policy.load_checkpoint(stage1_ckpt)
            logger.info("Loaded Stage 1 checkpoint: %s", stage1_ckpt)
        else:
            logger.warning("No Stage 1 checkpoint found at %s, training from scratch", stage1_ckpt)

    # Unfreeze ALL parameters for joint training
    for param in policy.parameters():
        param.requires_grad = True
    logger.info("Stage 2: All parameters UNFROZEN for joint training")

    # Override LR if specified
    lr_override = stage_cfg.get("lr_override")
    if lr_override:
        hp["lr_encoder"] = lr_override
        hp["lr_critic"] = lr_override * 3  # maintain ratio

    trainer = A2CTrainer(
        policy=policy,
        hyperparams=hp,
        economics=econ,
        log_dir=f"runs/stage2_seed{args.seed}",
    )
    # Run training loop
    trainer.train(
        instances=dataloader,
        n_episodes=n_episodes,
        eval_interval=1000,
        eval_instance=target_instance,
        checkpoint_dir=args.checkpoint_dir,
        stage_name="stage2",
        start_episode=start_episode,
    )


if __name__ == "__main__":
    main()

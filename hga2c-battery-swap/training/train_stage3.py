"""Stage 3: Fine-tuning on target instance (§6.2).

Fine-tunes both levels on the exact 9-region/20-scooter target instance
using instance augmentation (8 grid symmetries).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from data.instance_generator import generate_augmented_instance
from models.hga2c_policy import build_policy_from_config
from training.a2c_trainer import A2CTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--hyperparams", type=str, default="configs/hyperparams.yaml")
    parser.add_argument("--economics", type=str, default="configs/economics.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.hyperparams) as f:
        hp = yaml.safe_load(f)
    with open(args.economics) as f:
        econ = yaml.safe_load(f)

    hp["seed"] = args.seed
    stage_cfg = hp.get("stage3", {})
    n_episodes = args.episodes or stage_cfg.get("episodes", 5000)

    # Load target instance
    with open("configs/instance.json") as f:
        target_instance = json.load(f)

    # Generate augmented instances (8 symmetries)
    instances = [target_instance]  # Original
    if stage_cfg.get("use_augmentation", True):
        for aug_idx in range(1, 8):
            aug_instance = generate_augmented_instance(target_instance, aug_idx)
            instances.append(aug_instance)
        logger.info("Created %d augmented instances (8 symmetries)", len(instances))
    else:
        logger.info("Augmentation disabled, training on original instance only")

    # Build policy
    policy = build_policy_from_config(hp)

    # Load Stage 2 checkpoint
    stage2_ckpt = args.resume or str(Path(args.checkpoint_dir) / "stage2_final.pt")
    if Path(stage2_ckpt).exists():
        policy.load_checkpoint(stage2_ckpt)
        logger.info("Loaded Stage 2 checkpoint: %s", stage2_ckpt)
    else:
        logger.warning("No Stage 2 checkpoint found at %s", stage2_ckpt)

    # Unfreeze all
    for param in policy.parameters():
        param.requires_grad = True

    # Reduced LR for fine-tuning
    lr_override = stage_cfg.get("lr_override", 5e-5)
    hp["lr_encoder"] = lr_override
    hp["lr_critic"] = lr_override * 3

    logger.info("Stage 3: Fine-tuning with LR=%.1e on target + augmentations", lr_override)

    trainer = A2CTrainer(
        policy=policy,
        hyperparams=hp,
        economics=econ,
        log_dir=f"runs/stage3_seed{args.seed}",
    )

    trainer.train(
        instances=instances,
        n_episodes=n_episodes,
        eval_interval=1000,
        eval_instance=target_instance,
        checkpoint_dir=args.checkpoint_dir,
        stage_name="stage3",
    )


if __name__ == "__main__":
    main()

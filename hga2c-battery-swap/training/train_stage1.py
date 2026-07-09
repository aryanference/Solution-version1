"""Stage 1: Routing warm-start (§6.2).

Trains only the Level-2 routing actor with Level-1 allocation fixed to
MILP ground-truth solutions on small random instances.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from data.instance_generator import generate_dataset
from models.hga2c_policy import build_policy_from_config
from training.a2c_trainer import A2CTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Routing warm-start")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count from config")
    parser.add_argument("--hyperparams", type=str, default="configs/hyperparams.yaml")
    parser.add_argument("--economics", type=str, default="configs/economics.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.hyperparams) as f:
        hp = yaml.safe_load(f)
    with open(args.economics) as f:
        econ = yaml.safe_load(f)

    hp["seed"] = args.seed
    stage_cfg = hp.get("stage1", {})
    n_episodes = args.episodes or stage_cfg.get("episodes", 5000)

    # Load pre-computed MILP labeled training instances
    labeled_data_path = Path("data/labeled_stage1_dataset.json")
    if not labeled_data_path.exists():
        raise FileNotFoundError(
            f"{labeled_data_path} not found. Run data/generate_labeled_dataset.py first."
        )
    logger.info("Loading labeled instances from %s for Stage 1...", labeled_data_path)
    with open(labeled_data_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    # Load target instance for evaluation
    with open("configs/instance.json") as f:
        target_instance = json.load(f)

    # Build policy
    policy = build_policy_from_config(hp)
    logger.info("Policy parameters: %d", sum(p.numel() for p in policy.parameters()))

    # Freeze Level-1 allocation actor (only train routing + encoder + critic)
    for param in policy.allocation_actor.parameters():
        param.requires_grad = False
    logger.info("Stage 1: Allocation actor FROZEN, training routing only")

    # Train
    trainer = A2CTrainer(
        policy=policy,
        hyperparams=hp,
        economics=econ,
        log_dir=f"runs/stage1_seed{args.seed}",
    )

    trainer.train(
        instances=instances,
        n_episodes=n_episodes,
        eval_interval=hp.get("eval_interval", 100),
        eval_instance=target_instance,
        checkpoint_dir=args.checkpoint_dir,
        stage_name="stage1",
    )


if __name__ == "__main__":
    main()

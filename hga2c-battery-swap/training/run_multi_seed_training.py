"""Orchestrate multi-seed training across all curriculum stages."""
import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed training orchestrator")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        ckpt_dir = f"checkpoints/seed_{seed}"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"=== Starting Training Pipeline for Seed {seed} ===")

        if not args.skip_stage1:
            if Path(f"{ckpt_dir}/stage1_final.pt").exists():
                logger.info(f"--- Stage 1: Warm-start (Seed {seed}) already complete, skipping ---")
            else:
                logger.info(f"--- Stage 1: Warm-start (Seed {seed}) ---")
                subprocess.run([
                    "python", "-m", "training.train_stage1", 
                    "--seed", str(seed),
                    "--checkpoint-dir", ckpt_dir
                ], check=True)

        if not args.skip_stage2:
            if Path(f"{ckpt_dir}/stage2_final.pt").exists():
                logger.info(f"--- Stage 2: Joint Training (Seed {seed}) already complete, skipping ---")
            else:
                logger.info(f"--- Stage 2: Joint Training (Seed {seed}) ---")
                subprocess.run([
                    "python", "-m", "training.train_stage2", 
                    "--seed", str(seed),
                    "--checkpoint-dir", ckpt_dir,
                    "--resume", f"{ckpt_dir}/stage1_final.pt"
                ], check=True)

        if not args.skip_stage3:
            if Path(f"{ckpt_dir}/stage3_final.pt").exists():
                logger.info(f"--- Stage 3: Fine-tuning (Seed {seed}) already complete, skipping ---")
            else:
                logger.info(f"--- Stage 3: Fine-tuning (Seed {seed}) ---")
                subprocess.run([
                    "python", "-m", "training.train_stage3", 
                    "--seed", str(seed),
                    "--checkpoint-dir", ckpt_dir,
                    "--resume", f"{ckpt_dir}/stage2_final.pt"
                ], check=True)
            
        logger.info(f"=== Completed Training Pipeline for Seed {seed} ===")


if __name__ == "__main__":
    main()

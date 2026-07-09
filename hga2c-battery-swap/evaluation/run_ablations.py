"""Architecture ablation orchestration.

Orchestrates multi-seed training and evaluation for:
1. No travel-time attention bias (use_travel_time_bias: false)
2. No Stage 1 warm-start (train end-to-end from scratch)
3. No Stage 3 fine-tuning (evaluate directly on Stage 2 checkpoint)
"""
import argparse
import csv
import logging
import subprocess
import yaml
from pathlib import Path

from evaluation.evaluate import evaluate_hga2c

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def run_ablation(name, hyperparams, seeds, skip_stage1=False, skip_stage3=False):
    """Run training and evaluation for a specific ablation."""
    
    # Save temp config
    temp_hp_path = f"configs/hyperparams_{name}.yaml"
    with open(temp_hp_path, "w") as f:
        yaml.dump(hyperparams, f)
        
    for seed in seeds:
        ckpt_dir = f"checkpoints/{name}_seed_{seed}"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"=== Starting Ablation {name} for Seed {seed} ===")

        if not skip_stage1:
            subprocess.run([
                "python", "-m", "training.train_stage1", 
                "--seed", str(seed),
                "--hyperparams", temp_hp_path,
                "--checkpoint-dir", ckpt_dir
            ], check=True)

        logger.info(f"--- Stage 2 ---")
        subprocess.run([
            "python", "-m", "training.train_stage2", 
            "--seed", str(seed),
            "--hyperparams", temp_hp_path,
            "--checkpoint-dir", ckpt_dir,
            "--resume", f"{ckpt_dir}/stage1_final.pt" if not skip_stage1 else "none"
        ], check=True)

        if not skip_stage3:
            logger.info(f"--- Stage 3 ---")
            subprocess.run([
                "python", "-m", "training.train_stage3", 
                "--seed", str(seed),
                "--hyperparams", temp_hp_path,
                "--checkpoint-dir", ckpt_dir,
                "--resume", f"{ckpt_dir}/stage2_final.pt"
            ], check=True)
            

def evaluate_ablation(name, seeds, use_stage2=False):
    """Evaluate an ablation and return results."""
    with open("configs/instance.json") as f:
        import json
        instance = json.load(f)
    with open("configs/economics.yaml") as f:
        economics = yaml.safe_load(f)
        
    temp_hp_path = f"configs/hyperparams_{name}.yaml"
    with open(temp_hp_path) as f:
        hp = yaml.safe_load(f)
        
    results = []
    ckpt_name = "stage2_final.pt" if use_stage2 else "stage3_final.pt"
    
    for seed in seeds:
        ckpt_path = f"checkpoints/{name}_seed_{seed}/{ckpt_name}"
        if not Path(ckpt_path).exists():
            continue
            
        res = evaluate_hga2c(
            instance, economics, ckpt_path, hp,
            n_rollouts=1, greedy=True, seed=seed
        )[0]
        
        results.append({
            "Method": name, "Seed": seed, "Objective": res["objective_z"],
            "Fulfillment": res["demand_fulfillment_rate"]
        })
        
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    with open("configs/hyperparams.yaml") as f:
        base_hp = yaml.safe_load(f)

    ablations = [
        ("NoTravelBias", {"use_travel_time_bias": False}, False, False, False),
        ("NoStage1", {"use_travel_time_bias": True}, True, False, False),
        ("NoStage3", {"use_travel_time_bias": True}, False, True, True),  # True at end means evaluate on stage2
    ]
    
    all_results = []
    
    for name, overrides, skip_s1, skip_s3, eval_s2 in ablations:
        hp = base_hp.copy()
        hp.update(overrides)
        
        if not args.skip_train:
            run_ablation(name, hp, args.seeds, skip_s1, skip_s3)
            
        res = evaluate_ablation(name, args.seeds, use_stage2=eval_s2)
        all_results.extend(res)

    out_csv = Path("paper/tables/raw_architecture_ablation.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Seed", "Objective", "Fulfillment"])
        writer.writeheader()
        writer.writerows(all_results)
        
    logger.info("Ablation evaluations complete. Written to %s", out_csv)


if __name__ == "__main__":
    main()

"""Generate paper-ready assets (plots, LaTeX tables, Pareto curves).

This script orchestrates the lambda_unmet Pareto ablation and consumes the
raw CSV outputs from evaluate.py and generalization_sweep.py to produce
publication-ready figures and tables.
"""
import argparse
import csv
import logging
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from evaluation.evaluate import evaluate_hga2c

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def generate_pareto_curve(seeds: list[int], skip_train: bool = False):
    """Run lambda_unmet ablation and plot Pareto curve."""
    logger.info("=== Generating Pareto Curve (lambda_unmet ablation) ===")
    
    with open("configs/economics.yaml") as f:
        econ_base = yaml.safe_load(f)
        
    with open("configs/hyperparams.yaml") as f:
        hp_base = yaml.safe_load(f)
        
    with open("configs/instance.json") as f:
        import json
        instance = json.load(f)
        
    lambda_sweep = econ_base.get("lambda_unmet_sweep", [10, 50, 100, 200])
    results = []
    
    for l_val in lambda_sweep:
        logger.info(f"--- Running Pareto for lambda_unmet={l_val} ---")
        econ = econ_base.copy()
        econ["lambda_unmet"] = float(l_val)
        
        temp_econ_path = f"configs/economics_pareto_{l_val}.yaml"
        with open(temp_econ_path, "w") as f:
            yaml.dump(econ, f)
            
        for seed in seeds:
            ckpt_dir = f"checkpoints/pareto_{l_val}_seed_{seed}"
            
            if not skip_train:
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                # We fine-tune Stage 3 from the Stage 2 checkpoint
                # (Assuming run_multi_seed_training already produced stage2_final.pt)
                src_ckpt = f"checkpoints/seed_{seed}/stage2_final.pt"
                if not Path(src_ckpt).exists():
                    logger.warning(f"Missing Stage 2 checkpoint {src_ckpt} for seed {seed}. Skipping.")
                    continue
                    
                subprocess.run([
                    "python", "-m", "training.train_stage3",
                    "--seed", str(seed),
                    "--hyperparams", "configs/hyperparams.yaml",
                    "--economics", temp_econ_path,
                    "--checkpoint-dir", ckpt_dir,
                    "--resume", src_ckpt
                ], check=True)
                
            ckpt_path = f"{ckpt_dir}/stage3_final.pt"
            if Path(ckpt_path).exists():
                res = evaluate_hga2c(
                    instance, econ, ckpt_path, hp_base,
                    n_rollouts=1, greedy=True, seed=seed
                )[0]
                
                results.append({
                    "LambdaUnmet": l_val, "Seed": seed,
                    "TravelCost": res["travel_cost"],
                    "Fulfillment": res["demand_fulfillment_rate"]
                })
                
    # Dump to CSV
    out_csv = Path("paper/tables/raw_pareto_ablation.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["LambdaUnmet", "Seed", "TravelCost", "Fulfillment"])
        writer.writeheader()
        writer.writerows(results)
        
    # Plotting
    if not results:
        logger.warning("No Pareto results to plot.")
        return
        
    l_vals = sorted(list(set([r["LambdaUnmet"] for r in results])))
    avg_travel = []
    avg_fulfill = []
    
    for l_val in l_vals:
        tcs = [r["TravelCost"] for r in results if r["LambdaUnmet"] == l_val]
        fs = [r["Fulfillment"] for r in results if r["LambdaUnmet"] == l_val]
        avg_travel.append(np.mean(tcs))
        avg_fulfill.append(np.mean(fs))
        
    plt.figure(figsize=(8, 5))
    plt.plot(avg_travel, avg_fulfill, marker='o', linestyle='-', linewidth=2, markersize=8)
    for i, l_val in enumerate(l_vals):
        plt.annotate(f"λ'={l_val}", (avg_travel[i], avg_fulfill[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.xlabel("Travel Cost ($)")
    plt.ylabel("Demand Fulfillment Rate")
    plt.title("Pareto Curve: Travel Cost vs Fulfillment")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_fig = Path("paper/figures/pareto_ablation.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Pareto curve to %s", out_fig)


def generate_generalization_plots():
    """Plot Gap vs Size and Fulfillment vs Size."""
    logger.info("=== Generating Generalization Sweep Plots ===")
    
    csv_path = Path("paper/tables/raw_generalization_sweep.csv")
    if not csv_path.exists():
        logger.warning(f"File {csv_path} not found. Skip generalization plots.")
        return
        
    data = {"HGA2C": {}, "NearestNeighbor": {}, "LegacyHeuristic": {}}
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meth = row["Method"]
            if meth not in data:
                continue
                
            size_key = int(row["Regions"])
            gap = float(row["Gap"])
            fulfill = float(row["Fulfillment"])
            
            if size_key not in data[meth]:
                data[meth][size_key] = {"gap": [], "fulfill": []}
                
            data[meth][size_key]["gap"].append(gap)
            data[meth][size_key]["fulfill"].append(fulfill)
            
    sizes = sorted(list(data["HGA2C"].keys()))
    if not sizes:
        return
        
    # Plot Gap vs Size
    plt.figure(figsize=(8, 5))
    for meth, style in zip(["HGA2C", "NearestNeighbor", "LegacyHeuristic"], ['o-', 's--', '^:']):
        gaps = [np.mean(data[meth][s]["gap"]) * 100 for s in sizes]
        plt.plot(sizes, gaps, style, label=meth, linewidth=2, markersize=8)
        
    plt.xlabel("Instance Size (|R|)")
    plt.ylabel("Optimality Gap (%)")
    plt.title("Zero-Shot Generalization: Optimality Gap vs Instance Size")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sizes)
    out_fig = Path("paper/figures/gap_vs_size.png")
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Fulfillment vs Size
    plt.figure(figsize=(8, 5))
    for meth, style in zip(["HGA2C", "NearestNeighbor", "LegacyHeuristic"], ['o-', 's--', '^:']):
        fs = [np.mean(data[meth][s]["fulfill"]) * 100 for s in sizes]
        plt.plot(sizes, fs, style, label=meth, linewidth=2, markersize=8)
        
    plt.xlabel("Instance Size (|R|)")
    plt.ylabel("Demand Fulfillment Rate (%)")
    plt.title("Zero-Shot Generalization: Fulfillment vs Instance Size")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sizes)
    out_fig2 = Path("paper/figures/fulfill_vs_size.png")
    plt.savefig(out_fig2, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved generalization plots to paper/figures/")


def generate_latex_table():
    """Convert raw evaluation CSV to a formatted LaTeX table."""
    logger.info("=== Generating LaTeX Table ===")
    csv_path = Path("paper/tables/raw_evaluation_reference.csv")
    if not csv_path.exists():
        logger.warning(f"File {csv_path} not found. Skip LaTeX table.")
        return
        
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meth = row["Method"]
            if meth not in data:
                data[meth] = {"obj": [], "f": [], "time": [], "gap": []}
                
            data[meth]["obj"].append(float(row["Objective"]))
            data[meth]["f"].append(float(row["Fulfillment"]))
            data[meth]["time"].append(float(row["Time"]))
            if row["Gap"] != "None":
                data[meth]["gap"].append(float(row["Gap"]))
                
    order = ["MILP", "NearestNeighbor", "OR-Tools", "Random", "LegacyHeuristic", "HGA2C"]
    
    tex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Performance on the Reference Instance ($|R|=9$, $|V|=2$)}",
        "\\begin{tabular}{l c c c c}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Objective} & \\textbf{Fulfillment} & \\textbf{Time (s)} & \\textbf{Gap} \\\\",
        "\\midrule"
    ]
    
    for meth in order:
        if meth not in data:
            continue
            
        obj_m = np.mean(data[meth]["obj"])
        obj_s = np.std(data[meth]["obj"])
        f_m = np.mean(data[meth]["f"]) * 100
        t_m = np.mean(data[meth]["time"])
        gap_vals = data[meth]["gap"]
        
        if len(gap_vals) > 0:
            g_m = np.mean(gap_vals) * 100
            g_str = f"{g_m:.1f}\\%"
        else:
            g_str = "-"
            
        obj_str = f"{obj_m:.1f} $\\pm$ {obj_s:.1f}" if obj_s > 0.1 else f"{obj_m:.1f}"
        tex_lines.append(f"{meth} & {obj_str} & {f_m:.1f}\\% & {t_m:.2f} & {g_str} \\\\")
        
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    out_tex = Path("paper/tables/reference_results.tex")
    with open(out_tex, "w") as f:
        f.write("\n".join(tex_lines))
        
    logger.info("Saved LaTeX table to %s", out_tex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    
    generate_pareto_curve(args.seeds, args.skip_train)
    generate_generalization_plots()
    generate_latex_table()


if __name__ == "__main__":
    main()

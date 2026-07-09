"""Statistical testing module for HGA²C evaluation.

Consumes the raw CSV generalization outputs and runs Wilcoxon signed-rank
tests on Objective Value and Demand Fulfillment Rate to prove statistically
significant outperformance against baselines.
"""
import argparse
import csv
import logging
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def run_wilcoxon(data1, data2, method_name, metric):
    try:
        # Wilcoxon requires differences to be non-zero for all pairs,
        # but scipy handles it by dropping zeros.
        stat, p_val = wilcoxon(data1, data2)
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        logger.info(f"HGA2C vs {method_name} ({metric}): p-value = {p_val:.3e} {significance}")
    except Exception as e:
        logger.warning(f"Failed to run Wilcoxon for {method_name} ({metric}): {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", type=str, default="paper/tables/raw_generalization_sweep.csv")
    args = parser.parse_args()

    # Data structure: metrics[size][instance][method] = avg_value
    # Since HGA2C has multiple seeds, we'll average over seeds per instance before testing against baselines.
    
    obj_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fulfill_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    try:
        with open(args.in_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                size = f"R{row['Regions']}_V{row['Vehicles']}"
                inst = int(row['InstanceIdx'])
                meth = row['Method']
                
                obj_data[size][inst][meth].append(float(row['Objective']))
                fulfill_data[size][inst][meth].append(float(row['Fulfillment']))
    except FileNotFoundError:
        logger.error(f"File {args.in_csv} not found. Run generalization_sweep.py first.")
        return

    for size in obj_data.keys():
        logger.info(f"\n=== Statistical Tests for {size} ===")
        
        methods = list(next(iter(obj_data[size].values())).keys())
        if "HGA2C" not in methods:
            logger.warning("HGA2C not found in this size group.")
            continue
            
        baselines = [m for m in methods if m not in ["HGA2C", "GroundTruth"]]
        
        for baseline in baselines:
            hga2c_obj = []
            base_obj = []
            hga2c_f = []
            base_f = []
            
            for inst in obj_data[size].keys():
                # Average over seeds for HGA2C
                h_obj = np.mean(obj_data[size][inst]["HGA2C"])
                b_obj = np.mean(obj_data[size][inst][baseline])
                
                h_f = np.mean(fulfill_data[size][inst]["HGA2C"])
                b_f = np.mean(fulfill_data[size][inst][baseline])
                
                hga2c_obj.append(h_obj)
                base_obj.append(b_obj)
                hga2c_f.append(h_f)
                base_f.append(b_f)
                
            run_wilcoxon(hga2c_obj, base_obj, baseline, "Objective (Z)")
            run_wilcoxon(hga2c_f, base_f, baseline, "Fulfillment Rate")


if __name__ == "__main__":
    main()

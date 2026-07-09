# HGA²C: Hierarchical Graph-Attention Actor-Critic for E-Scooter Battery Swap & Relocation

## Overview

A research artifact and production-quality deep reinforcement learning system that solves the E-Scooter Battery Swap & Relocation Problem using a **Hierarchical Graph Attention Actor-Critic (HGA²C)** policy. This repository is structured to support rigorous academic publication, featuring:
- **Zero-Shot Generalization**: Performance validation across varying instance sizes without retraining.
- **Robust Baselines**: Benchmarked against exact MILP solvers, OR-Tools, Nearest-Neighbor, and a reconstructed Legacy Heuristic (nearest-unserved-region).
- **Statistical Rigor**: All training and evaluations are orchestrated across multiple independent seeds, with raw CSV outputs and Wilcoxon signed-rank statistical testing.
- **Headline Metrics**: Explicit tracking of demand-fulfillment rate alongside traditional cost objectives.

## Architecture

```
Instance (JSON) ──► Environment (Gymnasium) ──► HGA²C Policy ──► Simulator ──► Reward
                         │                         │
                    Feasibility Masks          GAT Encoder
                    (MILP constraints)         ├─ Level-1: Allocation Actor (x_r, p_rl)
                                               ├─ Level-2: Routing Actor (PDP Pointer Net)
                                               └─ Critic (Value baseline)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run foundation tests
pytest tests/ -v

# Train HGA²C across 5 independent seeds
python -m training.run_multi_seed_training

# Evaluate all methods and generate raw CSV dumps
python -m evaluation.evaluate

# Run zero-shot generalization sweep
python -m evaluation.generalization_sweep

# Launch interactive dashboard
streamlit run simulation/dashboard.py
```

## Manuscript Structure

The intended manuscript structure supported by this repository is:
1. **Introduction**: The gap being addressed (allocation and pickup-delivery routing learned jointly, validated against exact solvers).
2. **Related Work**: Attention-model VRP/PDP literature and hierarchical RL for resource allocation.
3. **Problem Formulation**: The exact MILP model (§5).
4. **Method**: The two-level HGA²C architecture and curriculum training.
5. **Experimental Setup**: Training datasets, metrics, and baselines (including the Legacy Heuristic).
6. **Results**: The 6-method baseline table, generalization sweep line plots, lambda' ablation Pareto curve, and architecture ablations.
7. **Limitations**: Disclosures regarding the single-period scope and fleet homogeneity.
8. **Conclusion**.

## Limitations and Disclosures

This work relies on several explicit modeling choices and assumptions:
- **Single-Period Scope**: The model optimizes a single pre-rush operating window, not a rolling multi-shift horizon.
- **Homogeneous Fleet**: Service vehicles are assumed to have identical battery and scooter capacities.
- **Economic Assumptions**: The cost parameters ($\lambda, \lambda', T_r, h_r$) in `configs/economics.yaml` are documented assumptions rather than values strictly fit to historical operating data. Where applicable, parameter bands can be calibrated against open datasets (e.g., Chicago, Louisville) to ground these assumptions further.

## Business Assumptions

> ⚠️ **Tune these with your actual operating economics.** All values live in config files, never hardcoded.

| Parameter | Default | Config Key | Rationale |
|---|---|---|---|
| λ (travel cost, $/min) | 1.0 | `economics.lambda_travel` | Normalizes travel cost to raw minutes |
| λ' (unmet demand penalty) | 50 | `economics.lambda_unmet` | Must dominate travel cost |
| Period length (min) | 60 | `economics.period_length` | Pre-rush operating window |
| λ'' (delay penalty) | λ'/period ≈ 0.833 | Derived | Per model definition |
| T_r processing: swap | 1.5 min/swap | `economics.swap_time_min` | Estimated handling time |
| T_r processing: reloc | 1.0 min/op | `economics.reloc_time_min` | Pickup/dropoff time |
| h_r (urgency weight) | h_r = D_r | `economics.urgency_weighting` | Demand-proportional |
| Vehicle count | 2 | `instance.vehicle_count` | Configurable 1–N |
| Battery capacity | 5/vehicle | `instance.battery_carrying_capacity` | Homogeneous fleet |
| Scooter capacity | 5/vehicle | `instance.scooter_carrying_capacity` | Homogeneous fleet |

## Reproducibility

Every evaluation and training script dumps raw per-instance outputs into `paper/tables/` to ensure all figures and metrics can be regenerated without rerunning heavy model inferences.

## License

Research use only.

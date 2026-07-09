"""Benchmark report generator — produces comparison charts/tables (§8).

Generates Plotly bar charts for objective value, unmet demand, and inference
time across all methods. Saves as HTML and PNG.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def generate_benchmark_charts(
    results: dict[str, dict[str, Any]],
    output_dir: str = "evaluation/results",
) -> None:
    """Generate comparison bar charts from evaluation results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    colors = {
        "MILP": "#2ecc71",
        "NearestNeighbor": "#e67e22",
        "OR-Tools": "#3498db",
        "Random": "#e74c3c",
        "HGA2C_greedy": "#9b59b6",
        "HGA2C_stochastic": "#8e44ad",
    }

    # --- Chart 1: Objective Value Comparison ---
    fig1 = go.Figure()
    z_values = [results[m].get("objective_z", 0) for m in methods]
    z_stds = [results[m].get("objective_z_std", 0) or 0 for m in methods]
    fig1.add_trace(go.Bar(
        x=methods, y=z_values,
        error_y=dict(type="data", array=z_stds, visible=True),
        marker_color=[colors.get(m, "#95a5a6") for m in methods],
        text=[f"{z:.1f}" for z in z_values],
        textposition="auto",
    ))
    fig1.update_layout(
        title="Objective Value Z (lower is better)",
        yaxis_title="Objective Z",
        template="plotly_dark",
    )
    fig1.write_html(str(out / "objective_comparison.html"))

    # --- Chart 2: Unmet Demand ---
    fig2 = go.Figure()
    unmet = [results[m].get("unmet_demand", 0) for m in methods]
    fig2.add_trace(go.Bar(
        x=methods, y=unmet,
        marker_color=[colors.get(m, "#95a5a6") for m in methods],
        text=[f"{u:.1f}" for u in unmet],
        textposition="auto",
    ))
    fig2.update_layout(
        title="Unmet Demand (lower is better)",
        yaxis_title="Unmet Demand",
        template="plotly_dark",
    )
    fig2.write_html(str(out / "unmet_demand_comparison.html"))

    # --- Chart 3: Inference Time ---
    fig3 = go.Figure()
    times = [results[m].get("inference_time", 0) for m in methods]
    fig3.add_trace(go.Bar(
        x=methods, y=times,
        marker_color=[colors.get(m, "#95a5a6") for m in methods],
        text=[f"{t:.3f}s" for t in times],
        textposition="auto",
    ))
    fig3.update_layout(
        title="Inference Time (lower is better)",
        yaxis_title="Time (seconds)",
        yaxis_type="log",
        template="plotly_dark",
    )
    fig3.write_html(str(out / "inference_time_comparison.html"))

    # --- Combined subplot ---
    fig4 = make_subplots(rows=1, cols=3,
                         subplot_titles=["Objective Z", "Unmet Demand", "Inference Time"])
    for i, (vals, title) in enumerate([
        (z_values, "Objective"), (unmet, "Unmet"), (times, "Time")
    ]):
        fig4.add_trace(go.Bar(
            x=methods, y=vals,
            marker_color=[colors.get(m, "#95a5a6") for m in methods],
            showlegend=False,
        ), row=1, col=i+1)
    fig4.update_layout(title="Full Benchmark Comparison", template="plotly_dark", height=400)
    fig4.write_html(str(out / "full_comparison.html"))

    logger.info("Charts saved to %s", out)


if __name__ == "__main__":
    results_path = Path("evaluation/results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        generate_benchmark_charts(results)
    else:
        print("No results.json found. Run evaluate.py first.")

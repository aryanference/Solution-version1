"""using Streamlit now as a  Dashboard for E-Scooter Battery Swap Simulation .

Features:
  1. Map view: Hub, 9 regions, 20 scooters colored by battery status, vehicles.
  2. Animated playback: vehicle routes with play/pause/speed controls.
  3. Live metrics panel: travel cost, unmet demand, batteries remaining.
  4. Side-by-side comparison: bar charts of objective + inference time.

Run with: streamlit run simulation/dashboard.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from env.simulator import (
    Plan,
    PlanResult,
    VehicleRoute,
    build_travel_time_matrix,
    simulate_plan,
)
from baselines.nearest_neighbor import solve_nearest_neighbor
from baselines.random_policy import solve_random
from baselines.legacy_heuristic import solve_legacy_heuristic
from data.instance_generator import generate_instance

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HGA²C E-Scooter Battery Swap Simulator",
    page_icon="🛴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown( unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_instance() -> dict[str, Any]:
    with open(Path(__file__).parent.parent / "configs" / "instance.json") as f:
        return json.load(f)


@st.cache_data
def load_economics() -> dict[str, Any]:
    with open(Path(__file__).parent.parent / "configs" / "economics.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def solve_method(method_name: str, instance: dict, economics: dict) -> tuple[Plan, PlanResult]:
    """Solve with a given method and cache the result."""
    if method_name == "Nearest Neighbor":
        plan, _ = solve_nearest_neighbor(instance, economics)
    elif method_name == "Legacy Heuristic":
        plan, _ = solve_legacy_heuristic(instance, economics)
    elif method_name == "Random":
        plan, _ = solve_random(instance, economics, seed=42)
    elif method_name == "MILP":
        try:
            from baselines.exact_milp import solve_milp
            plan, _ = solve_milp(instance, economics, time_limit=120)
        except Exception as e:
            st.error(f"MILP solver failed: {e}")
            plan, _ = solve_nearest_neighbor(instance, economics)
    elif method_name == "HGA²C":
        try:
            import torch
            from models.hga2c_policy import build_policy_from_config
            from env.battery_swap_env import make_env
            with open(Path(__file__).parent.parent / "configs" / "hyperparams.yaml") as f:
                hp = yaml.safe_load(f)
            policy = build_policy_from_config(hp)
            ckpt = Path(__file__).parent.parent / "checkpoints" / "seed_42" / "stage3_final.pt"
            if ckpt.exists():
                policy.load_checkpoint(ckpt)
            policy.eval()
            env = make_env(instance=instance, economics=economics)
            obs, _ = env.reset(seed=42)
            with torch.no_grad():
                output = policy.forward(obs, instance, economics, greedy=True)
            vehicle_routes_obj = [
                VehicleRoute(vehicle_id=v, route=r)
                for v, r in enumerate(output["vehicle_routes"])
            ]
            plan = Plan(
                x=output["x"], p=output["p"],
                vehicle_routes=vehicle_routes_obj,
                vehicle_assignments={
                    r: v for v, route in enumerate(output["vehicle_routes"]) for r in route
                },
            )
        except Exception as e:
            st.warning(f"HGA²C not available ({e}), falling back to NN")
            plan, _ = solve_nearest_neighbor(instance, economics)
    else:
        plan, _ = solve_random(instance, economics)

    result = simulate_plan(plan, instance, economics)
    return plan, result


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------
def create_map_figure(
    instance: dict,
    plan: Plan,
    animation_step: int = -1,
) -> go.Figure:
    """Create the main map visualization."""
    fig = go.Figure()
    threshold = instance["battery_threshold"]

    # --- Region boundaries (subtle grid) ---
    regions = instance["regions"]
    for r in regions:
        fig.add_shape(
            type="rect",
            x0=r["x"] - 2.3, y0=r["y"] - 2.3,
            x1=r["x"] + 2.3, y1=r["y"] + 2.3,
            line=dict(color="rgba(100, 100, 255, 0.15)", width=1),
            fillcolor="rgba(100, 100, 255, 0.03)",
        )

    # --- Region centers with demand ---
    region_x = [r["x"] for r in regions]
    region_y = [r["y"] for r in regions]
    region_text = [
        f"R{r['id']}<br>D={r['demand']}<br>Ŝ={r['functional']} S̆={r['non_functional']}"
        for r in regions
    ]
    region_sizes = [max(20, r["demand"] * 12) for r in regions]
    region_colors = ["rgba(100, 200, 255, 0.5)"] * len(regions)

    fig.add_trace(go.Scatter(
        x=region_x, y=region_y,
        mode="markers+text",
        marker=dict(size=region_sizes, color=region_colors, line=dict(color="white", width=1)),
        text=[f"R{r['id']}" for r in regions],
        textposition="top center",
        textfont=dict(color="white", size=11),
        hovertext=region_text,
        hoverinfo="text",
        name="Regions",
    ))

    # --- Hub ---
    hub = instance["hub"]
    fig.add_trace(go.Scatter(
        x=[hub["x"]], y=[hub["y"]],
        mode="markers+text",
        marker=dict(size=18, color="#ffd700", symbol="star", line=dict(color="white", width=2)),
        text=["HUB"],
        textposition="bottom center",
        textfont=dict(color="#ffd700", size=12, family="Arial Black"),
        name="Hub",
    ))

    # --- Scooters ---
    scooters = instance["scooters"]
    for s in scooters:
        color = "#2ecc71" if s["battery"] >= threshold else "#e74c3c"
        # Check if this scooter was swapped
        r = s["region"]
        if plan.x[r] > 0 and s["battery"] < threshold:
            color = "#f39c12"  # swapped (orange → becoming green)

        fig.add_trace(go.Scatter(
            x=[s["x"]], y=[s["y"]],
            mode="markers",
            marker=dict(size=8, color=color, symbol="circle",
                       line=dict(color="white", width=0.5)),
            hovertext=f"{s['id']}<br>Battery: {s['battery']}/{instance['max_battery']}<br>Region: {r}",
            hoverinfo="text",
            showlegend=False,
        ))

    # --- Vehicle routes ---
    vehicle_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    hub_xy = (hub["x"], hub["y"])

    for vr in plan.vehicle_routes:
        if not vr.route:
            continue

        color = vehicle_colors[vr.vehicle_id % len(vehicle_colors)]

        # Build route path: Hub → regions → Hub
        route_x = [hub_xy[0]]
        route_y = [hub_xy[1]]
        for r in vr.route:
            route_x.append(regions[r]["x"])
            route_y.append(regions[r]["y"])
        route_x.append(hub_xy[0])
        route_y.append(hub_xy[1])

        # Determine how much of the route to show based on animation_step
        if animation_step >= 0:
            show_up_to = min(animation_step + 1, len(route_x))
            route_x = route_x[:show_up_to + 1]
            route_y = route_y[:show_up_to + 1]

        fig.add_trace(go.Scatter(
            x=route_x, y=route_y,
            mode="lines+markers",
            line=dict(color=color, width=3, dash="dot"),
            marker=dict(size=6, color=color),
            name=f"Vehicle {vr.vehicle_id}",
        ))

        # Vehicle icon at current position
        if route_x:
            fig.add_trace(go.Scatter(
                x=[route_x[-1]], y=[route_y[-1]],
                mode="markers",
                marker=dict(size=16, color=color, symbol="square",
                           line=dict(color="white", width=2)),
                showlegend=False,
                hovertext=f"Vehicle {vr.vehicle_id}",
                hoverinfo="text",
            ))

    # --- Layout ---
    fig.update_layout(
        title=dict(text="🛴 E-Scooter Battery Swap Map", font=dict(size=20, color="#2c3e50")),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(
            title="X Coordinate", range=[-1, 16],
            gridcolor="#ecf0f1", zeroline=False,
            tickfont=dict(color="#7f8c8d"),
        ),
        yaxis=dict(
            title="Y Coordinate", range=[-1, 16],
            gridcolor="#ecf0f1", zeroline=False,
            scaleanchor="x", scaleratio=1,
            tickfont=dict(color="#7f8c8d"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)", font=dict(color="#2c3e50"),
            bordercolor="#bdc3c7", borderwidth=1,
        ),
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_comparison_chart(all_results: dict[str, PlanResult]) -> go.Figure:
    """Create side-by-side comparison bar chart."""
    methods = list(all_results.keys())
    objectives = [all_results[m].objective_z for m in methods]
    unmet = [all_results[m].total_unmet_demand for m in methods]

    colors = ["#2ecc71", "#e67e22", "#3498db", "#e74c3c", "#9b59b6"]

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Objective Z (lower=better)", "Unmet Demand"])

    fig.add_trace(go.Bar(
        x=methods, y=objectives,
        marker_color=colors[:len(methods)],
        text=[f"{z:.1f}" for z in objectives],
        textposition="auto",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=methods, y=unmet,
        marker_color=colors[:len(methods)],
        text=[f"{u:.0f}" for u in unmet],
        textposition="auto",
    ), row=1, col=2)

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#2c3e50"),
        height=350,
    )

    return fig


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
def main() -> None:
    if "custom_instance" not in st.session_state:
        st.session_state["custom_instance"] = load_instance()

    instance = st.session_state["custom_instance"]
    economics = load_economics()

    # --- Sidebar ---
    st.sidebar.title("🔧 Controls")

    method = st.sidebar.selectbox(
        "Solution Method",
        ["Nearest Neighbor", "Legacy Heuristic", "Random", "MILP", "HGA²C"],
        index=0,
    )

    show_comparison = st.sidebar.checkbox("📊 Show Comparison Mode", value=True)

    speed = st.sidebar.slider("Animation Speed", 0.1, 3.0, 1.0, 0.1)
    anim_step = st.sidebar.slider(
        "Animation Step", -1, 10, -1,
        help="-1 = show complete plan"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎲 Custom Instance Generator")
    
    with st.sidebar.form("instance_generator_form"):
        r_cnt = st.number_input("Region Count", min_value=1, max_value=50, value=instance.get('region_count', 9))
        s_cnt = st.number_input("Scooter Count", min_value=1, max_value=100, value=instance.get('scooter_count', 20))
        v_cnt = st.number_input("Vehicle Count", min_value=1, max_value=10, value=instance.get('vehicle_count', 2))
        max_bat = st.number_input("Max Battery", min_value=1, max_value=20, value=instance.get('max_battery', 5))
        bat_thresh = st.number_input("Battery Threshold", min_value=1, max_value=20, value=instance.get('battery_threshold', 2))
        ex_bat = st.number_input("Extra Batteries", min_value=1, max_value=100, value=instance.get('extra_batteries', 15))
        bat_cap = st.number_input("Battery Carrying Capacity", min_value=1, max_value=50, value=instance.get('battery_carrying_capacity', 5))
        sc_cap = st.number_input("Scooter Carrying Capacity", min_value=1, max_value=50, value=instance.get('scooter_carrying_capacity', 5))
        
        generate_btn = st.form_submit_button("Generate New Instance")
        
        if generate_btn:
            try:
                new_inst = generate_instance(
                    n_regions=r_cnt,
                    n_scooters=s_cnt,
                    n_vehicles=v_cnt,
                    max_battery=max_bat,
                    battery_threshold=bat_thresh,
                    battery_carrying_capacity=bat_cap,
                    scooter_carrying_capacity=sc_cap,
                    extra_batteries_explicit=ex_bat,
                )
                st.session_state["custom_instance"] = new_inst
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate instance: {e}")

    # --- Solve ---
    plan, result = solve_method(method, instance, economics)

    # --- Title ---
    st.markdown(
        "<h1 style='text-align:center;'>🛴 HGA²C E-Scooter Battery Swap Simulator</h1>",
        unsafe_allow_html=True,
    )

    # --- Metrics row ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Objective Z</div>
            <div class="metric-value">{result.objective_z:.1f}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Travel Cost</div>
            <div class="metric-value">{result.travel_cost:.1f}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fulfillment</div>
            <div class="metric-value">{result.demand_fulfillment_rate*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Delay Penalty</div>
            <div class="metric-value">{result.delay_penalty:.1f}</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        batteries_used = sum(plan.x)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Batteries Used</div>
            <div class="metric-value">{batteries_used}/{instance['extra_batteries']}</div>
        </div>""", unsafe_allow_html=True)

    # --- Map ---
    st.plotly_chart(
        create_map_figure(instance, plan, anim_step),
        use_container_width=True,
    )

    # --- Plan details ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📋 Allocation Plan")
        st.markdown(f"**Swaps (x):** `{plan.x}`")
        reloc_info = []
        for r in range(instance["region_count"]):
            for l in range(instance["region_count"]):
                if plan.p[r][l] > 0:
                    reloc_info.append(f"R{r}→R{l}: {plan.p[r][l]}")
        if reloc_info:
            st.markdown(f"**Relocations:** {', '.join(reloc_info)}")
        else:
            st.markdown("**Relocations:** None")

    with col_right:
        st.markdown("### 🚐 Vehicle Routes")
        for vr in plan.vehicle_routes:
            if vr.route:
                route_str = " → ".join([f"R{r}" for r in vr.route])
                st.markdown(f"**Vehicle {vr.vehicle_id}:** Hub → {route_str} → Hub")
            else:
                st.markdown(f"**Vehicle {vr.vehicle_id}:** *Not used*")

    # --- Cost decomposition ---
    st.markdown("### 💰 Cost Decomposition")
    decomp_fig = go.Figure(go.Waterfall(
        name="Cost", orientation="v",
        x=["Travel Cost", "Unmet Demand", "Delay Penalty", "Total Z"],
        y=[result.travel_cost, result.unmet_demand_penalty, result.delay_penalty, 0],
        measure=["relative", "relative", "relative", "total"],
        text=[f"{result.travel_cost:.1f}", f"{result.unmet_demand_penalty:.1f}",
              f"{result.delay_penalty:.1f}", f"{result.objective_z:.1f}"],
        textposition="outside",
        connector=dict(line=dict(color="rgba(200,200,200,0.3)")),
        increasing=dict(marker=dict(color="#e74c3c")),
        decreasing=dict(marker=dict(color="#2ecc71")),
        totals=dict(marker=dict(color="#3498db")),
    ))
    decomp_fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#2c3e50"),
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(decomp_fig, use_container_width=True)

    # --- Comparison mode ---
    if show_comparison:
        st.markdown("---")
        st.markdown("### 📊 Method Comparison")

        all_results: dict[str, PlanResult] = {}
        methods_to_compare = ["Nearest Neighbor", "Legacy Heuristic", "Random"]
        for m in methods_to_compare:
            _, res = solve_method(m, instance, economics)
            all_results[m] = res
        all_results[method] = result

        comp_fig = create_comparison_chart(all_results)
        st.plotly_chart(comp_fig, use_container_width=True)

        # Per-region unmet demand table
        st.markdown("### 📍 Per-Region Unmet Demand")
        import pandas as pd
        region_data = []
        for r in range(instance["region_count"]):
            row = {"Region": f"R{r}", "Demand": instance["regions"][r]["demand"],
                   "Functional": instance["regions"][r]["functional"],
                   "Non-functional": instance["regions"][r]["non_functional"]}
            for m_name, m_result in all_results.items():
                row[f"Unmet ({m_name})"] = m_result.per_region_unmet[r]
            region_data.append(row)
        df = pd.DataFrame(region_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

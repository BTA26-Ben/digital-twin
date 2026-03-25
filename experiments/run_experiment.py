#!/usr/bin/env python3
"""
run_experiment.py
-----------------
Execute a full 24-hour simulation for one or more building agents,
collect metrics, save CSV results, and generate diagnostic plots.

Usage
-----
    python experiments/run_experiment.py
    python experiments/run_experiment.py --n-hours 48 --seed 99 --output-dir results/

Environment variables
---------------------
    DT_SEED          : Override RNG seed (default: 42)
    DT_N_HOURS       : Simulation duration in hours (default: 24)
    DT_OUTPUT_DIR    : Directory for outputs (default: experiments/output)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from digital_twin.agent_coordinator import (
    AgentConfig,
    AgentCoordinator,
    BuildingAgent,
    CoordinatorConfig,
)
from digital_twin.data_store import DataStore
from digital_twin.metrics import compute_metrics, history_to_dataframe, plot_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger("experiment")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Digital Twin 24-hour simulation")
    p.add_argument("--n-hours",     type=int,   default=24,
                   help="Simulation duration in hours (default: 24)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="RNG seed for reproducibility (default: 42)")
    p.add_argument("--outdoor-mean", type=float, default=10.0,
                   help="Mean outdoor temperature °C (default: 10)")
    p.add_argument("--n-agents",    type=int,   default=2,
                   help="Number of building agents (default: 2)")
    p.add_argument("--output-dir",  type=str,   default="experiments/output",
                   help="Output directory for CSV and plots")
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip plot generation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    n_hours:      int   = 24,
    seed:         int   = 42,
    outdoor_mean: float = 10.0,
    n_agents:     int   = 2,
    output_dir:   str   = "experiments/output",
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Run the simulation and return a results dict.

    Parameters
    ----------
    n_hours       : Simulation duration.
    seed          : Master RNG seed (each agent gets seed+i).
    outdoor_mean  : Mean outdoor temperature for the daily profile.
    n_agents      : Number of building agents.
    output_dir    : Where to write CSV files and plots.
    generate_plots: Whether to call matplotlib.

    Returns
    -------
    Dict with 'metrics', 'experiment_id', 'csv_path', 'plot_paths'.
    """
    os.makedirs(output_dir, exist_ok=True)

    experiment_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_seed{seed}"
    log.info(f"Starting experiment {experiment_id}")
    log.info(f"  n_hours={n_hours}  seed={seed}  n_agents={n_agents}  T_out_mean={outdoor_mean}°C")

    # ---- Set up agents ----
    agents = [
        BuildingAgent(AgentConfig(f"building_{chr(ord('a') + i)}", seed=seed + i))
        for i in range(n_agents)
    ]
    store = DataStore()
    coord = AgentCoordinator(
        agents,
        CoordinatorConfig(campus_power_limit_kw=200.0),
    )

    # ---- Simulation loop ----
    dt_s  = 300.0   # 5-minute timesteps
    n_steps = n_hours * 12   # 12 steps per hour
    rng   = np.random.default_rng(seed)

    t_wall_start = time.perf_counter()
    log.info(f"Running {n_steps} timesteps …")

    for step in range(n_steps):
        t = float(step * dt_s)
        # Daily outdoor temperature profile + small random perturbation
        T_out = (
            outdoor_mean
            + 5.0 * np.sin(2 * np.pi * t / 86400 - np.pi / 2)
            + float(rng.normal(0.0, 0.5))
        )
        day_of_year = (step * int(dt_s) // 86400) % 365 + 1

        results = coord.step(
            unix_time=t,
            outdoor_temp=float(T_out),
            day_of_year=day_of_year,
        )

        # Persist to store
        for agent in agents:
            if agent.current_state:
                store.append_agent_state(agent.current_state)
            if agent.agent_id in results:
                store.append_opt_result(results[agent.agent_id])

        if (step + 1) % 12 == 0:
            elapsed = time.perf_counter() - t_wall_start
            log.info(f"  Step {step + 1}/{n_steps} ({100*(step+1)//n_steps}%)  wall={elapsed:.1f}s")

    t_wall_total = time.perf_counter() - t_wall_start
    log.info(f"Simulation complete in {t_wall_total:.2f}s")

    # ---- Aggregate metrics per agent ----
    per_agent_metrics: Dict[str, Dict] = {}
    all_history = []
    for agent in agents:
        m = compute_metrics(agent.history)
        per_agent_metrics[agent.agent_id] = m
        all_history.extend(agent.history)
        log.info(f"  {agent.agent_id}: energy={m['total_energy_kwh']:.1f} kWh  "
                 f"cost=£{m['total_cost_gbp']:.2f}  "
                 f"comfort_viol={m['comfort_violations_dm']:.1f} °C·min  "
                 f"avg_solve={m['avg_solve_time_s']*1000:.1f} ms  "
                 f"infeasible={m['n_infeasible']}")
    all_history.sort(key=lambda x: x["timestamp"])
    campus_metrics = compute_metrics(all_history)

    # ---- Save CSV ----
    csv_path = os.path.join(output_dir, f"{experiment_id}_metrics.csv")
    df = history_to_dataframe(all_history)
    df.to_csv(csv_path, index=False)
    log.info(f"Metrics CSV saved: {csv_path}")

    # ---- Save experiment record JSON ----
    record: Dict[str, Any] = {
        "experiment_id":  experiment_id,
        "git_commit":     _get_git_commit(),
        "seed":           seed,
        "n_hours":        n_hours,
        "n_agents":       n_agents,
        "outdoor_mean":   outdoor_mean,
        "campus_metrics": campus_metrics,
        "per_agent":      per_agent_metrics,
        "wall_time_s":    round(t_wall_total, 2),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }
    json_path = os.path.join(output_dir, f"{experiment_id}.json")
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)
    log.info(f"Experiment record saved: {json_path}")

    # ---- Generate plots ----
    plot_paths: list = []
    if generate_plots:
        plot_dir = os.path.join(output_dir, experiment_id)
        # Plot per agent
        for agent in agents:
            agent_plot_dir = os.path.join(plot_dir, agent.agent_id)
            paths = plot_results(agent.history, output_dir=agent_plot_dir)
            plot_paths.extend(paths)
        log.info(f"Generated {len(plot_paths)} plot(s) in {plot_dir!r}")

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print(f"Experiment: {experiment_id}")
    print("=" * 60)
    print(f"  Total energy:     {campus_metrics['total_energy_kwh']:.2f} kWh")
    print(f"  Total cost:       £{campus_metrics['total_cost_gbp']:.4f}")
    print(f"  Total carbon:     {campus_metrics['total_carbon_kg']:.2f} kgCO₂")
    print(f"  Comfort viol.:    {campus_metrics['comfort_violations_dm']:.1f} °C·min")
    print(f"  Avg solve time:   {campus_metrics['avg_solve_time_s']*1000:.1f} ms")
    print(f"  Peak power:       {campus_metrics['peak_power_kw']:.1f} kW")
    print(f"  Infeasible steps: {campus_metrics['n_infeasible']}")
    print(f"  Wall time:        {t_wall_total:.2f}s")
    print(f"  CSV:              {csv_path}")
    print("=" * 60 + "\n")

    return {
        "metrics":       campus_metrics,
        "experiment_id": experiment_id,
        "csv_path":      csv_path,
        "json_path":     json_path,
        "plot_paths":    plot_paths,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        n_hours=args.n_hours,
        seed=args.seed,
        outdoor_mean=args.outdoor_mean,
        n_agents=args.n_agents,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
    )

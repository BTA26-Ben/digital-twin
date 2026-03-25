"""
metrics.py
----------
Compute evaluation metrics from simulation history and produce reports.

Metrics
-------
- total_energy_kwh      : Sum of all HVAC + flex load energy consumed.
- total_cost_gbp        : Energy cost at grid tariff.
- comfort_violations_dm : Comfort violations in degree-minutes (area above/below bounds).
- avg_solve_time_s      : Mean MPC solve time.
- n_infeasible          : Number of infeasible or fallback solves.
- peak_power_kw         : Maximum instantaneous total power draw.
- carbon_kg             : Total carbon emitted.

Public API
----------
compute_metrics(history, config) → dict
plot_results(history, output_dir)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    history:       List[Dict[str, Any]],
    dt_seconds:    float = 300.0,
    tariff:        float = 0.28,
    carbon_factor: float = 200.0,
    constraints_min: float = 20.0,
    constraints_max: float = 24.0,
) -> Dict[str, float]:
    """Compute summary metrics from agent history list.

    Parameters
    ----------
    history          : List of dicts from BuildingAgent.history.
    dt_seconds       : Timestep duration for energy integration.
    tariff           : £/kWh.
    carbon_factor    : gCO2/kWh.
    constraints_min  : Occupied comfort minimum [°C].
    constraints_max  : Occupied comfort maximum [°C].

    Returns
    -------
    Dict of scalar metric values.
    """
    if not history:
        return {}

    dt_h = dt_seconds / 3600.0  # hours per timestep

    total_energy    = 0.0
    total_cost      = 0.0
    total_carbon    = 0.0
    comfort_viol_dm = 0.0
    solve_times     = []
    n_infeasible    = 0
    peak_power      = 0.0

    for row in history:
        hvac_kw  = sum(row.get("hvac_kw", {}).values())
        energy   = hvac_kw * dt_h
        total_energy += energy
        total_cost   += energy * tariff
        total_carbon += energy * carbon_factor / 1000.0  # gCO2 → kgCO2

        peak_power = max(peak_power, hvac_kw)

        # Comfort violations
        for zid, T in row.get("zone_temps", {}).items():
            occ = row.get("occupancy", {}).get(zid, 0)
            if occ > 0:
                lo, hi = constraints_min, constraints_max
            else:
                lo, hi = 16.0, 28.0
            viol = max(0.0, lo - T) + max(0.0, T - hi)  # °C deviation
            comfort_viol_dm += viol * dt_seconds / 60.0  # degree-minutes

        st = row.get("solve_time_s")
        if st is not None:
            solve_times.append(st)

        status = row.get("mpc_status", "optimal")
        if status not in ("optimal", "optimal_inaccurate"):
            n_infeasible += 1

    return {
        "total_energy_kwh":      round(total_energy, 3),
        "total_cost_gbp":        round(total_cost, 4),
        "total_carbon_kg":       round(total_carbon, 3),
        "comfort_violations_dm": round(comfort_viol_dm, 2),
        "avg_solve_time_s":      round(float(np.mean(solve_times)), 4) if solve_times else 0.0,
        "max_solve_time_s":      round(float(np.max(solve_times)), 4) if solve_times else 0.0,
        "n_infeasible":          n_infeasible,
        "peak_power_kw":         round(peak_power, 2),
        "n_timesteps":           len(history),
    }


def history_to_dataframe(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten agent history list into a wide-format DataFrame."""
    rows = []
    for row in history:
        flat: Dict[str, Any] = {"timestamp": row["timestamp"]}
        for k, v in row.get("zone_temps", {}).items():
            flat[f"temp_{k}"] = v
        for k, v in row.get("estimated_temps", {}).items():
            flat[f"est_{k}"] = v
        for k, v in row.get("occupancy", {}).items():
            flat[f"occ_{k}"] = v
        for k, v in row.get("hvac_kw", {}).items():
            flat[f"hvac_{k}_kw"] = v
        flat["outdoor_temp"]  = row.get("outdoor_temp", 0.0)
        flat["solve_time_s"]  = row.get("solve_time_s", 0.0)
        flat["mpc_status"]    = row.get("mpc_status", "")
        flat["objective"]     = row.get("objective")
        rows.append(flat)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def plot_results(
    history:    List[Dict[str, Any]],
    output_dir: str = ".",
    show:       bool = False,
) -> List[str]:
    """Generate and save diagnostic plots.

    Returns
    -------
    List of file paths where plots were saved.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        log.warning("matplotlib not installed; skipping plot generation.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    df = history_to_dataframe(history)
    if df.empty:
        return []

    # Convert timestamp to datetime for x-axis
    import datetime
    times = [datetime.datetime.utcfromtimestamp(t) for t in df["timestamp"]]

    saved = []

    # ---- Plot 1: Zone temperatures ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Zone Temperatures vs Comfort Bounds", fontsize=13)
    for ax, col, label in zip(
        axes,
        ["temp_zone_a", "temp_zone_b"],
        ["Zone A", "Zone B"],
    ):
        if col not in df.columns:
            continue
        ax.plot(times, df[col], label=f"{label} (simulated)", linewidth=1.5, color="steelblue")
        if f"est_{col.replace('temp_', '')}" in df.columns:
            ax.plot(times, df[f"est_{col.replace('temp_', '')}"],
                    label=f"{label} (estimated)", linestyle="--", color="orange", linewidth=1)
        ax.axhline(20.0, color="green",  linestyle=":", linewidth=1, label="T_min occupied")
        ax.axhline(24.0, color="red",    linestyle=":", linewidth=1, label="T_max occupied")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(label)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    p = os.path.join(output_dir, "zone_temperatures.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ---- Plot 2: HVAC power ----
    fig, ax = plt.subplots(figsize=(12, 4))
    for col in [c for c in df.columns if c.startswith("hvac_")]:
        ax.plot(times, df[col], label=col, linewidth=1.5)
    ax.set_ylabel("Power (kW)")
    ax.set_title("HVAC Power Schedule")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    p = os.path.join(output_dir, "hvac_power.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ---- Plot 3: Solver performance ----
    if "solve_time_s" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(len(df)), df["solve_time_s"] * 1000, color="teal", alpha=0.7)
        ax.axhline(5000, color="red", linestyle="--", label="5 s limit")
        ax.set_ylabel("Solve time (ms)")
        ax.set_xlabel("Timestep")
        ax.set_title("MPC Solver Time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        p = os.path.join(output_dir, "solver_times.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # ---- Plot 4: Outdoor temperature ----
    if "outdoor_temp" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(times, df["outdoor_temp"], color="purple", linewidth=1.5)
        ax.set_ylabel("Outdoor Temp (°C)")
        ax.set_title("Outdoor Temperature Profile")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()
        p = os.path.join(output_dir, "outdoor_temp.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    log.info(f"Saved {len(saved)} plots to {output_dir!r}")
    return saved

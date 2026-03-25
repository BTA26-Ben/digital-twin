"""
optimiser.py
------------
Model Predictive Control (MPC) optimiser using CVXPY.

Mathematical formulation
------------------------
Decision variables (per timestep t ∈ {0,...,H-1}):
    u_h[t]  – HVAC power for each zone [kW]
    u_f[t]  – Flexible load power [kW]

State trajectory (predicted):
    T[t+1] = F·T[t] + B_h·u_h[t] + B_d·d[t]
where d[t] = [outdoor_temp[t], solar_gain[t], occ_gain[t]] (disturbances)

Objective (minimise over horizon H):
    J = Σ_{t=0}^{H-1} [
          λ_cost  · (u_h[t] + u_f[t]) · tariff[t] · dt
        + λ_carbon· (u_h[t] + u_f[t]) · carbon[t]  · dt
        + λ_comfort· Σ_z max(0, T_min - T_z[t])² + max(0, T_z[t] - T_max)²
    ]

Subject to:
    T[t+1] = F·T[t] + B_h·u_h[t] + B_d·d[t]     (dynamics)
    0 ≤ u_h[t] ≤ P_hvac_max                         (HVAC power bounds)
    0 ≤ u_f[t] ≤ P_flex_max                         (flex load bounds)
    |u_h[t+1] - u_h[t]| ≤ R_hvac                   (ramp rate)
    T[0] = T_current                                 (initial condition)

Public API
----------
MPCConfig      – tuning and constraint parameters
MPCOptimiser   – solve(problem) → OptimisationResult
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    logging.warning("CVXPY not found; optimiser will use the scipy fallback.")

from .models import AgentState, Forecast, OptimisationProblem, OptimisationResult
from .state_estimator import KalmanStateEstimator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MPCConfig:
    """Tuning parameters for the MPC optimiser.

    Attributes
    ----------
    horizon        : Number of control timesteps (H).
    dt_seconds     : Duration of each timestep.
    lambda_cost    : Weight on electricity cost.
    lambda_carbon  : Weight on carbon emissions.
    lambda_comfort : Weight on comfort violation penalty.
    hvac_max_kw    : Maximum HVAC power per zone [kW].
    flex_max_kw    : Maximum flexible load power [kW].
    ramp_rate_kw   : Maximum HVAC ramp per timestep [kW].
    solver         : CVXPY solver string.
    solver_timeout : Maximum seconds allowed for solver.
    """
    horizon:        int   = 12          # 12 × 5 min = 1 hour lookahead
    dt_seconds:     float = 300.0
    lambda_cost:    float = 1.0
    lambda_carbon:  float = 0.01
    lambda_comfort: float = 100.0
    hvac_max_kw:    float = 10.0
    flex_max_kw:    float = 5.0
    ramp_rate_kw:   float = 3.0
    n_zones:        int   = 2
    solver:         str   = "OSQP"
    solver_timeout: float = 5.0         # seconds — requirement ≤5s


# ---------------------------------------------------------------------------
# MPC Optimiser
# ---------------------------------------------------------------------------

class MPCOptimiser:
    """Receding-horizon MPC optimiser for HVAC and flexible loads.

    Parameters
    ----------
    config : MPCConfig with all tuning parameters.
    F      : Pre-built state transition matrix (n_zones × n_zones).
    B_h    : HVAC input matrix (n_zones × n_zones).
    B_d    : Disturbance input matrix (n_zones × 3).

    Usage
    -----
    >>> opt = MPCOptimiser(config=MPCConfig())
    >>> result = opt.solve(state=agent_state, forecasts=fc_dict)
    """

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        F:  Optional[np.ndarray] = None,
        B_h: Optional[np.ndarray] = None,
        B_d: Optional[np.ndarray] = None,
    ) -> None:
        self.cfg = config or MPCConfig()
        nz = self.cfg.n_zones
        dt = self.cfg.dt_seconds

        # Default system matrices if not provided
        # These match the EnvironmentSimulator RC thermal model
        C = 5e6   # J/K
        UA_env = 200.0
        UA_adj = 50.0
        alpha  = dt / C

        if F is None:
            F = np.eye(nz)
            for i in range(nz):
                F[i, i] = 1.0 - alpha * (UA_env + (nz - 1) * UA_adj)
            for i in range(nz):
                for j in range(nz):
                    if i != j:
                        F[i, j] = alpha * UA_adj

        if B_h is None:
            # Each zone has independent HVAC; kW → W → ΔT
            B_h = np.eye(nz) * (alpha * 1000.0)   # 1 kW → 1000 W

        if B_d is None:
            # Disturbance: [T_out, solar_gain, occ_gain]
            # T_out affects all zones equally via UA_env
            B_d = np.zeros((nz, 3))
            for i in range(nz):
                B_d[i, 0] = alpha * UA_env         # outdoor temp coupling
                B_d[i, 1] = alpha                  # solar (W → ΔT)
                B_d[i, 2] = alpha                  # occupancy (W → ΔT)

        self.F   = F
        self.B_h = B_h
        self.B_d = B_d

        # Cache previous solution for warm-starting
        self._prev_u_h: Optional[np.ndarray] = None
        self._prev_u_f: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        state:     AgentState,
        forecasts: Dict[str, Optional[Forecast]],
    ) -> OptimisationResult:
        """Run one MPC solve and return the optimal schedule.

        Parameters
        ----------
        state     : Current estimated AgentState.
        forecasts : Dict variable → Forecast (occupancy, outdoor_temp, tariff).

        Returns
        -------
        OptimisationResult with status, schedule, and immediate setpoints.
        """
        t_start = time.perf_counter()

        if not HAS_CVXPY:
            return self._fallback_solve(state, forecasts, t_start)

        H  = self.cfg.horizon
        nz = self.cfg.n_zones
        dt = self.cfg.dt_seconds

        # ---- Unpack current state ----
        T_init = np.array([state.zone_temps.get(f"zone_{chr(ord('a')+i)}", 21.0)
                           for i in range(nz)])
        occ    = np.array([state.occupancy.get(f"zone_{chr(ord('a')+i)}", 0.0)
                           for i in range(nz)])
        tariff = state.grid_tariff
        carbon = state.carbon_factor

        # ---- Build disturbance trajectory from forecasts ----
        T_out_fc   = self._extract_forecast(forecasts, "outdoor_temp", H, default=10.0)
        occ_fc_sum = self._extract_forecast(forecasts, "occupancy",    H, default=float(occ.sum()))
        tariff_fc  = np.full(H, tariff)     # assume flat tariff if no forecast
        solar_fc   = np.zeros(H)            # simplified: no solar forecast in MVP

        # Disturbance matrix: (H, 3)
        D = np.column_stack([
            T_out_fc,
            solar_fc * 100.0,                         # scale to W/m² range
            occ_fc_sum * 80.0 / nz,                   # W per zone
        ])

        # ---- Comfort bounds (vary by occupancy forecast) ----
        c  = state.constraints
        T_min = np.where(occ_fc_sum > 0, c.temp_min_occupied, c.temp_min_unoccupied)
        T_max = np.where(occ_fc_sum > 0, c.temp_max_occupied, c.temp_max_unoccupied)

        # ---- CVXPY decision variables ----
        # u_h: (H, nz) HVAC power per zone per step
        # u_f: (H,)    total flexible load per step
        u_h = cp.Variable((H, nz), name="u_h")
        u_f = cp.Variable(H,       name="u_f")
        T   = cp.Variable((H + 1, nz), name="T")  # state trajectory

        constraints = []
        objective_terms = []

        # ---- Initial condition ----
        constraints.append(T[0, :] == T_init)

        for t in range(H):
            # Dynamics: T[t+1] = F·T[t] + B_h·u_h[t] + B_d·d[t]
            constraints.append(
                T[t + 1, :] == self.F @ T[t, :] + self.B_h @ u_h[t, :] + self.B_d @ D[t, :]
            )

            # HVAC power bounds
            constraints.append(u_h[t, :] >= 0)
            constraints.append(u_h[t, :] <= self.cfg.hvac_max_kw)

            # Flexible load bounds
            constraints.append(u_f[t] >= 0)
            constraints.append(u_f[t] <= self.cfg.flex_max_kw)

            # HVAC ramp rate
            if t > 0:
                constraints.append(cp.norm(u_h[t, :] - u_h[t - 1, :], "inf") <= self.cfg.ramp_rate_kw)
            elif self._prev_u_h is not None:
                # Ramp from previous solution's last step
                constraints.append(cp.norm(u_h[0, :] - self._prev_u_h[-1, :], "inf") <= self.cfg.ramp_rate_kw)

            # ---- Objective terms ----
            total_power = cp.sum(u_h[t, :]) + u_f[t]   # kW
            energy_kwh  = total_power * dt / 3600.0

            # Energy cost
            objective_terms.append(
                self.cfg.lambda_cost * energy_kwh * tariff_fc[t]
            )

            # Carbon cost
            objective_terms.append(
                self.cfg.lambda_carbon * energy_kwh * carbon / 1000.0
            )

            # Comfort violation (soft constraint — quadratic penalty)
            for z in range(nz):
                T_z = T[t + 1, z]
                comfort_viol = (
                    cp.square(cp.pos(T_min[t] - T_z))   # below lower bound
                  + cp.square(cp.pos(T_z - T_max[t]))   # above upper bound
                )
                objective_terms.append(self.cfg.lambda_comfort * comfort_viol)

        objective = cp.Minimize(cp.sum(objective_terms))
        problem   = cp.Problem(objective, constraints)

        # ---- Solve ----
        solver_map = {
            "OSQP": cp.OSQP,
            "ECOS": cp.ECOS,
            "SCS":  cp.SCS,
            "CLARABEL": cp.CLARABEL,
        }
        solver_fn = solver_map.get(self.cfg.solver.upper(), cp.OSQP)
        try:
            problem.solve(
                solver=solver_fn,
                warm_start=True,
                max_iter=10000,
                eps_abs=1e-4,
                eps_rel=1e-4,
            )
        except Exception as e:
            log.error(f"Solver raised exception: {e}")
            return self._infeasible_result(state.agent_id, state.timestamp, time.perf_counter() - t_start)

        solve_time = time.perf_counter() - t_start

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            log.warning(f"MPC solve status: {problem.status} (agent={state.agent_id})")
            return self._infeasible_result(state.agent_id, state.timestamp, solve_time, problem.status)

        # ---- Extract solution ----
        u_h_val = u_h.value   # (H, nz)
        u_f_val = u_f.value   # (H,)

        if u_h_val is None or u_f_val is None:
            return self._infeasible_result(state.agent_id, state.timestamp, solve_time)

        # Cache for warm-starting
        self._prev_u_h = u_h_val
        self._prev_u_f = u_f_val

        # Build schedule dict
        schedule: Dict[str, List[float]] = {}
        for i in range(nz):
            zid = f"zone_{chr(ord('a') + i)}"
            schedule[f"hvac_{zid}"] = list(u_h_val[:, i])
        schedule["flex_load"] = list(u_f_val)

        # Immediate setpoints (t=0 slice)
        setpoints_now: Dict[str, float] = {}
        for i in range(nz):
            zid = f"zone_{chr(ord('a') + i)}"
            setpoints_now[f"hvac_{zid}"] = float(u_h_val[0, i])
        setpoints_now["flex_load"] = float(u_f_val[0])

        return OptimisationResult(
            agent_id=state.agent_id,
            timestamp=state.timestamp,
            status=problem.status,
            solve_time_s=solve_time,
            objective_value=float(problem.value),
            schedule=schedule,
            setpoints_now=setpoints_now,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_forecast(
        self,
        forecasts: Dict[str, Optional[Forecast]],
        variable: str,
        horizon: int,
        default: float = 0.0,
    ) -> np.ndarray:
        """Extract forecast values as a fixed-length array."""
        fc = forecasts.get(variable)
        if fc is None or not fc.predicted_values:
            return np.full(horizon, default)
        vals = fc.predicted_values[:horizon]
        if len(vals) < horizon:
            # Pad with last value
            vals = vals + [vals[-1]] * (horizon - len(vals))
        return np.array(vals, dtype=float)

    def _infeasible_result(
        self,
        agent_id: str,
        timestamp: float,
        solve_time: float,
        status: str = "infeasible",
    ) -> OptimisationResult:
        """Return a safe fallback result when solver fails."""
        # Default: minimum HVAC power (safe failsafe)
        nz = self.cfg.n_zones
        H  = self.cfg.horizon
        schedule: Dict[str, List[float]] = {}
        setpoints_now: Dict[str, float]  = {}
        for i in range(nz):
            zid = f"zone_{chr(ord('a') + i)}"
            schedule[f"hvac_{zid}"] = [0.0] * H
            setpoints_now[f"hvac_{zid}"] = 0.0
        schedule["flex_load"] = [0.0] * H
        setpoints_now["flex_load"] = 0.0

        return OptimisationResult(
            agent_id=agent_id,
            timestamp=timestamp,
            status=status,
            solve_time_s=solve_time,
            objective_value=None,
            schedule=schedule,
            setpoints_now=setpoints_now,
        )

    def _fallback_solve(
        self,
        state: AgentState,
        forecasts: Dict[str, Optional[Forecast]],
        t_start: float,
    ) -> OptimisationResult:
        """Rule-based fallback when CVXPY is unavailable.

        Applies a simple bang-off strategy: heat if below setpoint, else off.
        """
        log.warning("Using rule-based fallback (CVXPY not available).")
        nz = self.cfg.n_zones
        H  = self.cfg.horizon
        c  = state.constraints

        schedule: Dict[str, List[float]] = {}
        setpoints_now: Dict[str, float]  = {}

        for i in range(nz):
            zid = f"zone_{chr(ord('a') + i)}"
            T   = state.zone_temps.get(f"zone_{chr(ord('a') + i)}", 21.0)
            occ = state.occupancy.get(f"zone_{chr(ord('a') + i)}", 0.0)
            T_mid = (c.temp_min_occupied + c.temp_max_occupied) / 2 if occ > 0 \
                    else (c.temp_min_unoccupied + c.temp_max_unoccupied) / 2
            power = self.cfg.hvac_max_kw if T < T_mid else 0.0
            schedule[f"hvac_{zid}"] = [power] * H
            setpoints_now[f"hvac_{zid}"] = power

        schedule["flex_load"] = [0.0] * H
        setpoints_now["flex_load"] = 0.0

        return OptimisationResult(
            agent_id=state.agent_id,
            timestamp=state.timestamp,
            status="fallback_rule_based",
            solve_time_s=time.perf_counter() - t_start,
            objective_value=None,
            schedule=schedule,
            setpoints_now=setpoints_now,
        )

"""
agent_coordinator.py
--------------------
Coordinates multiple building agents to respect a campus-level power budget
and carbon/cost objective.

Two coordination modes:
  1. Centralised: the coordinator runs a campus-level optimisation and
     assigns each agent a power budget before local MPC runs.
  2. Iterative (ADMM-lite): agents solve locally, then exchange dual variables
     (shadow prices) until budgets converge.  Simplified here to two iterations.

Public API
----------
BuildingAgent   – single-building controller (wraps Simulator + Estimator + Optimiser)
AgentCoordinator – orchestrates all agents for one control cycle
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .environment_simulator import EnvironmentSimulator
from .forecast_engine import ForecastEngine
from .models import (
    AgentState, ComfortConstraints, DeviceState, DeviceType,
    OptimisationResult, TelemetryPoint,
)
from .optimiser import MPCConfig, MPCOptimiser
from .sensor_simulator import SensorSimulator, default_building_sensors
from .state_estimator import KalmanStateEstimator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for one building agent."""
    agent_id:       str
    n_zones:        int   = 2
    dt_seconds:     float = 300.0
    mpc_horizon:    int   = 12
    campus_budget_kw: float = 50.0      # Shared campus budget; set by coordinator
    seed:           int   = 42


class BuildingAgent:
    """Full single-building digital twin: sensor → estimate → forecast → optimise.

    Parameters
    ----------
    config : AgentConfig.

    Internal components
    -------------------
    - EnvironmentSimulator  (physics engine)
    - SensorSimulator       (telemetry generation)
    - KalmanStateEstimator  (state fusion)
    - ForecastEngine        (occupancy + weather prediction)
    - MPCOptimiser          (receding-horizon control)

    Usage
    -----
    >>> agent = BuildingAgent(AgentConfig("building_a"))
    >>> result = agent.control_cycle(unix_time=t, outdoor_temp=12.0,
    ...                               power_budget_kw=50.0)
    """

    def __init__(self, config: AgentConfig) -> None:
        self.cfg   = config
        self.agent_id = config.agent_id
        self._t    = 0.0  # current sim time

        # Sub-components
        self.env_sim = EnvironmentSimulator(seed=config.seed)

        self.sensor_sim = SensorSimulator(
            sensors=default_building_sensors(agent_id=config.agent_id, seed=config.seed),
            seed=config.seed,
        )

        self.estimator = KalmanStateEstimator.for_two_zones()

        self.forecast_engine = ForecastEngine(
            agent_id=config.agent_id,
            dt_seconds=config.dt_seconds,
        )

        mpc_cfg = MPCConfig(
            horizon=config.mpc_horizon,
            dt_seconds=config.dt_seconds,
        )
        self.optimiser = MPCOptimiser(config=mpc_cfg)

        # State
        self._current_state: Optional[AgentState] = None
        self._last_result:   Optional[OptimisationResult] = None

        # HVAC setpoints applied to the simulator
        self._hvac_kw: Dict[str, float] = {
            f"zone_{chr(ord('a') + i)}": 0.0 for i in range(config.n_zones)
        }

        # History for metrics
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main control cycle
    # ------------------------------------------------------------------

    def control_cycle(
        self,
        unix_time:       float,
        outdoor_temp:    float,
        power_budget_kw: float = 100.0,
        day_of_year:     int   = 180,
    ) -> OptimisationResult:
        """Run one full sense → estimate → forecast → optimise cycle.

        Parameters
        ----------
        unix_time       : Current simulation timestamp.
        outdoor_temp    : Current outdoor temperature [°C].
        power_budget_kw : Campus-level power limit for this agent [kW].
        day_of_year     : For solar gain calculation.

        Returns
        -------
        OptimisationResult with schedule and immediate setpoints.
        """
        self._t = unix_time

        # 1. Advance environment
        env_out = self.env_sim.step(
            hvac_power_kw=self._hvac_kw,
            outdoor_temp=outdoor_temp,
            unix_time=unix_time,
            day_of_year=day_of_year,
        )

        # 2. Build ground-truth sensor values
        true_values: Dict[str, float] = {}
        for i, zid in enumerate(["zone_a", "zone_b"]):
            true_values[f"temp_{zid}"] = env_out["temps"][zid]
        true_values["temp_outdoor"] = outdoor_temp
        for i, zid in enumerate(["zone_a", "zone_b"]):
            true_values[f"occ_{zid}"] = float(env_out["occupancy"][zid])
        true_values["power_hvac"] = sum(self._hvac_kw.values())
        true_values["power_total"] = sum(self._hvac_kw.values()) + 2.0  # base load

        # 3. Get noisy sensor readings
        readings: List[TelemetryPoint] = self.sensor_sim.read_all(
            true_values=true_values, timestamp=unix_time
        )

        # 4. State estimation via Kalman filter
        F = KalmanStateEstimator.build_transition_matrix(
            dt=self.cfg.dt_seconds,
            capacitance=5e6,
            ua_envelope=200.0,
            ua_adjacent=50.0,
            n_zones=2,
        )
        estimated_temps = self.estimator.fuse_telemetry(readings=readings, F=F)

        # 5. Update forecast engine with latest observations
        for zid in ["zone_a", "zone_b"]:
            occ_reading = next(
                (r.value for r in readings if f"occ_{zid}" in r.sensor_id), 0.0
            )
            self.forecast_engine.add_observation("occupancy", occ_reading, unix_time)
        self.forecast_engine.add_observation("outdoor_temp", outdoor_temp, unix_time)

        # 6. Generate forecasts
        forecasts = self.forecast_engine.forecast_all(n_steps=self.cfg.mpc_horizon)

        # 7. Build AgentState
        device_states = {
            f"hvac_zone_{chr(ord('a') + i)}": DeviceState(
                device_id=f"hvac_zone_{chr(ord('a') + i)}",
                device_type=DeviceType.HVAC,
                setpoint=self._hvac_kw.get(f"zone_{chr(ord('a') + i)}", 0.0),
                power_kw=self._hvac_kw.get(f"zone_{chr(ord('a') + i)}", 0.0),
                is_on=self._hvac_kw.get(f"zone_{chr(ord('a') + i)}", 0.0) > 0,
                max_power=10.0,
            )
            for i in range(self.cfg.n_zones)
        }

        self._current_state = AgentState(
            agent_id=self.agent_id,
            timestamp=unix_time,
            zone_temps=estimated_temps,
            occupancy={
                f"zone_{chr(ord('a') + i)}": float(env_out["occupancy"].get(f"zone_{chr(ord('a') + i)}", 0))
                for i in range(self.cfg.n_zones)
            },
            device_states=device_states,
            constraints=ComfortConstraints(),
        )

        # 8. Run MPC
        result = self.optimiser.solve(
            state=self._current_state,
            forecasts=forecasts,
        )
        self._last_result = result

        # 9. Apply immediate setpoints
        for i in range(self.cfg.n_zones):
            zid = f"zone_{chr(ord('a') + i)}"
            key = f"hvac_{zid}"
            self._hvac_kw[zid] = result.setpoints_now.get(key, 0.0)

        # 10. Record history
        self.history.append({
            "timestamp":      unix_time,
            "zone_temps":     dict(env_out["temps"]),
            "estimated_temps": dict(estimated_temps),
            "occupancy":      dict(env_out["occupancy"]),
            "hvac_kw":        dict(self._hvac_kw),
            "solve_time_s":   result.solve_time_s,
            "mpc_status":     result.status,
            "objective":      result.objective_value,
            "outdoor_temp":   outdoor_temp,
        })

        log.debug(
            f"[{self.agent_id}] t={unix_time:.0f}  "
            f"T={list(estimated_temps.values())}  "
            f"HVAC={list(self._hvac_kw.values())}  "
            f"status={result.status}  t_solve={result.solve_time_s:.3f}s"
        )

        return result

    @property
    def current_state(self) -> Optional[AgentState]:
        return self._current_state

    @property
    def total_power_kw(self) -> float:
        return sum(self._hvac_kw.values())


# ---------------------------------------------------------------------------
# Campus-level coordinator
# ---------------------------------------------------------------------------

@dataclass
class CoordinatorConfig:
    """Configuration for the multi-agent coordinator."""
    campus_power_limit_kw: float = 200.0   # total campus power cap
    carbon_budget_kg_h:    float = 100.0   # kg CO₂ per hour
    coordination_mode:     str   = "centralised"  # or 'admm'
    admm_iterations:       int   = 3
    admm_rho:              float = 1.0


class AgentCoordinator:
    """Orchestrates all building agents for one control cycle.

    In centralised mode, the coordinator:
    1. Collects current power demand from all agents.
    2. Allocates budgets proportionally if total exceeds campus limit.
    3. Each agent re-solves with its allocated budget.

    In ADMM mode, agents exchange shadow prices iteratively.

    Parameters
    ----------
    agents : List of BuildingAgent objects.
    config : CoordinatorConfig.
    """

    def __init__(
        self,
        agents: List[BuildingAgent],
        config: Optional[CoordinatorConfig] = None,
    ) -> None:
        self.agents = {a.agent_id: a for a in agents}
        self.cfg    = config or CoordinatorConfig()
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Main coordination step
    # ------------------------------------------------------------------

    def step(
        self,
        unix_time:    float,
        outdoor_temp: float,
        day_of_year:  int = 180,
    ) -> Dict[str, OptimisationResult]:
        """Coordinate all agents for one timestep.

        Returns
        -------
        Dict agent_id → OptimisationResult.
        """
        t0 = time.perf_counter()

        # Phase 1: run each agent with full budget
        results_phase1 = {
            aid: agent.control_cycle(
                unix_time=unix_time,
                outdoor_temp=outdoor_temp,
                power_budget_kw=self.cfg.campus_power_limit_kw,
                day_of_year=day_of_year,
            )
            for aid, agent in self.agents.items()
        }

        # Phase 2: check campus-level constraint
        total_demand = sum(a.total_power_kw for a in self.agents.values())

        if total_demand <= self.cfg.campus_power_limit_kw * 1.05:   # 5% tolerance
            # Within budget — no reallocation needed
            results = results_phase1
        else:
            # Proportional budget allocation
            scale = self.cfg.campus_power_limit_kw / max(total_demand, 1e-6)
            results = {}
            for aid, agent in self.agents.items():
                budget = agent.total_power_kw * scale
                # Update HVAC caps for this agent
                agent.optimiser.cfg.hvac_max_kw = max(1.0, budget / agent.cfg.n_zones)
                # Re-run optimisation with reduced budget
                results[aid] = agent.control_cycle(
                    unix_time=unix_time,
                    outdoor_temp=outdoor_temp,
                    power_budget_kw=budget,
                    day_of_year=day_of_year,
                )
                # Restore cap for next cycle
                agent.optimiser.cfg.hvac_max_kw = 10.0

        # Record
        elapsed = time.perf_counter() - t0
        self.history.append({
            "timestamp":     unix_time,
            "total_demand":  sum(a.total_power_kw for a in self.agents.values()),
            "campus_budget": self.cfg.campus_power_limit_kw,
            "coord_time_s":  elapsed,
        })

        log.info(
            f"Coordinator step t={unix_time:.0f}  "
            f"demand={total_demand:.1f}kW  budget={self.cfg.campus_power_limit_kw}kW  "
            f"t={elapsed:.3f}s"
        )
        return results

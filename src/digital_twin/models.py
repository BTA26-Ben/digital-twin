"""
models.py
---------
Pydantic / dataclass definitions for every data type exchanged between modules.
All classes are intentionally plain-Python so they can be serialised to JSON,
stored in SQLite, and passed between processes without coupling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    OCCUPANCY   = "occupancy"
    POWER       = "power"
    HUMIDITY    = "humidity"
    CO2         = "co2"


class DeviceType(str, Enum):
    HVAC     = "hvac"
    LIGHTING = "lighting"
    EV_CHARGER = "ev_charger"
    BATTERY  = "battery"


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@dataclass
class TelemetryPoint:
    """A single sensor reading from a field device.

    Attributes
    ----------
    timestamp   : Unix epoch seconds (float for sub-second precision).
    sensor_id   : Globally unique sensor identifier string.
    sensor_type : One of SensorType enum values.
    value       : Numeric measurement.
    unit        : Physical unit string (e.g. "°C", "W", "persons").
    quality     : Optional quality flag (0–1); 1 = fully reliable.
    agent_id    : Building/zone agent that owns this sensor.
    """
    timestamp:   float
    sensor_id:   str
    sensor_type: SensorType
    value:       float
    unit:        str
    quality:     float = 1.0
    agent_id:    str   = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":   self.timestamp,
            "sensor_id":   self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "value":       self.value,
            "unit":        self.unit,
            "quality":     self.quality,
            "agent_id":    self.agent_id,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TelemetryPoint":
        return TelemetryPoint(
            timestamp=d["timestamp"],
            sensor_id=d["sensor_id"],
            sensor_type=SensorType(d["sensor_type"]),
            value=d["value"],
            unit=d["unit"],
            quality=d.get("quality", 1.0),
            agent_id=d.get("agent_id", "unknown"),
        )


# ---------------------------------------------------------------------------
# Device state
# ---------------------------------------------------------------------------

@dataclass
class DeviceState:
    """Current setpoint and power draw of a single controllable device.

    Attributes
    ----------
    device_id   : Unique device identifier.
    device_type : DeviceType enum.
    setpoint    : Current setpoint (temperature °C for HVAC, kW for others).
    power_kw    : Current power consumption in kW.
    is_on       : Binary on/off state.
    min_power   : Minimum allowable power in kW.
    max_power   : Maximum allowable power in kW.
    ramp_rate   : Maximum change in kW per timestep.
    """
    device_id:   str
    device_type: DeviceType
    setpoint:    float
    power_kw:    float
    is_on:       bool
    min_power:   float = 0.0
    max_power:   float = 10.0
    ramp_rate:   float = 2.0          # kW per control interval

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id":   self.device_id,
            "device_type": self.device_type.value,
            "setpoint":    self.setpoint,
            "power_kw":    self.power_kw,
            "is_on":       self.is_on,
            "min_power":   self.min_power,
            "max_power":   self.max_power,
            "ramp_rate":   self.ramp_rate,
        }


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass
class ComfortConstraints:
    """Thermal comfort bounds for a zone."""
    temp_min_occupied:   float = 20.0   # °C
    temp_max_occupied:   float = 24.0   # °C
    temp_min_unoccupied: float = 16.0   # °C
    temp_max_unoccupied: float = 28.0   # °C
    co2_max_ppm:         float = 1000.0


@dataclass
class AgentState:
    """Full state snapshot of a building agent at one timestep.

    Attributes
    ----------
    agent_id      : Unique identifier for the building/zone agent.
    timestamp     : Unix epoch of this snapshot.
    zone_temps    : Mapping zone_id → temperature °C.
    occupancy     : Mapping zone_id → estimated occupant count.
    device_states : Mapping device_id → DeviceState.
    constraints   : Comfort and capacity constraints.
    grid_tariff   : Current electricity price (£/kWh).
    carbon_factor : Current carbon intensity (gCO2/kWh).
    """
    agent_id:      str
    timestamp:     float
    zone_temps:    Dict[str, float]
    occupancy:     Dict[str, float]
    device_states: Dict[str, DeviceState]
    constraints:   ComfortConstraints
    grid_tariff:   float = 0.28          # £/kWh
    carbon_factor: float = 200.0         # gCO2/kWh

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":      self.agent_id,
            "timestamp":     self.timestamp,
            "zone_temps":    self.zone_temps,
            "occupancy":     self.occupancy,
            "device_states": {k: v.to_dict() for k, v in self.device_states.items()},
            "constraints": {
                "temp_min_occupied":   self.constraints.temp_min_occupied,
                "temp_max_occupied":   self.constraints.temp_max_occupied,
                "temp_min_unoccupied": self.constraints.temp_min_unoccupied,
                "temp_max_unoccupied": self.constraints.temp_max_unoccupied,
                "co2_max_ppm":         self.constraints.co2_max_ppm,
            },
            "grid_tariff":   self.grid_tariff,
            "carbon_factor": self.carbon_factor,
        }


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

@dataclass
class Forecast:
    """Short-term prediction output from the ForecastEngine.

    Attributes
    ----------
    agent_id         : Which agent/zone this forecast is for.
    variable         : What is being forecast ("occupancy", "outdoor_temp" …).
    horizon          : Number of timesteps ahead.
    timestamps       : List of future Unix epoch times.
    predicted_values : Point forecast values.
    lower_bound      : Lower 90% confidence interval values.
    upper_bound      : Upper 90% confidence interval values.
    model_name       : Which model produced this forecast.
    created_at       : When this forecast was generated.
    """
    agent_id:         str
    variable:         str
    horizon:          int
    timestamps:       List[float]
    predicted_values: List[float]
    lower_bound:      List[float]
    upper_bound:      List[float]
    model_name:       str   = "baseline"
    created_at:       float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":         self.agent_id,
            "variable":         self.variable,
            "horizon":          self.horizon,
            "timestamps":       self.timestamps,
            "predicted_values": self.predicted_values,
            "lower_bound":      self.lower_bound,
            "upper_bound":      self.upper_bound,
            "model_name":       self.model_name,
            "created_at":       self.created_at,
        }


# ---------------------------------------------------------------------------
# Optimisation problem
# ---------------------------------------------------------------------------

@dataclass
class OptimisationProblem:
    """Parameterisation of one MPC solve.

    Attributes
    ----------
    agent_id        : Agent this problem belongs to.
    horizon         : Control horizon in timesteps.
    dt_seconds      : Duration of each timestep in seconds.
    decision_vars   : Names of decision variables (e.g. ['hvac_power', 'lighting']).
    objective_terms : Dict of term_name → weight (e.g. {'energy_cost': 1.0}).
    constraints     : List of constraint description dicts.
    initial_state   : Current state used as initial condition.
    forecasts       : Forecasts relevant to this solve.
    solver          : Solver name to use ('OSQP', 'ECOS', 'SCS').
    warm_start      : Whether to warm-start from previous solution.
    """
    agent_id:        str
    horizon:         int
    dt_seconds:      float
    decision_vars:   List[str]
    objective_terms: Dict[str, float]
    constraints:     List[Dict[str, Any]]
    initial_state:   AgentState
    forecasts:       List[Forecast]
    solver:          str  = "OSQP"
    warm_start:      bool = True


# ---------------------------------------------------------------------------
# Optimisation result
# ---------------------------------------------------------------------------

@dataclass
class OptimisationResult:
    """Output of one MPC solve.

    Attributes
    ----------
    agent_id       : Agent this result belongs to.
    timestamp      : When the solve was requested.
    status         : Solver status string (e.g. 'optimal', 'infeasible').
    solve_time_s   : Wall-clock solve time in seconds.
    objective_value: Optimal objective value (if feasible).
    schedule       : Dict device_id → list of power setpoints over horizon.
    setpoints_now  : The immediate (t=0) setpoint for each device to apply.
    """
    agent_id:        str
    timestamp:       float
    status:          str
    solve_time_s:    float
    objective_value: Optional[float]
    schedule:        Dict[str, List[float]]
    setpoints_now:   Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":        self.agent_id,
            "timestamp":       self.timestamp,
            "status":          self.status,
            "solve_time_s":    self.solve_time_s,
            "objective_value": self.objective_value,
            "schedule":        self.schedule,
            "setpoints_now":   self.setpoints_now,
        }


# ---------------------------------------------------------------------------
# Experiment record
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """Metadata and summary metrics for one simulation run."""
    experiment_id:      str
    git_commit:         str
    seed:               int
    config:             Dict[str, Any]
    start_time:         float
    end_time:           float
    total_energy_kwh:   float
    total_cost_gbp:     float
    comfort_violations: float   # degree-minutes above/below bounds
    avg_solve_time_s:   float
    n_infeasible:       int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

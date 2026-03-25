"""
sensor_simulator.py
-------------------
Generates realistic synthetic sensor telemetry with configurable Gaussian noise
and random packet-dropout.  Used in place of real hardware during simulation
and testing.

Public API
----------
SensorConfig       – per-sensor configuration dataclass
SensorSimulator    – produces TelemetryPoint streams
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np

from .models import SensorType, TelemetryPoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """Configuration for a single simulated sensor.

    Parameters
    ----------
    sensor_id    : Unique identifier.
    sensor_type  : Physical quantity being measured.
    unit         : Physical unit string.
    noise_std    : Standard deviation of additive Gaussian noise.
    dropout_prob : Probability [0, 1] that a reading is silently dropped.
    bias         : Constant additive bias (models systematic error).
    agent_id     : Owning agent.
    """
    sensor_id:    str
    sensor_type:  SensorType
    unit:         str
    noise_std:    float = 0.5
    dropout_prob: float = 0.05
    bias:         float = 0.0
    agent_id:     str   = "agent_0"


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class SensorSimulator:
    """Wraps a set of sensors and emits TelemetryPoint readings.

    Parameters
    ----------
    sensors   : List of SensorConfig objects.
    seed      : RNG seed for reproducibility.

    Usage
    -----
    >>> sim = SensorSimulator(sensors=[cfg], seed=42)
    >>> readings = sim.read_all(true_values={"temp_01": 21.5})
    """

    def __init__(self, sensors: List[SensorConfig], seed: int = 42) -> None:
        self._sensors: Dict[str, SensorConfig] = {s.sensor_id: s for s in sensors}
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read_all(
        self,
        true_values: Dict[str, float],
        timestamp: Optional[float] = None,
    ) -> List[TelemetryPoint]:
        """Return noisy readings for all sensors given ground-truth values.

        Parameters
        ----------
        true_values : Mapping sensor_id → true physical value.
        timestamp   : Unix epoch; defaults to time.time().

        Returns
        -------
        List of TelemetryPoint (dropped sensors omitted from list).
        """
        ts = timestamp if timestamp is not None else time.time()
        readings: List[TelemetryPoint] = []

        for sid, cfg in self._sensors.items():
            # Apply dropout
            if self._py_rng.random() < cfg.dropout_prob:
                continue  # packet lost

            true_val = true_values.get(sid)
            if true_val is None:
                continue  # sensor has no ground truth this step

            noisy_val = float(
                true_val
                + cfg.bias
                + self._rng.normal(0.0, cfg.noise_std)
            )

            # Quality degrades when noise is large relative to value range
            quality = float(np.clip(1.0 - cfg.noise_std / (abs(true_val) + 1e-6), 0.0, 1.0))

            readings.append(
                TelemetryPoint(
                    timestamp=ts,
                    sensor_id=sid,
                    sensor_type=cfg.sensor_type,
                    value=noisy_val,
                    unit=cfg.unit,
                    quality=quality,
                    agent_id=cfg.agent_id,
                )
            )
        return readings

    def read_sensor(
        self,
        sensor_id: str,
        true_value: float,
        timestamp: Optional[float] = None,
    ) -> Optional[TelemetryPoint]:
        """Read a single sensor.  Returns None if the packet is dropped."""
        results = self.read_all(
            true_values={sensor_id: true_value},
            timestamp=timestamp,
        )
        return results[0] if results else None

    def stream(
        self,
        true_value_sequence: List[Dict[str, float]],
        start_time: float = 0.0,
        dt: float = 300.0,
    ) -> Iterator[List[TelemetryPoint]]:
        """Yield batches of readings for each timestep in the sequence.

        Parameters
        ----------
        true_value_sequence : List of {sensor_id: value} dicts per timestep.
        start_time          : Unix epoch of first timestep.
        dt                  : Seconds between timesteps.
        """
        for step, true_values in enumerate(true_value_sequence):
            ts = start_time + step * dt
            yield self.read_all(true_values=true_values, timestamp=ts)

    def add_sensor(self, cfg: SensorConfig) -> None:
        """Dynamically register a new sensor (supports extensibility)."""
        self._sensors[cfg.sensor_id] = cfg

    def remove_sensor(self, sensor_id: str) -> None:
        """Deregister a sensor by ID."""
        self._sensors.pop(sensor_id, None)

    @property
    def sensor_ids(self) -> List[str]:
        """Return list of registered sensor IDs."""
        return list(self._sensors.keys())


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def default_building_sensors(agent_id: str = "agent_0", seed: int = 42) -> List[SensorConfig]:
    """Return a representative set of sensors for a single-zone building."""
    return [
        SensorConfig("temp_zone_a",    SensorType.TEMPERATURE, "°C",      0.3,  0.05, agent_id=agent_id),
        SensorConfig("temp_zone_b",    SensorType.TEMPERATURE, "°C",      0.3,  0.05, agent_id=agent_id),
        SensorConfig("temp_outdoor",   SensorType.TEMPERATURE, "°C",      0.5,  0.02, agent_id=agent_id),
        SensorConfig("occ_zone_a",     SensorType.OCCUPANCY,   "persons", 0.5,  0.10, agent_id=agent_id),
        SensorConfig("occ_zone_b",     SensorType.OCCUPANCY,   "persons", 0.5,  0.10, agent_id=agent_id),
        SensorConfig("power_hvac",     SensorType.POWER,       "kW",      0.2,  0.02, agent_id=agent_id),
        SensorConfig("power_total",    SensorType.POWER,       "kW",      0.3,  0.02, agent_id=agent_id),
        SensorConfig("co2_zone_a",     SensorType.CO2,         "ppm",     10.0, 0.05, agent_id=agent_id),
    ]

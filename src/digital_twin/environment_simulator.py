"""
environment_simulator.py
------------------------
Physics-based thermal model (RC-network) and stochastic occupancy model.

The thermal model is a first-order resistance-capacitance (1R-1C) lumped
model per zone, extended to two zones with an inter-zone coupling term.

Mathematical model (discrete-time)
-----------------------------------
    T_z[t+1] = T_z[t]
              + (dt/C_z) * (
                    UA_env * (T_out[t] - T_z[t])
                  + UA_adj * (T_adj[t] - T_z[t])   # for zone B ↔ A
                  + Q_hvac[t]
                  + Q_solar[t]
                  + Q_occ[t]
                )

where
    C_z      thermal capacitance [J/K]
    UA_env   overall heat loss coefficient to outdoors [W/K]
    UA_adj   inter-zone coupling coefficient [W/K]
    Q_hvac   HVAC heat injection (positive = heating, negative = cooling) [W]
    Q_solar  solar irradiance gain [W]
    Q_occ    internal gain from occupants (~80 W/person)

Public API
----------
BuildingConfig      – physical parameters and initial conditions
OccupancyModel      – probabilistic occupancy profile
EnvironmentSimulator – advances the building state one timestep
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ZoneConfig:
    """Physical parameters for one thermal zone."""
    zone_id:           str
    capacitance_j_k:   float = 5e6      # J/K  (≈ medium office floor)
    ua_envelope:       float = 200.0    # W/K  (heat loss to outdoors)
    ua_adjacent:       float = 50.0     # W/K  (to neighbouring zone)
    init_temp:         float = 20.0     # °C
    area_m2:           float = 100.0    # for solar gain calculation
    glazing_fraction:  float = 0.3
    shgc:              float = 0.4      # solar heat gain coefficient


@dataclass
class BuildingConfig:
    """Configuration for a two-zone building."""
    zones:             List[ZoneConfig] = field(default_factory=list)
    dt_seconds:        float = 300.0    # 5-minute timestep
    # Outdoor conditions will be provided externally
    latitude_deg:      float = 51.5     # London latitude


# ---------------------------------------------------------------------------
# Solar irradiance helper (simplified sinusoidal model)
# ---------------------------------------------------------------------------

def solar_irradiance_w_m2(hour_of_day: float, day_of_year: int = 180) -> float:
    """Return approximate global horizontal irradiance in W/m².

    Uses a simple sinusoidal model suitable for clear-sky conditions.
    """
    declination = math.radians(23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81))))
    lat = math.radians(51.5)
    hour_angle = math.radians(15 * (hour_of_day - 12))
    cos_z = (
        math.sin(lat) * math.sin(declination)
        + math.cos(lat) * math.cos(declination) * math.cos(hour_angle)
    )
    ghi = max(0.0, 900.0 * cos_z)  # peak 900 W/m²
    return ghi


# ---------------------------------------------------------------------------
# Occupancy model
# ---------------------------------------------------------------------------

class OccupancyModel:
    """Stochastic occupancy schedule based on a probability profile.

    The profile is a list of 24 hourly probabilities of someone being present.
    At each timestep, occupant count is drawn from a Binomial distribution.

    Parameters
    ----------
    max_occupants : Maximum number of people in the zone.
    profile_24h   : Length-24 list of occupancy probabilities (0–1) per hour.
    seed          : RNG seed.
    """

    DEFAULT_OFFICE_PROFILE = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # 00–05
        0.0, 0.1, 0.5, 0.9, 0.9, 0.9,   # 06–11
        0.8, 0.9, 0.9, 0.9, 0.8, 0.5,   # 12–17
        0.2, 0.1, 0.0, 0.0, 0.0, 0.0,   # 18–23
    ]

    def __init__(
        self,
        max_occupants: int = 20,
        profile_24h: Optional[List[float]] = None,
        seed: int = 42,
    ) -> None:
        self.max_occupants = max_occupants
        self.profile = profile_24h or self.DEFAULT_OFFICE_PROFILE
        self._rng = np.random.default_rng(seed)

    def occupants_at(self, unix_time: float) -> int:
        """Return stochastic occupant count at the given Unix timestamp."""
        hour = (unix_time % 86400) / 3600
        hour_int = int(hour) % 24
        prob = self.profile[hour_int]
        return int(self._rng.binomial(self.max_occupants, prob))


# ---------------------------------------------------------------------------
# Environment simulator
# ---------------------------------------------------------------------------

class EnvironmentSimulator:
    """Advances the building thermal model one timestep at a time.

    Parameters
    ----------
    config   : BuildingConfig defining zones and timestep size.
    seed     : RNG seed for occupancy model.

    Usage
    -----
    >>> sim = EnvironmentSimulator(config)
    >>> state = sim.step(hvac_power_kw={"zone_a": 5.0}, outdoor_temp=10.0,
    ...                   unix_time=start_time)
    """

    HEAT_PER_PERSON_W = 80.0  # sensible heat gain per occupant [W]

    def __init__(self, config: Optional[BuildingConfig] = None, seed: int = 42) -> None:
        self.config = config or self._default_config()
        self._zone_map: Dict[str, ZoneConfig] = {z.zone_id: z for z in self.config.zones}
        # Initialise zone temperatures
        self._temps: Dict[str, float] = {z.zone_id: z.init_temp for z in self.config.zones}
        # Per-zone occupancy models
        self._occ_models: Dict[str, OccupancyModel] = {
            z.zone_id: OccupancyModel(seed=seed + i)
            for i, z in enumerate(self.config.zones)
        }
        self._step_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        hvac_power_kw: Dict[str, float],
        outdoor_temp: float,
        unix_time: float,
        day_of_year: int = 180,
    ) -> Dict[str, Dict]:
        """Advance the simulator by one timestep.

        Parameters
        ----------
        hvac_power_kw : Mapping zone_id → HVAC power in kW
                        (positive = heating, negative = cooling).
        outdoor_temp  : Outdoor dry-bulb temperature [°C].
        unix_time     : Current Unix timestamp.
        day_of_year   : Day of year for solar gain calculation.

        Returns
        -------
        Dict with keys "temps", "occupancy", "solar_irr", "outdoor_temp".
        """
        dt = self.config.dt_seconds
        zone_ids = [z.zone_id for z in self.config.zones]
        n_zones = len(zone_ids)

        # Compute solar irradiance
        hour = (unix_time % 86400) / 3600
        ghi = solar_irradiance_w_m2(hour, day_of_year)

        # Compute occupancy
        occupancy: Dict[str, int] = {
            zid: self._occ_models[zid].occupants_at(unix_time)
            for zid in zone_ids
        }

        # Update temperatures using forward Euler
        new_temps: Dict[str, float] = {}
        for i, zid in enumerate(zone_ids):
            z = self._zone_map[zid]
            T = self._temps[zid]
            Q_hvac = hvac_power_kw.get(zid, 0.0) * 1000.0  # kW → W

            # Solar gain
            Q_solar = ghi * z.area_m2 * z.glazing_fraction * z.shgc

            # Internal gains (occupants)
            Q_occ = occupancy[zid] * self.HEAT_PER_PERSON_W

            # Envelope loss
            Q_env = z.ua_envelope * (outdoor_temp - T)

            # Inter-zone coupling (other zones)
            Q_adj = 0.0
            for j, other_zid in enumerate(zone_ids):
                if i != j:
                    Q_adj += z.ua_adjacent * (self._temps[other_zid] - T)

            dT = (dt / z.capacitance_j_k) * (Q_env + Q_adj + Q_hvac + Q_solar + Q_occ)
            new_temps[zid] = T + dT

        self._temps = new_temps
        self._step_count += 1

        return {
            "temps":       dict(self._temps),
            "occupancy":   occupancy,
            "solar_irr":   ghi,
            "outdoor_temp": outdoor_temp,
        }

    def reset(self) -> None:
        """Reset zone temperatures to initial conditions."""
        for z in self.config.zones:
            self._temps[z.zone_id] = z.init_temp
        self._step_count = 0

    def get_temps(self) -> Dict[str, float]:
        """Return current zone temperatures."""
        return dict(self._temps)

    def set_temps(self, temps: Dict[str, float]) -> None:
        """Manually override zone temperatures (e.g. for warm-start)."""
        self._temps.update(temps)

    # ------------------------------------------------------------------
    # Outdoor temperature profile (simple sinusoidal)
    # ------------------------------------------------------------------

    @staticmethod
    def outdoor_temp_at(unix_time: float, mean: float = 10.0, amplitude: float = 5.0) -> float:
        """Sinusoidal outdoor temperature profile (°C).

        Daily cycle with minimum at 04:00 and maximum at 14:00.
        """
        hour = (unix_time % 86400) / 3600
        # Phase: minimum at 04:00 → offset = 4 hours past midnight
        radians = 2 * math.pi * (hour - 4) / 24
        return mean + amplitude * math.sin(radians - math.pi / 2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_config() -> BuildingConfig:
        return BuildingConfig(
            zones=[
                ZoneConfig("zone_a", init_temp=21.0),
                ZoneConfig("zone_b", init_temp=21.5),
            ]
        )

"""
state_estimator.py
------------------
Linear Kalman Filter for fusing noisy / missing sensor readings with a
process model to produce a best-estimate of building state.

For the thermal model the state vector is:
    x = [T_zone_a, T_zone_b]  (zone temperatures in °C)

The process model is the linearised 1R-1C thermal model (same as
EnvironmentSimulator) — so the estimator and simulator share the same
physics, which is good practice.

Public API
----------
KalmanConfig       – filter tuning parameters
KalmanStateEstimator – update(measurements) → estimated state
ParticleFilterEstimator – non-linear alternative (for reference)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import SensorType, TelemetryPoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KalmanConfig:
    """Tuning parameters for the Kalman filter.

    Attributes
    ----------
    state_dim  : Dimension of the state vector.
    obs_dim    : Dimension of the observation vector.
    Q          : Process noise covariance matrix (state_dim × state_dim).
    R          : Measurement noise covariance matrix (obs_dim × obs_dim).
    x0         : Initial state vector.
    P0         : Initial estimate covariance.
    """
    state_dim: int
    obs_dim:   int
    Q:         np.ndarray                    # process noise covariance
    R:         np.ndarray                    # measurement noise covariance
    x0:        np.ndarray                    # initial state
    P0:        np.ndarray                    # initial covariance


# ---------------------------------------------------------------------------
# Kalman Filter
# ---------------------------------------------------------------------------

class KalmanStateEstimator:
    """Discrete-time linear Kalman filter for zone temperature estimation.

    Parameters
    ----------
    config     : KalmanConfig with initial parameters.
    zone_ids   : Ordered list of zone identifiers matching state vector.

    The filter maintains state x and covariance P and exposes two methods:
        predict(F, B, u) – time-update step
        update(z, H)     – measurement-update step

    Usage
    -----
    >>> est = KalmanStateEstimator.for_two_zones()
    >>> est.predict(F=F_matrix, B=B_matrix, u=control_input)
    >>> x_hat = est.update(measurements=readings)
    """

    def __init__(self, config: KalmanConfig, zone_ids: List[str]) -> None:
        self.cfg       = config
        self.zone_ids  = zone_ids
        self.x: np.ndarray = config.x0.copy().astype(float)  # state estimate
        self.P: np.ndarray = config.P0.copy().astype(float)  # covariance
        self._n        = config.state_dim

    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------

    def predict(
        self,
        F: np.ndarray,
        B: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Time-update (prediction) step.

        x̂⁻ = F x̂ + B u
        P⁻  = F P Fᵀ + Q

        Parameters
        ----------
        F  : State transition matrix (n×n).
        B  : Control input matrix (n×m).  None if no control input.
        u  : Control input vector (m,).

        Returns
        -------
        Predicted state estimate x̂⁻.
        """
        # Project state
        self.x = F @ self.x
        if B is not None and u is not None:
            self.x = self.x + B @ u

        # Project covariance
        self.P = F @ self.P @ F.T + self.cfg.Q
        return self.x.copy()

    def update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Measurement-update step.

        K  = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
        x̂  = x̂⁻ + K (z − H x̂⁻)
        P  = (I − K H) P⁻

        Parameters
        ----------
        z          : Observation vector.
        H          : Observation matrix (m×n).
        R_override : Optional per-call measurement noise covariance.

        Returns
        -------
        Updated state estimate x̂.
        """
        R = R_override if R_override is not None else self.cfg.R
        S = H @ self.P @ H.T + R                     # innovation covariance
        try:
            K = self.P @ H.T @ np.linalg.inv(S)      # Kalman gain
        except np.linalg.LinAlgError:
            log.warning("Singular innovation covariance; skipping update.")
            return self.x.copy()

        innovation = z - H @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(self._n) - K @ H) @ self.P
        return self.x.copy()

    # ------------------------------------------------------------------
    # High-level convenience method
    # ------------------------------------------------------------------

    def fuse_telemetry(
        self,
        readings: List[TelemetryPoint],
        F: np.ndarray,
        B: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        dt: float = 300.0,
    ) -> Dict[str, float]:
        """Run one full predict→update cycle from raw TelemetryPoints.

        Only temperature readings are used for the state estimate in this
        implementation.  Occupancy and power readings are passed through
        without filtering (returned separately).

        Parameters
        ----------
        readings : All sensor readings from one timestep.
        F        : State transition matrix for the predict step.
        B, u     : Optional control input.
        dt       : Timestep duration (s) — used to build F dynamically if
                   caller passes None; ignored when F is explicit.

        Returns
        -------
        Dict zone_id → estimated temperature (°C).
        """
        # ---- Predict ----
        self.predict(F=F, B=B, u=u)

        # ---- Build observation from temperature readings ----
        temp_readings: Dict[str, float] = {}
        for r in readings:
            if r.sensor_type == SensorType.TEMPERATURE:
                # Map sensor to zone by checking if zone_id is in sensor_id
                for i, zid in enumerate(self.zone_ids):
                    if zid in r.sensor_id or r.sensor_id.endswith(f"_{chr(ord('a') + i)}"):
                        temp_readings[zid] = r.value

        if not temp_readings:
            # No temperature observations — return prediction only
            return {zid: float(self.x[i]) for i, zid in enumerate(self.zone_ids)}

        # Build observation vector and H matrix from available readings
        obs_indices = []
        obs_values  = []
        for i, zid in enumerate(self.zone_ids):
            if zid in temp_readings:
                obs_indices.append(i)
                obs_values.append(temp_readings[zid])

        m = len(obs_indices)
        H = np.zeros((m, self._n))
        for row, col in enumerate(obs_indices):
            H[row, col] = 1.0
        z = np.array(obs_values)
        R = np.eye(m) * self.cfg.R[0, 0]  # assume uniform sensor noise

        # ---- Update ----
        self.update(z=z, H=H, R_override=R)

        return {zid: float(self.x[i]) for i, zid in enumerate(self.zone_ids)}

    def get_uncertainty(self) -> Dict[str, float]:
        """Return 1-sigma uncertainty (√P_ii) for each state variable."""
        return {zid: float(np.sqrt(self.P[i, i])) for i, zid in enumerate(self.zone_ids)}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_two_zones(
        cls,
        init_temps: Tuple[float, float] = (21.0, 21.5),
        process_noise: float = 0.1,
        obs_noise: float = 0.5,
    ) -> "KalmanStateEstimator":
        """Construct a filter pre-configured for a two-zone building."""
        n = 2
        config = KalmanConfig(
            state_dim = n,
            obs_dim   = n,
            Q  = np.eye(n) * process_noise**2,
            R  = np.eye(n) * obs_noise**2,
            x0 = np.array(list(init_temps)),
            P0 = np.eye(n) * 1.0,
        )
        return cls(config=config, zone_ids=["zone_a", "zone_b"])

    @staticmethod
    def build_transition_matrix(
        dt: float,
        capacitance: float,
        ua_envelope: float,
        ua_adjacent: float,
        n_zones: int = 2,
    ) -> np.ndarray:
        """Build the state-transition matrix F for the RC thermal model.

        F = I + dt/C * (-UA_total * I + UA_adj * (A_adj))

        where A_adj is the zone-coupling adjacency (off-diagonal terms).
        """
        alpha = dt / capacitance  # [s / (J/K)] = K/W·s · (1/s) ??? check units
        # Diagonal: self-cooling
        F = np.eye(n_zones)
        for i in range(n_zones):
            F[i, i] = 1.0 - alpha * (ua_envelope + (n_zones - 1) * ua_adjacent)
        # Off-diagonal: inter-zone coupling
        for i in range(n_zones):
            for j in range(n_zones):
                if i != j:
                    F[i, j] = alpha * ua_adjacent
        return F


# ---------------------------------------------------------------------------
# Simple Particle Filter (non-linear state estimator — for extensibility)
# ---------------------------------------------------------------------------

class ParticleFilterEstimator:
    """Sequential importance resampling (SIR) particle filter.

    Suitable when the process or observation model is non-linear.
    This is a reference implementation; for the MVP the Kalman filter is
    preferred.

    Parameters
    ----------
    n_particles : Number of particles.
    state_dim   : Dimension of the state.
    process_std : Standard deviation of the process noise.
    obs_std     : Standard deviation of the observation noise.
    seed        : RNG seed.
    """

    def __init__(
        self,
        n_particles: int = 500,
        state_dim: int = 2,
        process_std: float = 0.2,
        obs_std: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.N = n_particles
        self.d = state_dim
        self._rng = np.random.default_rng(seed)
        self.process_std = process_std
        self.obs_std = obs_std
        # Initialise particles uniformly around 21°C
        self.particles = self._rng.normal(21.0, 1.0, (self.N, self.d))
        self.weights = np.ones(self.N) / self.N

    def predict(self, F: np.ndarray, u: Optional[np.ndarray] = None) -> None:
        """Propagate particles through the process model + noise."""
        noise = self._rng.normal(0, self.process_std, self.particles.shape)
        if u is not None:
            self.particles = (F @ self.particles.T).T + noise + u
        else:
            self.particles = (F @ self.particles.T).T + noise

    def update(self, z: np.ndarray, H: np.ndarray) -> None:
        """Weight particles by likelihood of observation."""
        predicted_obs = (H @ self.particles.T).T     # (N, m)
        diff = predicted_obs - z                      # (N, m)
        # Gaussian likelihood
        log_w = -0.5 * np.sum(diff**2, axis=1) / self.obs_std**2
        log_w -= log_w.max()                          # numerical stability
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()
        # Resample
        self._resample()

    def _resample(self) -> None:
        """Systematic resampling."""
        cumsum = np.cumsum(self.weights)
        positions = (self._rng.random() + np.arange(self.N)) / self.N
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    def estimate(self) -> np.ndarray:
        """Return weighted mean state estimate."""
        return np.average(self.particles, weights=self.weights, axis=0)

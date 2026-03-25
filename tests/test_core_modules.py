"""
test_environment_simulator.py / test_state_estimator.py /
test_forecast_engine.py / test_optimiser.py
-------------------------------------------
Unit tests for core simulation and control modules.
"""

from __future__ import annotations

import math
import time
import pytest
import numpy as np

from digital_twin.environment_simulator import (
    BuildingConfig,
    EnvironmentSimulator,
    OccupancyModel,
    ZoneConfig,
    solar_irradiance_w_m2,
)
from digital_twin.state_estimator import (
    KalmanStateEstimator,
    ParticleFilterEstimator,
)
from digital_twin.forecast_engine import (
    ExponentialSmoothingForecaster,
    ForecastEngine,
)
from digital_twin.models import SensorType, TelemetryPoint


# ============================================================
# Environment Simulator Tests
# ============================================================

class TestSolarIrradiance:
    """Solar model sanity checks."""

    def test_zero_at_night(self):
        assert solar_irradiance_w_m2(0.0) == 0.0   # midnight

    def test_positive_at_noon(self):
        assert solar_irradiance_w_m2(12.0) > 0.0

    def test_symmetric_around_noon(self):
        morning = solar_irradiance_w_m2(10.0)
        afternoon = solar_irradiance_w_m2(14.0)
        assert abs(morning - afternoon) < 50  # rough symmetry


class TestOccupancyModel:
    """Occupancy model probabilistic behaviour."""

    def test_zero_at_midnight(self):
        model = OccupancyModel(max_occupants=20, seed=42)
        # 00:00 = unix 0 → hour 0
        count = model.occupants_at(0.0)
        assert count == 0

    def test_nonzero_during_business_hours(self):
        model = OccupancyModel(max_occupants=20, seed=42)
        # 10:00 = unix 36000
        counts = [model.occupants_at(36000.0) for _ in range(50)]
        assert max(counts) > 0

    def test_count_bounded(self):
        model = OccupancyModel(max_occupants=10, seed=1)
        for t in range(0, 86400, 300):
            assert 0 <= model.occupants_at(float(t)) <= 10


class TestEnvironmentSimulator:
    """Thermal model correctness."""

    @pytest.fixture
    def sim(self):
        return EnvironmentSimulator(seed=42)

    def test_step_returns_expected_keys(self, sim):
        out = sim.step(hvac_power_kw={"zone_a": 0.0, "zone_b": 0.0},
                       outdoor_temp=10.0, unix_time=0.0)
        assert "temps" in out
        assert "occupancy" in out
        assert "solar_irr" in out
        assert "outdoor_temp" in out

    def test_heating_raises_temperature(self, sim):
        t0 = sim.get_temps().copy()
        for _ in range(10):
            sim.step(hvac_power_kw={"zone_a": 10.0, "zone_b": 10.0},
                     outdoor_temp=5.0, unix_time=0.0)
        t1 = sim.get_temps()
        assert t1["zone_a"] > t0["zone_a"]

    def test_no_heating_cold_outdoor_cools_room(self, sim):
        sim.set_temps({"zone_a": 25.0, "zone_b": 25.0})
        for i in range(50):
            sim.step(hvac_power_kw={"zone_a": 0.0, "zone_b": 0.0},
                     outdoor_temp=-10.0, unix_time=float(i * 300))
        temps = sim.get_temps()
        assert temps["zone_a"] < 25.0

    def test_reset_returns_initial_temps(self):
        cfg = BuildingConfig(zones=[ZoneConfig("zone_a", init_temp=21.0),
                                    ZoneConfig("zone_b", init_temp=22.0)])
        sim = EnvironmentSimulator(config=cfg, seed=0)
        sim.step({"zone_a": 5.0, "zone_b": 5.0}, outdoor_temp=0.0, unix_time=0.0)
        sim.reset()
        assert abs(sim.get_temps()["zone_a"] - 21.0) < 0.01

    def test_outdoor_temp_profile_daily_cycle(self):
        """Min should be at ~04:00 and max at ~14:00."""
        temps = [EnvironmentSimulator.outdoor_temp_at(float(h * 3600)) for h in range(24)]
        assert temps.index(min(temps)) in [3, 4, 5]
        assert temps.index(max(temps)) in [13, 14, 15]


# ============================================================
# State Estimator Tests
# ============================================================

class TestKalmanStateEstimator:
    """Kalman filter correctness."""

    @pytest.fixture
    def estimator(self):
        return KalmanStateEstimator.for_two_zones(
            init_temps=(21.0, 21.5),
            process_noise=0.1,
            obs_noise=0.5,
        )

    @pytest.fixture
    def F(self):
        return KalmanStateEstimator.build_transition_matrix(
            dt=300.0, capacitance=5e6, ua_envelope=200.0, ua_adjacent=50.0
        )

    def test_predict_does_not_explode(self, estimator, F):
        """After predict, state should remain bounded."""
        estimator.predict(F=F)
        assert all(10 < v < 35 for v in estimator.x)

    def test_update_moves_toward_observation(self, estimator):
        """After update with far observation, state should shift toward obs."""
        H = np.eye(2)
        z = np.array([25.0, 25.0])
        x_before = estimator.x.copy()
        estimator.update(z=z, H=H)
        # State should move toward z=25
        for i in range(2):
            assert estimator.x[i] > x_before[i]

    def test_fuse_telemetry_returns_zone_dict(self, estimator, F):
        readings = [
            TelemetryPoint(0.0, "temp_zone_a", SensorType.TEMPERATURE, 21.5, "°C", agent_id="a"),
            TelemetryPoint(0.0, "temp_zone_b", SensorType.TEMPERATURE, 22.0, "°C", agent_id="a"),
        ]
        result = estimator.fuse_telemetry(readings, F=F)
        assert set(result.keys()) == {"zone_a", "zone_b"}
        assert 19.0 < result["zone_a"] < 23.0

    def test_uncertainty_positive(self, estimator):
        unc = estimator.get_uncertainty()
        assert all(v > 0 for v in unc.values())

    def test_fuse_empty_telemetry_returns_prediction(self, estimator, F):
        """No observations → just return the predicted state."""
        x_pred = estimator.predict(F=F)
        result = estimator.fuse_telemetry(readings=[], F=F)
        # Should still return all zone keys
        assert len(result) == 2

    def test_transition_matrix_diagonal_less_than_one(self, F):
        """Diagonal entries of F must be < 1 (stable system)."""
        assert all(F[i, i] < 1.0 for i in range(F.shape[0]))


class TestParticleFilter:
    """Particle filter sanity."""

    def test_estimate_bounded(self):
        pf = ParticleFilterEstimator(n_particles=100, state_dim=2, seed=42)
        est = pf.estimate()
        assert est.shape == (2,)
        assert all(10 < v < 35 for v in est)

    def test_update_weights_sum_to_one(self):
        pf = ParticleFilterEstimator(n_particles=100, seed=0)
        F  = np.eye(2) * 0.99
        pf.predict(F=F)
        H  = np.eye(2)
        pf.update(z=np.array([21.0, 22.0]), H=H)
        assert abs(pf.weights.sum() - 1.0) < 1e-6


# ============================================================
# Forecast Engine Tests
# ============================================================

class TestExponentialSmoothing:
    """Double exp smoothing forecaster."""

    def test_fit_and_predict_returns_correct_length(self):
        fc = ExponentialSmoothingForecaster()
        fc.fit([float(i) for i in range(50)])
        result = fc.predict(n_steps=12, last_timestamp=1000.0, dt=300.0,
                             agent_id="a", variable="occupancy")
        assert len(result.predicted_values) == 12
        assert len(result.timestamps) == 12
        assert len(result.lower_bound) == 12
        assert len(result.upper_bound) == 12

    def test_ci_lower_less_than_upper(self):
        fc = ExponentialSmoothingForecaster()
        fc.fit([20.0 + i * 0.1 for i in range(30)])
        result = fc.predict(12, 0.0, 300.0, "a", "temp")
        for lo, hi in zip(result.lower_bound, result.upper_bound):
            assert lo <= hi

    def test_predict_without_fit_raises(self):
        fc = ExponentialSmoothingForecaster()
        with pytest.raises(RuntimeError):
            fc.predict(5, 0.0, 300.0, "a", "v")

    def test_trend_captured(self):
        """Increasing history → forecast should trend upward."""
        fc = ExponentialSmoothingForecaster(alpha=0.5, beta=0.3)
        fc.fit([float(i) for i in range(20)])   # linear increasing
        result = fc.predict(5, 0.0, 300.0, "a", "v")
        # Forecast should be increasing
        assert result.predicted_values[-1] > result.predicted_values[0]


class TestForecastEngine:
    """ForecastEngine integration."""

    def test_forecast_returns_none_with_insufficient_history(self):
        engine = ForecastEngine("a")
        result = engine.forecast("occupancy", n_steps=5)
        assert result is None

    def test_forecast_after_observations(self):
        engine = ForecastEngine("a", dt_seconds=300.0)
        for i in range(30):
            engine.add_observation("occupancy", float(i % 20), timestamp=float(i * 300))
        result = engine.forecast("occupancy", n_steps=6)
        assert result is not None
        assert result.horizon == 6
        assert len(result.predicted_values) == 6

    def test_forecast_all_returns_dict(self):
        engine = ForecastEngine("a")
        for i in range(30):
            engine.add_observation("occupancy", float(i), float(i * 300))
            engine.add_observation("outdoor_temp", 10.0 + i * 0.1, float(i * 300))
        results = engine.forecast_all(n_steps=5)
        assert "occupancy" in results
        assert "outdoor_temp" in results


# ============================================================
# Optimiser Tests
# ============================================================

class TestMPCOptimiser:
    """MPC optimiser correctness and performance."""

    @pytest.fixture
    def agent_state(self):
        from digital_twin.models import (
            AgentState, ComfortConstraints, DeviceState, DeviceType
        )
        return AgentState(
            agent_id="test_agent",
            timestamp=0.0,
            zone_temps={"zone_a": 18.0, "zone_b": 19.0},  # below comfort
            occupancy={"zone_a": 5.0, "zone_b": 3.0},
            device_states={},
            constraints=ComfortConstraints(),
            grid_tariff=0.28,
            carbon_factor=200.0,
        )

    def test_solve_returns_result(self, agent_state):
        from digital_twin.optimiser import MPCOptimiser, MPCConfig
        opt    = MPCOptimiser(config=MPCConfig(horizon=6))
        result = opt.solve(agent_state, forecasts={})
        assert result.agent_id == "test_agent"
        assert result.status in ["optimal", "optimal_inaccurate", "infeasible",
                                  "fallback_rule_based", "solver_error"]

    def test_solve_time_under_5_seconds(self, agent_state):
        """Performance requirement: solve time ≤ 5 s."""
        from digital_twin.optimiser import MPCOptimiser, MPCConfig
        opt    = MPCOptimiser(config=MPCConfig(horizon=12))
        t0     = time.perf_counter()
        opt.solve(agent_state, forecasts={})
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Solve took {elapsed:.2f}s — exceeds 5s limit"

    def test_setpoints_within_bounds(self, agent_state):
        """Setpoints in optimal result should respect power limits."""
        from digital_twin.optimiser import MPCOptimiser, MPCConfig
        opt    = MPCOptimiser(config=MPCConfig(horizon=6, hvac_max_kw=8.0))
        result = opt.solve(agent_state, forecasts={})
        if result.status == "optimal":
            for k, v in result.setpoints_now.items():
                if "hvac" in k:
                    assert 0.0 <= v <= 8.0 + 0.1   # small tolerance

    def test_infeasible_result_has_safe_fallback(self, agent_state):
        """Infeasible solver returns zero-power safe fallback schedule."""
        from digital_twin.optimiser import MPCOptimiser, MPCConfig
        # Force infeasibility by setting impossible bounds
        opt = MPCOptimiser(config=MPCConfig(horizon=6))
        # Directly call private method
        result = opt._infeasible_result(
            agent_id=agent_state.agent_id,
            timestamp=agent_state.timestamp,
            solve_time=0.001,
        )
        assert result.status == "infeasible"
        for v in result.setpoints_now.values():
            assert v == 0.0

    def test_solve_with_forecast(self, agent_state):
        """Solver should accept forecast objects without crashing."""
        from digital_twin.optimiser import MPCOptimiser, MPCConfig
        from digital_twin.models import Forecast
        fc = Forecast(
            agent_id="test_agent",
            variable="outdoor_temp",
            horizon=12,
            timestamps=[float(i * 300) for i in range(12)],
            predicted_values=[10.0 + i * 0.2 for i in range(12)],
            lower_bound=[9.0] * 12,
            upper_bound=[12.0] * 12,
        )
        opt = MPCOptimiser(config=MPCConfig(horizon=6))
        result = opt.solve(agent_state, forecasts={"outdoor_temp": fc})
        assert result is not None

    def test_varying_horizon_performance(self, agent_state):
        """Solve time should remain < 5 s for horizons 6, 12, 24."""
        from digital_twin.optimiser import MPCOptimiser, MPCConfig
        for H in [6, 12, 24]:
            opt = MPCOptimiser(config=MPCConfig(horizon=H))
            t0  = time.perf_counter()
            opt.solve(agent_state, forecasts={})
            elapsed = time.perf_counter() - t0
            assert elapsed < 5.0, f"H={H}: {elapsed:.2f}s exceeds 5s"

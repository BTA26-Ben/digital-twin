"""
test_sensor_simulator.py
------------------------
Unit tests for the SensorSimulator module.

Acceptance criteria covered
---------------------------
AC-SS-01: Readings contain Gaussian noise with correct std dev.
AC-SS-02: Dropout probability is respected statistically.
AC-SS-03: Bias is applied correctly.
AC-SS-04: All sensor types are handled.
AC-SS-05: Stream produces correct number of batches.
AC-SS-06: Dynamic add/remove sensor works.
"""

import math
import pytest
import numpy as np

from digital_twin.models import SensorType
from digital_twin.sensor_simulator import (
    SensorConfig,
    SensorSimulator,
    default_building_sensors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_sensor():
    return SensorConfig(
        sensor_id="temp_zone_a",
        sensor_type=SensorType.TEMPERATURE,
        unit="°C",
        noise_std=0.5,
        dropout_prob=0.0,   # no dropout for most tests
        bias=0.0,
    )


@pytest.fixture
def sim_no_dropout(temp_sensor):
    return SensorSimulator(sensors=[temp_sensor], seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSensorReadings:
    """AC-SS-01, AC-SS-03: Noise and bias correctness."""

    def test_reading_close_to_true_value(self, sim_no_dropout):
        """Noisy reading should be within 3σ of true value (probabilistic)."""
        true_temp = 21.0
        readings  = [
            sim_no_dropout.read_sensor("temp_zone_a", true_temp, timestamp=float(i))
            for i in range(200)
        ]
        readings = [r for r in readings if r is not None]
        values = [r.value for r in readings]
        assert abs(np.mean(values) - true_temp) < 0.3   # mean close to true

    def test_noise_std_matches_config(self):
        """Empirical std of readings should match configured noise_std."""
        cfg = SensorConfig("s1", SensorType.TEMPERATURE, "°C", noise_std=1.0, dropout_prob=0.0)
        sim = SensorSimulator([cfg], seed=0)
        values = [sim.read_sensor("s1", 20.0, timestamp=float(i)).value for i in range(1000)]
        assert 0.7 < np.std(values) < 1.3   # within 30% of target std

    def test_bias_applied(self):
        """Mean reading should equal true_value + bias."""
        cfg = SensorConfig("s2", SensorType.TEMPERATURE, "°C",
                           noise_std=0.01, dropout_prob=0.0, bias=2.0)
        sim = SensorSimulator([cfg], seed=1)
        values = [sim.read_sensor("s2", 20.0, timestamp=float(i)).value for i in range(500)]
        assert abs(np.mean(values) - 22.0) < 0.2

    def test_correct_sensor_type_returned(self, sim_no_dropout):
        r = sim_no_dropout.read_sensor("temp_zone_a", 21.0, timestamp=0.0)
        assert r is not None
        assert r.sensor_type == SensorType.TEMPERATURE

    def test_correct_unit_returned(self, sim_no_dropout):
        r = sim_no_dropout.read_sensor("temp_zone_a", 21.0, timestamp=0.0)
        assert r.unit == "°C"


class TestDropout:
    """AC-SS-02: Dropout probability."""

    def test_no_dropout_when_prob_zero(self, sim_no_dropout):
        """With dropout_prob=0, every read should return a reading."""
        for i in range(50):
            r = sim_no_dropout.read_sensor("temp_zone_a", 21.0, timestamp=float(i))
            assert r is not None

    def test_dropout_rate_approximately_correct(self):
        """Empirical dropout rate should be within 5 pp of configured rate."""
        cfg     = SensorConfig("s3", SensorType.TEMPERATURE, "°C", noise_std=0.1, dropout_prob=0.2)
        sim     = SensorSimulator([cfg], seed=99)
        n_total = 2000
        n_dropped = sum(1 for i in range(n_total)
                        if sim.read_sensor("s3", 20.0, timestamp=float(i)) is None)
        empirical_rate = n_dropped / n_total
        assert abs(empirical_rate - 0.2) < 0.05


class TestReadAll:
    """read_all returns correct subset when some sensors have no true value."""

    def test_read_all_returns_only_available(self):
        cfgs = [
            SensorConfig("t1", SensorType.TEMPERATURE, "°C", dropout_prob=0.0),
            SensorConfig("t2", SensorType.TEMPERATURE, "°C", dropout_prob=0.0),
        ]
        sim      = SensorSimulator(cfgs, seed=7)
        readings = sim.read_all(true_values={"t1": 20.0}, timestamp=0.0)
        assert len(readings) == 1
        assert readings[0].sensor_id == "t1"

    def test_read_all_empty_when_no_true_values(self, sim_no_dropout):
        readings = sim_no_dropout.read_all(true_values={}, timestamp=0.0)
        assert readings == []


class TestStream:
    """AC-SS-05: Stream produces the expected number of timestep batches."""

    def test_stream_length(self):
        cfg    = SensorConfig("s", SensorType.TEMPERATURE, "°C", dropout_prob=0.0)
        sim    = SensorSimulator([cfg], seed=3)
        seq    = [{"s": float(i)} for i in range(5)]
        batches = list(sim.stream(seq, start_time=0.0, dt=60.0))
        assert len(batches) == 5

    def test_stream_timestamps_monotonically_increasing(self):
        cfg    = SensorConfig("s", SensorType.TEMPERATURE, "°C", dropout_prob=0.0)
        sim    = SensorSimulator([cfg], seed=3)
        seq    = [{"s": float(i)} for i in range(5)]
        batches = list(sim.stream(seq, start_time=1000.0, dt=300.0))
        timestamps = [b[0].timestamp for b in batches if b]
        assert timestamps == sorted(timestamps)


class TestDynamicSensors:
    """AC-SS-06: Dynamic sensor registration."""

    def test_add_sensor(self, sim_no_dropout):
        new_cfg = SensorConfig("new_s", SensorType.POWER, "kW", dropout_prob=0.0)
        sim_no_dropout.add_sensor(new_cfg)
        assert "new_s" in sim_no_dropout.sensor_ids

    def test_remove_sensor(self, sim_no_dropout):
        sim_no_dropout.remove_sensor("temp_zone_a")
        assert "temp_zone_a" not in sim_no_dropout.sensor_ids


class TestDefaultSensors:
    """Sanity check for factory helper."""

    def test_default_sensors_count(self):
        sensors = default_building_sensors()
        assert len(sensors) == 8   # defined in sensor_simulator.py

    def test_default_sensors_all_valid_types(self):
        sensors = default_building_sensors()
        for s in sensors:
            assert isinstance(s.sensor_type, SensorType)

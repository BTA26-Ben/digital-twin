"""
conftest.py
-----------
Pytest configuration and shared fixtures for the digital twin test suite.
"""

import pytest
import numpy as np
import sys
import os

# Make sure the src/ package is importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def deterministic_seed():
    """Provide a consistent seed for all tests that need it."""
    return 42


@pytest.fixture(autouse=True)
def reset_numpy_seed():
    """Reset numpy random state before each test for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def sample_telemetry_points():
    """Three example telemetry rows (from JSON schema examples)."""
    from digital_twin.models import TelemetryPoint, SensorType
    return [
        TelemetryPoint(
            timestamp=1700000000.0,
            sensor_id="temp_zone_a",
            sensor_type=SensorType.TEMPERATURE,
            value=21.3,
            unit="°C",
            quality=0.98,
            agent_id="building_a",
        ),
        TelemetryPoint(
            timestamp=1700000300.0,
            sensor_id="occ_zone_a",
            sensor_type=SensorType.OCCUPANCY,
            value=8.0,
            unit="persons",
            quality=0.9,
            agent_id="building_a",
        ),
        TelemetryPoint(
            timestamp=1700000600.0,
            sensor_id="power_hvac",
            sensor_type=SensorType.POWER,
            value=5.2,
            unit="kW",
            quality=0.99,
            agent_id="building_a",
        ),
    ]


@pytest.fixture
def example_agent_config():
    """Example agent configuration as used in documentation."""
    from digital_twin.agent_coordinator import AgentConfig
    return AgentConfig(
        agent_id="building_a",
        n_zones=2,
        dt_seconds=300.0,
        mpc_horizon=12,
        campus_budget_kw=50.0,
        seed=42,
    )

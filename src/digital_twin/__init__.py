"""
digital_twin
============
Distributed Digital Twin for Smart Building Energy Optimisation.

Modules
-------
models              – Data schemas (TelemetryPoint, AgentState, Forecast, …)
sensor_simulator    – Noisy sensor stream generation
environment_simulator – RC thermal model + occupancy
state_estimator     – Kalman filter / particle filter
forecast_engine     – Exponential smoothing / ARIMA forecasters
optimiser           – CVXPY-based MPC
agent_coordinator   – Multi-agent coordination
data_store          – SQLite-backed time-series store
metrics             – Evaluation metrics + plots
api                 – FastAPI REST server
"""

__version__ = "1.0.0"
__author__  = "Digital Twin Team"

from .models import (
    AgentState,
    ComfortConstraints,
    DeviceState,
    DeviceType,
    ExperimentRecord,
    Forecast,
    OptimisationProblem,
    OptimisationResult,
    SensorType,
    TelemetryPoint,
)

__all__ = [
    "TelemetryPoint",
    "AgentState",
    "ComfortConstraints",
    "DeviceState",
    "DeviceType",
    "Forecast",
    "OptimisationProblem",
    "OptimisationResult",
    "ExperimentRecord",
    "SensorType",
]

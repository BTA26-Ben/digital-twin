"""
forecast_engine.py
------------------
Short-term forecasting for occupancy and outdoor temperature.

Two models are provided:
  1. ExponentialSmoothingForecaster – lightweight baseline (no statsmodels
     required), suitable for the MVP.
  2. ARIMAForecaster – wraps statsmodels ARIMA; falls back gracefully if the
     library is unavailable.

Both expose the same interface:
    fit(history)  → trains/updates the model
    predict(n)    → returns a Forecast dataclass

Public API
----------
BaseForecaster       – abstract base
ExponentialSmoothing – double-exp smoothing
ARIMAForecaster      – ARIMA(p,d,q)
ForecastEngine       – manages multiple per-variable forecasters for an agent
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Forecast

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseForecaster(ABC):
    """Common interface for all forecasting models."""

    @abstractmethod
    def fit(self, history: List[float]) -> None:
        """Train or update the model on a sequence of past values."""

    @abstractmethod
    def predict(
        self,
        n_steps: int,
        last_timestamp: float,
        dt: float,
        agent_id: str,
        variable: str,
    ) -> Forecast:
        """Generate an n-step ahead forecast.

        Parameters
        ----------
        n_steps        : Number of future timesteps to forecast.
        last_timestamp : Unix epoch of the last observed point.
        dt             : Duration of each timestep in seconds.
        agent_id       : Agent identifier.
        variable       : Variable name (e.g. 'occupancy', 'outdoor_temp').
        """


# ---------------------------------------------------------------------------
# Double Exponential Smoothing (Holt's method)
# ---------------------------------------------------------------------------

class ExponentialSmoothingForecaster(BaseForecaster):
    """Holt's double exponential smoothing — trend-adjusted forecasts.

    Parameters
    ----------
    alpha : Level smoothing factor (0 < α ≤ 1).
    beta  : Trend smoothing factor (0 < β ≤ 1).
    ci_factor : Multiplier on residual std for confidence interval width.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, ci_factor: float = 1.645) -> None:
        self.alpha = alpha
        self.beta  = beta
        self.ci_factor = ci_factor
        self._level: float = 0.0
        self._trend: float = 0.0
        self._residuals: List[float] = []
        self._fitted = False

    def fit(self, history: List[float]) -> None:
        """Initialise level and trend from history."""
        if len(history) < 2:
            self._level = history[0] if history else 0.0
            self._trend = 0.0
            self._fitted = True
            return

        # Initialise: level = first value, trend = average change
        self._level = float(history[0])
        self._trend = float(np.mean(np.diff(history[:min(len(history), 10)])))
        self._residuals = []

        for t, y in enumerate(history[1:], start=1):
            prev_level = self._level
            self._level = self.alpha * y + (1 - self.alpha) * (self._level + self._trend)
            self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend
            y_hat = prev_level + self._trend
            self._residuals.append(y - y_hat)

        self._fitted = True

    def predict(
        self,
        n_steps: int,
        last_timestamp: float,
        dt: float,
        agent_id: str,
        variable: str,
    ) -> Forecast:
        """Produce n-step ahead forecasts with confidence intervals."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        residual_std = float(np.std(self._residuals)) if self._residuals else 1.0
        timestamps = [last_timestamp + (i + 1) * dt for i in range(n_steps)]
        predicted  = []
        lower      = []
        upper      = []

        level = self._level
        trend = self._trend

        for h in range(1, n_steps + 1):
            y_hat = level + h * trend
            half_width = self.ci_factor * residual_std * np.sqrt(h)
            predicted.append(float(y_hat))
            lower.append(float(y_hat - half_width))
            upper.append(float(y_hat + half_width))

        return Forecast(
            agent_id=agent_id,
            variable=variable,
            horizon=n_steps,
            timestamps=timestamps,
            predicted_values=predicted,
            lower_bound=lower,
            upper_bound=upper,
            model_name="double_exp_smoothing",
        )


# ---------------------------------------------------------------------------
# ARIMA forecaster (optional — requires statsmodels)
# ---------------------------------------------------------------------------

class ARIMAForecaster(BaseForecaster):
    """ARIMA(p,d,q) forecaster wrapping statsmodels.

    Falls back to ExponentialSmoothingForecaster if statsmodels is not
    installed, with a warning.

    Parameters
    ----------
    order : (p, d, q) ARIMA order tuple.
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)) -> None:
        self.order = order
        self._model = None
        self._result = None
        self._fallback: Optional[ExponentialSmoothingForecaster] = None
        self._history: List[float] = []

        try:
            import statsmodels.tsa.arima.model  # noqa: F401
            self._has_statsmodels = True
        except ImportError:
            log.warning("statsmodels not found; ARIMA falling back to ExpSmoothing.")
            self._has_statsmodels = False
            self._fallback = ExponentialSmoothingForecaster()

    def fit(self, history: List[float]) -> None:
        self._history = list(history)
        if not self._has_statsmodels:
            self._fallback.fit(history)
            return

        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._model = ARIMA(history, order=self.order)
                self._result = self._model.fit()
            except Exception as e:
                log.warning(f"ARIMA fit failed ({e}); falling back to ExpSmoothing.")
                self._has_statsmodels = False
                self._fallback = ExponentialSmoothingForecaster()
                self._fallback.fit(history)

    def predict(
        self,
        n_steps: int,
        last_timestamp: float,
        dt: float,
        agent_id: str,
        variable: str,
    ) -> Forecast:
        if not self._has_statsmodels or self._result is None:
            return self._fallback.predict(n_steps, last_timestamp, dt, agent_id, variable)

        try:
            forecast_result = self._result.get_forecast(steps=n_steps)
            predicted = list(forecast_result.predicted_mean)
            conf_int  = forecast_result.conf_int(alpha=0.10)
            lower     = list(conf_int.iloc[:, 0])
            upper     = list(conf_int.iloc[:, 1])
        except Exception as e:
            log.warning(f"ARIMA predict failed ({e}); using fallback.")
            self._fallback = ExponentialSmoothingForecaster()
            self._fallback.fit(self._history)
            return self._fallback.predict(n_steps, last_timestamp, dt, agent_id, variable)

        timestamps = [last_timestamp + (i + 1) * dt for i in range(n_steps)]
        return Forecast(
            agent_id=agent_id,
            variable=variable,
            horizon=n_steps,
            timestamps=timestamps,
            predicted_values=[float(v) for v in predicted],
            lower_bound=[float(v) for v in lower],
            upper_bound=[float(v) for v in upper],
            model_name=f"ARIMA{self.order}",
        )


# ---------------------------------------------------------------------------
# Forecast Engine — manages multiple forecasters for one agent
# ---------------------------------------------------------------------------

class ForecastEngine:
    """Manages a set of forecasters (one per variable) for a building agent.

    Parameters
    ----------
    agent_id         : Owning agent.
    dt_seconds       : Timestep duration.
    occupancy_model  : Forecaster type for occupancy ('exp_smoothing'/'arima').
    temp_model       : Forecaster type for outdoor temperature.
    history_window   : How many past timesteps to keep for fitting.
    seed             : RNG seed (currently unused; reserved for future use).

    Usage
    -----
    >>> engine = ForecastEngine("agent_0")
    >>> engine.add_observation("occupancy", 15.0, t=ts)
    >>> fc = engine.forecast("occupancy", n_steps=12)
    """

    def __init__(
        self,
        agent_id:        str,
        dt_seconds:      float = 300.0,
        occupancy_model: str   = "exp_smoothing",
        temp_model:      str   = "exp_smoothing",
        history_window:  int   = 288,   # 24 h @ 5-min
        seed:            int   = 42,
    ) -> None:
        self.agent_id       = agent_id
        self.dt             = dt_seconds
        self.history_window = history_window

        # History buffers: variable → list of (timestamp, value)
        self._history: Dict[str, List[Tuple[float, float]]] = {}

        # Forecasters: variable → BaseForecaster
        self._forecasters: Dict[str, BaseForecaster] = {}
        self._model_type: Dict[str, str] = {
            "occupancy":    occupancy_model,
            "outdoor_temp": temp_model,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_observation(self, variable: str, value: float, timestamp: float) -> None:
        """Append one observation to a variable's history buffer."""
        if variable not in self._history:
            self._history[variable] = []
        self._history[variable].append((timestamp, value))
        # Trim to window size
        if len(self._history[variable]) > self.history_window:
            self._history[variable] = self._history[variable][-self.history_window:]

    def fit(self, variable: str) -> None:
        """Train / update the forecaster for `variable`."""
        if variable not in self._history or len(self._history[variable]) < 2:
            log.debug(f"Insufficient history for variable '{variable}'; skip fit.")
            return

        values = [v for _, v in self._history[variable]]
        forecaster = self._get_or_create_forecaster(variable)
        forecaster.fit(values)

    def forecast(self, variable: str, n_steps: int) -> Optional[Forecast]:
        """Generate a forecast for `variable` over `n_steps` timesteps.

        Returns None if insufficient history is available.
        """
        if variable not in self._history or len(self._history[variable]) < 2:
            return None

        self.fit(variable)
        forecaster = self._get_or_create_forecaster(variable)
        last_ts = self._history[variable][-1][0]
        return forecaster.predict(
            n_steps=n_steps,
            last_timestamp=last_ts,
            dt=self.dt,
            agent_id=self.agent_id,
            variable=variable,
        )

    def forecast_all(self, n_steps: int) -> Dict[str, Optional[Forecast]]:
        """Forecast all tracked variables."""
        return {var: self.forecast(var, n_steps) for var in self._history}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_forecaster(self, variable: str) -> BaseForecaster:
        if variable not in self._forecasters:
            model_type = self._model_type.get(variable, "exp_smoothing")
            if model_type == "arima":
                self._forecasters[variable] = ARIMAForecaster()
            else:
                self._forecasters[variable] = ExponentialSmoothingForecaster()
        return self._forecasters[variable]

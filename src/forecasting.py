"""Demand forecasting methods.

Two lightweight, transparent baselines that cover most short-term planning
needs in a small warehouse or social grocery:

- a moving average (good for stable, noisy demand);
- a linear trend fit by least squares (good when demand drifts up or down).

Both are pure NumPy so the results are easy to audit by hand.
"""

from __future__ import annotations

import numpy as np


def moving_average_forecast(history, window: int = 3, horizon: int = 1):
    """Forecast the next ``horizon`` periods with a simple moving average.

    The forecast for every future period is the average of the last
    ``window`` observed values. It does not extrapolate a trend; it assumes
    demand stays around its recent level.

    Parameters
    ----------
    history : sequence of float
        Historical demand, ordered from oldest to most recent.
    window : int
        Number of recent periods to average.
    horizon : int
        How many future periods to forecast.

    Returns
    -------
    numpy.ndarray
        Forecast values, one per future period.
    """
    history = np.asarray(history, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > history.size:
        raise ValueError("window cannot be larger than the history length")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    level = history[-window:].mean()
    return np.full(horizon, level)


def linear_trend_forecast(history, horizon: int = 1):
    """Forecast the next ``horizon`` periods by fitting a straight line.

    A least-squares line ``demand = a * t + b`` is fitted to the history and
    extended into the future. Useful when demand is steadily growing or
    shrinking.

    Parameters
    ----------
    history : sequence of float
        Historical demand, ordered from oldest to most recent.
    horizon : int
        How many future periods to forecast.

    Returns
    -------
    numpy.ndarray
        Forecast values, one per future period (clipped at zero, since
        negative demand is meaningless).
    """
    history = np.asarray(history, dtype=float)
    if history.size < 2:
        raise ValueError("need at least 2 observations to fit a trend")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    t = np.arange(history.size)
    slope, intercept = np.polyfit(t, history, deg=1)

    future_t = np.arange(history.size, history.size + horizon)
    forecast = slope * future_t + intercept
    return np.clip(forecast, 0.0, None)


def mean_absolute_error(actual, predicted) -> float:
    """Mean absolute error between two equal-length sequences."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    return float(np.mean(np.abs(actual - predicted)))

import math

import numpy as np
import pytest

from src.forecasting import (
    moving_average_forecast,
    linear_trend_forecast,
    mean_absolute_error,
)


def test_moving_average_uses_last_window():
    history = [10, 20, 30, 40]
    forecast = moving_average_forecast(history, window=2, horizon=1)
    assert forecast.tolist() == [35.0]  # mean of last two: (30 + 40) / 2


def test_moving_average_horizon_is_flat():
    forecast = moving_average_forecast([5, 5, 5], window=3, horizon=4)
    assert forecast.tolist() == [5.0, 5.0, 5.0, 5.0]


def test_moving_average_rejects_window_larger_than_history():
    with pytest.raises(ValueError):
        moving_average_forecast([1, 2], window=5)


def test_linear_trend_extends_straight_line():
    history = [0, 2, 4, 6]  # perfectly linear, slope 2
    forecast = linear_trend_forecast(history, horizon=2)
    assert np.allclose(forecast, [8.0, 10.0])


def test_linear_trend_is_clipped_at_zero():
    history = [10, 8, 6, 4]  # decreasing; would go negative if not clipped
    forecast = linear_trend_forecast(history, horizon=5)
    assert (forecast >= 0).all()


def test_linear_trend_needs_two_points():
    with pytest.raises(ValueError):
        linear_trend_forecast([42], horizon=1)


def test_mean_absolute_error():
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0
    assert math.isclose(mean_absolute_error([1, 2, 3], [2, 2, 5]), 1.0)


def test_mean_absolute_error_shape_mismatch():
    with pytest.raises(ValueError):
        mean_absolute_error([1, 2], [1, 2, 3])

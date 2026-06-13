"""Logistics optimization toolkit: demand forecasting and inventory control."""

from .forecasting import (
    moving_average_forecast,
    linear_trend_forecast,
    mean_absolute_error,
)
from .inventory import (
    economic_order_quantity,
    safety_stock,
    reorder_point,
    InventoryPolicy,
)

__all__ = [
    "moving_average_forecast",
    "linear_trend_forecast",
    "mean_absolute_error",
    "economic_order_quantity",
    "safety_stock",
    "reorder_point",
    "InventoryPolicy",
]

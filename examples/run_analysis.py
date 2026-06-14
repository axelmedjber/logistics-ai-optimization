"""End-to-end example: forecast demand and derive an inventory policy.

Run from the repository root:

    python -m examples.run_analysis

It loads ``data/demand_history.csv`` (daily demand for one product), compares
two forecasting baselines on the last 7 days, then computes a full (Q, R)
replenishment policy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting import (
    moving_average_forecast,
    linear_trend_forecast,
    mean_absolute_error,
)
from src.inventory import InventoryPolicy

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "demand_history.csv"

# Business assumptions for the inventory policy.
ORDER_COST = 25.0        # cost of placing one purchase order (currency)
HOLDING_COST = 3.0       # cost of holding one unit for a year (currency)
LEAD_TIME_DAYS = 5       # days between placing and receiving an order
SERVICE_LEVEL = 0.95     # target probability of no stock-out during lead time
BACKTEST_DAYS = 7        # how many recent days to hold out for evaluation


def main() -> None:
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    demand = df["units_sold"].to_numpy(dtype=float)

    print(f"Loaded {len(demand)} days of demand from {DATA_FILE.name}")
    print(f"  mean = {demand.mean():.1f} units/day, std = {demand.std(ddof=1):.1f}\n")

    # --- Forecast evaluation: hold out the last BACKTEST_DAYS days ---------
    train, test = demand[:-BACKTEST_DAYS], demand[-BACKTEST_DAYS:]

    ma = moving_average_forecast(train, window=7, horizon=BACKTEST_DAYS)
    trend = linear_trend_forecast(train, horizon=BACKTEST_DAYS)

    print(f"Backtest on the last {BACKTEST_DAYS} days (mean absolute error):")
    print(f"  moving average (7d) : {mean_absolute_error(test, ma):.2f}")
    print(f"  linear trend        : {mean_absolute_error(test, trend):.2f}\n")

    # --- Inventory policy from the full history ---------------------------
    annual_demand = demand.mean() * 365
    policy = InventoryPolicy.build(
        annual_demand=annual_demand,
        order_cost=ORDER_COST,
        holding_cost=HOLDING_COST,
        avg_demand_per_period=demand.mean(),
        demand_std_per_period=demand.std(ddof=1),
        lead_time_periods=LEAD_TIME_DAYS,
        service_level=SERVICE_LEVEL,
    )

    print(f"Recommended inventory policy (service level {SERVICE_LEVEL:.0%}):")
    print(f"  order quantity (EOQ) : {policy.order_quantity:.0f} units")
    print(f"  reorder point        : {policy.reorder_point:.0f} units")
    print(f"  safety stock         : {policy.safety_stock:.0f} units")
    print(f"  orders per year      : {policy.orders_per_year:.1f}")
    print(
        "\nReading: place an order of "
        f"{policy.order_quantity:.0f} units whenever stock falls to "
        f"{policy.reorder_point:.0f} units."
    )


if __name__ == "__main__":
    main()

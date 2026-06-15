"""Logistics Optimization -- interactive web demo (Streamlit).

Run locally with:

    streamlit run app.py

Adjust the cost and lead-time assumptions in the sidebar and watch the demand
forecast and the inventory policy (EOQ, reorder point, safety stock) update
live. The forecasting and inventory logic lives in src/ and is unit tested
independently of this UI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.forecasting import (
    moving_average_forecast,
    linear_trend_forecast,
    mean_absolute_error,
)
from src.inventory import InventoryPolicy
from src import ingest

DATA_FILE = Path(__file__).resolve().parent / "data" / "demand_history.csv"

st.set_page_config(page_title="Logistics Optimization", page_icon="📈", layout="wide")
st.title("📈 Logistics Optimization")
st.caption("Demand forecasting and inventory control (EOQ, safety stock, reorder point).")

# --- Data source ---------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload a demand CSV (date, units_sold)", type="csv")
with open(DATA_FILE, "rb") as _f:
    st.sidebar.download_button(
        "⬇️ Download template / sample CSV",
        _f.read(),
        file_name="demand_template.csv",
        mime="text/csv",
        help="Two columns: date (YYYY-MM-DD) and units_sold (integer).",
    )
if uploaded is None:
    df = ingest.coerce(pd.read_csv(DATA_FILE))
else:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this file as CSV: {exc}")
        st.stop()

    if set(ingest.REQUIRED).issubset(raw.columns):
        df = ingest.coerce(raw)
    else:
        st.warning(
            "Your CSV columns don't match the expected format. Map them below."
        )
        cols = list(raw.columns)

        def _idx(field):
            g = ingest.guess_column(cols, field)
            return cols.index(g) if g in cols else 0

        m1, m2 = st.columns(2)
        date_col = m1.selectbox("Date column", cols, index=_idx("date"))
        units_col = m2.selectbox("Demand / units column", cols, index=_idx("units_sold"))
        df = ingest.apply_mapping(raw, date_col, units_col)

    if df.empty:
        st.error("No valid rows after parsing — check your column mapping.")
        st.stop()

demand = df["units_sold"].astype(float).to_numpy()

# --- Assumptions ---------------------------------------------------------
st.sidebar.header("Assumptions")
order_cost = st.sidebar.slider("Order cost (per order)", 5, 100, 25)
holding_cost = st.sidebar.slider("Holding cost (per unit / year)", 1, 20, 3)
lead_time = st.sidebar.slider("Lead time (days)", 1, 30, 5)
service_level = st.sidebar.slider("Service level", 0.80, 0.99, 0.95, step=0.01)
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)
window = st.sidebar.slider("Moving-average window (days)", 3, 21, 7)

# --- Forecast ------------------------------------------------------------
ma = moving_average_forecast(demand, window=window, horizon=horizon)
trend = linear_trend_forecast(demand, horizon=horizon)

future_dates = pd.date_range(
    df["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
)
chart_df = pd.DataFrame(index=pd.concat([df["date"], pd.Series(future_dates)]).values)
chart_df["history"] = list(demand) + [np.nan] * horizon
chart_df["moving average"] = [np.nan] * len(demand) + list(ma)
chart_df["linear trend"] = [np.nan] * len(demand) + list(trend)

# Backtest accuracy on the last `horizon` days.
if len(demand) > horizon + window:
    train, test = demand[:-horizon], demand[-horizon:]
    mae_ma = mean_absolute_error(test, moving_average_forecast(train, window, horizon))
    mae_tr = mean_absolute_error(test, linear_trend_forecast(train, horizon))
else:
    mae_ma = mae_tr = float("nan")

# --- Inventory policy ----------------------------------------------------
annual_demand = demand.mean() * 365
policy = InventoryPolicy.build(
    annual_demand=annual_demand,
    order_cost=order_cost,
    holding_cost=holding_cost,
    avg_demand_per_period=demand.mean(),
    demand_std_per_period=demand.std(ddof=1),
    lead_time_periods=lead_time,
    service_level=service_level,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Order quantity (EOQ)", f"{policy.order_quantity:.0f}")
c2.metric("Reorder point", f"{policy.reorder_point:.0f}")
c3.metric("Safety stock", f"{policy.safety_stock:.0f}")
c4.metric("Orders / year", f"{policy.orders_per_year:.1f}")

st.info(
    f"Place an order of **{policy.order_quantity:.0f} units** whenever stock "
    f"falls to **{policy.reorder_point:.0f} units**."
)

st.divider()

left, right = st.columns([2, 1])
with left:
    st.subheader("Demand history & forecast")
    st.line_chart(chart_df)
with right:
    st.subheader("Forecast accuracy")
    st.caption(f"Mean absolute error on the last {horizon} days (lower is better):")
    st.metric("Moving average", f"{mae_ma:.2f}" if mae_ma == mae_ma else "n/a")
    st.metric("Linear trend", f"{mae_tr:.2f}" if mae_tr == mae_tr else "n/a")
    st.metric("Avg demand", f"{demand.mean():.1f} /day")

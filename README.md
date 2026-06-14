# Logistics Optimization

A small, dependency-light Python toolkit for two everyday supply-chain
decisions:

1. **Demand forecasting** — how much will we sell next week?
2. **Inventory control** — how much to order, and when?

It pairs **transparent, hand-auditable formulas** (EOQ, safety stock, reorder
point) with a **deep-learning demand model** (fastai), each with honest
benchmarks and tests on real data.

## Deep learning demand model

`notebooks/demand_forecasting_fastai.ipynb` trains a tabular **neural network**
(fastai / PyTorch) to predict demand from calendar and weather features, on the
public **UCI Bike Sharing** dataset. It is benchmarked on a held-out validation
set against two honest baselines:

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Mean predictor | 1497 | 1808 | -0.0 |
| Linear regression | 525 | 733 | 0.8 |
| **Deep learning (fastai)** | **483** | **651** | **0.9** |

The network beats both baselines on every metric. Run it end to end:

```bash
pip install -r requirements-ml.txt
python -m ml.train                 # train + benchmark, exports models/demand_fastai.pkl
python -m ml.predict               # predict demand on new rows
```

> The deep-learning stack (PyTorch) is kept in a separate `requirements-ml.txt`
> so the lightweight Streamlit demo below stays easy to deploy. The notebook
> also runs as-is on Kaggle / Google Colab (free GPU).

## Features

| Module | What it does |
| --- | --- |
| `ml/` + `notebooks/` | **Deep-learning** demand model (fastai): training, benchmarking and CSV inference. |
| `src/forecasting.py` | Moving-average and least-squares **linear-trend** forecasts (used as baselines), plus a mean-absolute-error metric. |
| `src/inventory.py` | **Economic Order Quantity (EOQ)**, **safety stock** for a target service level, **reorder point**, and a combined `(Q, R)` policy. |
| `app.py` | Interactive web demo (Streamlit): tune the cost/lead-time assumptions and watch the forecast and inventory policy update live. |
| `examples/run_analysis.py` | Loads 90 days of demand, backtests the forecasts, and prints a full replenishment policy. |

## Web demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sliders for order cost, holding cost, lead time and service level recompute
the EOQ, reorder point and safety stock in real time, next to a demand-forecast
chart. You can also upload your own `date, units_sold` CSV.

> **Deploy a public demo (free):** push this repo to GitHub, go to
> [share.streamlit.io](https://share.streamlit.io), click *New app*, pick the
> repo and set the main file to `app.py`. You get a public URL to share.

## Command-line demo

```bash
pip install -r requirements.txt
python -m examples.run_analysis
```

Example output:

```
Loaded 90 days of demand from demand_history.csv
  mean = 45.5 units/day, std = 7.5

Backtest on the last 7 days (mean absolute error):
  moving average (7d) : 5.45
  linear trend        : 5.54

Recommended inventory policy (service level 95%):
  order quantity (EOQ) : 526 units
  reorder point        : 255 units
  safety stock         : 28 units
  orders per year      : 31.6

Reading: place an order of 526 units whenever stock falls to 255 units.
```

## The formulas

**Economic Order Quantity** balances ordering cost against holding cost:

```
EOQ = sqrt(2 * D * S / H)
```
`D` = annual demand, `S` = cost per order, `H` = holding cost per unit per year.

**Safety stock** buffers against demand variability during the lead time:

```
SS = z * sigma * sqrt(L)
```
`z` = service-level z-score, `sigma` = demand std. dev. per period, `L` = lead time.

**Reorder point** is when to place the next order:

```
ROP = (average demand x lead time) + safety stock
```

## Tests

```bash
pytest
```

## Project layout

```
logistics-ai-optimization/
├── src/
│   ├── forecasting.py     # demand forecasting baselines
│   └── inventory.py       # EOQ, safety stock, reorder point
├── examples/
│   └── run_analysis.py    # end-to-end demo
├── data/
│   └── demand_history.csv # 90 days of sample daily demand
├── tests/                 # unit tests (pytest)
└── requirements.txt
```

## License

MIT

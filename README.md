# Logistics Optimization

A small, dependency-light Python toolkit for two everyday supply-chain
decisions:

1. **Demand forecasting** — how much will we sell next week?
2. **Inventory control** — how much to order, and when?

The goal is transparency over hype: every method is a well-known,
hand-auditable formula (no black-box models, no heavyweight ML stack), with
unit tests and a runnable example on realistic data.

## Features

| Module | What it does |
| --- | --- |
| `src/forecasting.py` | Moving-average and least-squares **linear-trend** forecasts, plus a mean-absolute-error metric for backtesting. |
| `src/inventory.py` | **Economic Order Quantity (EOQ)**, **safety stock** for a target service level, **reorder point**, and a combined `(Q, R)` policy. |
| `examples/run_analysis.py` | Loads 90 days of demand, backtests the forecasts, and prints a full replenishment policy. |

## Quick start

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

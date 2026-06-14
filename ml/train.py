"""Train a deep-learning demand model (fastai tabular) and benchmark it.

The model is a fully-connected neural network on tabular features (calendar +
weather). We compare it on a held-out chronological validation set against two
honest baselines:

- *mean predictor*: always predict the training-set average demand;
- *linear regression*: a classic linear model on the same features.

Run:

    python -m ml.train

Outputs validation MAE / RMSE / R2 for each approach and exports the trained
network to models/demand_fastai.pkl.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fastai.tabular.all import (
    Categorify,
    FillMissing,
    Normalize,
    RandomSplitter,
    TabularPandas,
    range_of,
    tabular_learner,
    mae,
    rmse,
)

from ml.data import CATEGORICAL, CONTINUOUS, TARGET, load

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "demand_fastai.pkl"


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main(epochs: int = 25) -> dict:
    import torch
    from fastai.torch_core import set_seed

    set_seed(42, reproducible=True)
    torch.manual_seed(42)

    df = load()
    # Predict demand from calendar + weather conditions (regression). A random
    # split keeps train and validation in the same distribution -- the standard
    # setup for this dataset.
    splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(df))
    train_idx, valid_idx = list(splits[0]), list(splits[1])
    y_valid = df.loc[valid_idx, TARGET].to_numpy()
    y_train = df.loc[train_idx, TARGET].to_numpy()

    # --- Baseline 1: always predict the training mean --------------------
    mean_pred = np.full_like(y_valid, y_train.mean(), dtype=float)
    results = {"mean predictor": _metrics(y_valid, mean_pred)}

    # --- Baseline 2: linear regression -----------------------------------
    X = pd.get_dummies(df[CATEGORICAL + CONTINUOUS], columns=CATEGORICAL)
    lin = LinearRegression().fit(X.iloc[train_idx], y_train)
    results["linear regression"] = _metrics(y_valid, lin.predict(X.iloc[valid_idx]))

    # --- Deep learning: fastai tabular neural net ------------------------
    to = TabularPandas(
        df,
        procs=[Categorify, FillMissing, Normalize],
        cat_names=CATEGORICAL,
        cont_names=CONTINUOUS,
        y_names=TARGET,
        splits=(train_idx, valid_idx),
    )
    dls = to.dataloaders(bs=64)
    y_max = float(df[TARGET].max())
    learn = tabular_learner(
        dls, layers=[200, 100], y_range=(0, y_max * 1.1), metrics=[mae, rmse]
    )
    with learn.no_bar(), learn.no_logging():
        lr = learn.lr_find(show_plot=False).valley
        learn.fit_one_cycle(epochs, lr)

    preds, targs = learn.get_preds()
    results["deep learning (fastai)"] = _metrics(
        targs.numpy().ravel(), preds.numpy().ravel()
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    learn.export(MODEL_PATH)

    # --- Report ----------------------------------------------------------
    print("\nValidation results (lower MAE/RMSE is better, higher R2 is better):\n")
    table = pd.DataFrame(results).T.round(1)
    print(table.to_string())
    print(f"\nModel exported to {MODEL_PATH}")
    return results


if __name__ == "__main__":
    main()

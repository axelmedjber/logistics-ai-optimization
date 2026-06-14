"""Run the trained deep-learning model on new data.

Given a CSV with the same feature columns as the training data (calendar +
weather, without the target), this loads the exported fastai model and adds a
``predicted_demand`` column.

Run:

    python -m ml.predict            # predicts on a few sample rows
    python -m ml.predict my.csv     # predicts on your own file
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastai.tabular.all import load_learner

from ml.data import CATEGORICAL, CONTINUOUS, load

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "demand_fastai.pkl"


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with an added integer ``predicted_demand`` column."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python -m ml.train` first."
        )
    learn = load_learner(MODEL_PATH)
    dl = learn.dls.test_dl(df)
    preds, _ = learn.get_preds(dl=dl)
    out = df.copy()
    out["predicted_demand"] = preds.numpy().ravel().round().astype(int)
    return out


def predict_csv(path: str) -> pd.DataFrame:
    return predict(pd.read_csv(path))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = predict_csv(sys.argv[1])
    else:
        # Demo on a few rows of the bundled dataset.
        sample = load()[CATEGORICAL + CONTINUOUS].head(5)
        result = predict(sample)
    cols = CATEGORICAL + CONTINUOUS + ["predicted_demand"]
    print(result[[c for c in cols if c in result.columns]].to_string(index=False))

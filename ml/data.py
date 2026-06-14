"""Load the UCI Bike Sharing dataset for demand forecasting.

The dataset records the daily number of bikes rented (``cnt`` -- our demand
target) together with calendar and weather features. It is a well-known,
freely available regression benchmark and a realistic stand-in for daily
product demand driven by season, weekday and conditions.

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
CACHE = Path(__file__).resolve().parent.parent / "data" / "bike_sharing_day.csv"

# Columns that would leak the target (they sum to `cnt`) or are pure indices.
LEAKAGE = ["casual", "registered", "instant", "dteday"]

CATEGORICAL = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
CONTINUOUS = ["temp", "atemp", "hum", "windspeed"]
TARGET = "cnt"


def download(force: bool = False) -> Path:
    """Download and cache the daily bike-sharing CSV. Returns its path."""
    if CACHE.exists() and not force:
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(DATA_URL, timeout=60) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open("day.csv") as f:
            df = pd.read_csv(f)
    df.to_csv(CACHE, index=False)
    return CACHE


def load() -> pd.DataFrame:
    """Load the dataset, dropping leakage/index columns."""
    df = pd.read_csv(download())
    return df.drop(columns=[c for c in LEAKAGE if c in df.columns])


def train_valid_split(df: pd.DataFrame, valid_frac: float = 0.2):
    """Chronological split (no shuffling -- this is a time series).

    Returns ``(train_idx, valid_idx)`` index lists for the last
    ``valid_frac`` of the rows as validation.
    """
    n = len(df)
    cut = int(n * (1 - valid_frac))
    return list(range(cut)), list(range(cut, n))


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print("Target stats (cnt):", df[TARGET].describe()[["mean", "min", "max"]].to_dict())
    print("Columns:", list(df.columns))

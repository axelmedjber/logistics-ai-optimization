"""Tests for the deep-learning module.

These are skipped automatically when the ML extras (fastai/torch) are not
installed, so the lightweight test run stays fast.
"""

import pytest

pytest.importorskip("fastai")

from ml.data import CATEGORICAL, CONTINUOUS, TARGET, load  # noqa: E402
from ml.predict import MODEL_PATH, predict  # noqa: E402


def test_load_has_expected_columns():
    df = load()
    for col in CATEGORICAL + CONTINUOUS + [TARGET]:
        assert col in df.columns
    assert len(df) > 100


def test_predict_adds_one_value_per_row():
    if not MODEL_PATH.exists():
        pytest.skip("trained model not present (run `python -m ml.train`)")
    sample = load()[CATEGORICAL + CONTINUOUS].head(8)
    out = predict(sample)
    assert "predicted_demand" in out.columns
    assert len(out) == len(sample)
    assert (out["predicted_demand"] >= 0).all()

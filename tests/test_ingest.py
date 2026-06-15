import pandas as pd

from src.ingest import REQUIRED, apply_mapping, coerce, guess_column


def test_guess_column_french_headers():
    cols = ["Jour", "Ventes"]
    assert guess_column(cols, "date") == "Jour"
    assert guess_column(cols, "units_sold") == "Ventes"


def test_apply_mapping_normalizes_and_sorts():
    raw = pd.DataFrame(
        {
            "Jour": ["2025-01-03", "2025-01-01", "2025-01-02"],
            "Ventes": ["30", "10", "20"],
        }
    )
    df = apply_mapping(raw, "Jour", "Ventes")
    assert list(df.columns) == REQUIRED
    assert df["units_sold"].tolist() == [10.0, 20.0, 30.0]  # sorted by date
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_coerce_drops_invalid_rows():
    raw = pd.DataFrame(
        {"date": ["2025-01-01", "bad"], "units_sold": [10, "x"]}
    )
    df = coerce(raw)
    assert len(df) == 1

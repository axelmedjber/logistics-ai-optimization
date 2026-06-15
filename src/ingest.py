"""Turn an arbitrary uploaded CSV into the canonical demand format.

The dashboard expects ``date, units_sold``. Real exports often use other
names, so this module maps any two columns onto that schema and cleans types.
"""

from __future__ import annotations

import pandas as pd

REQUIRED = ["date", "units_sold"]

HINTS = {
    "date": ["date", "jour", "day", "time", "period"],
    "units_sold": [
        "units", "sold", "sales", "demand", "quantity", "quantite", "quantité",
        "qty", "qte", "ventes", "demande", "nombre", "volume",
    ],
}


def guess_column(columns, field: str):
    """Best-guess column name for a canonical field, or None."""
    for col in columns:
        low = str(col).lower()
        if any(k in low for k in HINTS.get(field, [])):
            return col
    return None


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df = df.dropna(subset=["date", "units_sold"])
    return df.sort_values("date")[REQUIRED].reset_index(drop=True)


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a frame that already has the required columns."""
    return _finalize(df.copy())


def apply_mapping(raw: pd.DataFrame, date_col: str, units_col: str) -> pd.DataFrame:
    """Map arbitrary columns to ``date, units_sold``."""
    out = pd.DataFrame({"date": raw[date_col], "units_sold": raw[units_col]})
    return _finalize(out)

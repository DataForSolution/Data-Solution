"""Dataset contracts for the historical automobile-insurance fraud data."""
from __future__ import annotations

import pandas as pd

TARGET = "FraudFound_P"
YEAR = "Year"
ROW_ID = "PolicyNumber"
REQUIRED_COLUMNS = {TARGET, YEAR, ROW_ID}


def audit_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Return compact data-quality facts and fail on missing required fields."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("dataset is empty")
    target = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("FraudFound_P must be binary 0/1")
    years = sorted(pd.to_numeric(frame[YEAR], errors="raise").astype(int).unique().tolist())
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "fraud_count": int(target.sum()),
        "fraud_prevalence": float(target.mean()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "years": years,
        "unique_policy_numbers": int(frame[ROW_ID].nunique(dropna=False)),
    }


def temporal_split(
    frame: pd.DataFrame,
    *,
    train_year: int = 1994,
    validation_year: int = 1995,
    test_year: int = 1996,
):
    """Create chronological train/validation/test splits and exclude ID/time predictors."""
    audit_frame(frame)
    years = pd.to_numeric(frame[YEAR], errors="raise").astype(int)
    parts = {}
    for name, year in (
        ("train", train_year),
        ("validation", validation_year),
        ("test", test_year),
    ):
        part = frame.loc[years.eq(year)].copy()
        if part.empty:
            raise ValueError(f"no rows for {name} year {year}")
        y = pd.to_numeric(part.pop(TARGET), errors="raise").astype(int)
        part = part.drop(columns=[ROW_ID, YEAR])
        parts[name] = (part, y)
    return parts["train"], parts["validation"], parts["test"]

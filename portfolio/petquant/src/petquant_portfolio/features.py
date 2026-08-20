"""Data-normalization helpers recovered from PETQuant development lessons."""
from __future__ import annotations

from numbers import Number
from typing import Any

import pandas as pd


def extract_numeric_value(value: Any) -> float:
    """Extract a numeric scalar from raw/BigQuery-like values.

    Historical PETQuant exports sometimes represented dose as a nested mapping
    such as ``{"float": 9.55, "integer": None, "provided": "float"}``.
    Converting that mapping directly with ``pd.to_numeric(..., errors="coerce")``
    silently loses the dose. This helper preserves the actual value.
    """
    if value is None:
        return float("nan")
    if isinstance(value, Number) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("float", "integer", "value"):
            candidate = value.get(key)
            if isinstance(candidate, Number) and not isinstance(candidate, bool):
                return float(candidate)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_pet_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a bounded public PETQuant feature frame.

    Missing values remain explicit. The function does not silently median-fill
    a column when every value is missing because that creates a meaningless
    feature while hiding the underlying data-quality problem.
    """
    required = ["glucose_mg_dl", "uptake_time_min", "fdg_dose_mci", "diabetic"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = df.copy()
    out["fdg_dose_mci"] = out["fdg_dose_mci"].map(extract_numeric_value)
    out["glucose_mg_dl"] = pd.to_numeric(out["glucose_mg_dl"], errors="coerce")
    out["uptake_time_min"] = pd.to_numeric(out["uptake_time_min"], errors="coerce")
    out["diabetic"] = out["diabetic"].astype("boolean")

    for optional in ("weight_kg", "expected_dose_mci"):
        if optional in out.columns:
            if optional == "expected_dose_mci":
                out[optional] = out[optional].map(extract_numeric_value)
            else:
                out[optional] = pd.to_numeric(out[optional], errors="coerce")
            out[f"{optional}_missing"] = out[optional].isna().astype("int8")

    return out

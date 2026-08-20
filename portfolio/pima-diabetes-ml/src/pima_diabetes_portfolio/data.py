"""Dataset normalization for OpenML dataset 37 (Pima Indians Diabetes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURES = ("preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age")
ZERO_AS_MISSING = ("plas", "pres", "skin", "insu", "mass")
TARGET = "class"


def prepare_pima_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate OpenML-style columns and normalize the binary target.

    Zero-valued clinical measurements are not changed here; that happens
    inside the model pipeline so imputation is fit on training data only.
    """
    required = set(FEATURES) | {TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    X = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    raw_y = frame[TARGET]
    if pd.api.types.is_numeric_dtype(raw_y):
        y = pd.to_numeric(raw_y, errors="raise").astype(int)
    else:
        mapping = {
            "tested_negative": 0,
            "tested_positive": 1,
            "negative": 0,
            "positive": 1,
            "0": 0,
            "1": 1,
        }
        y = raw_y.astype(str).str.strip().str.lower().map(mapping)
        if y.isna().any():
            bad = sorted(raw_y[y.isna()].astype(str).unique().tolist())
            raise ValueError(f"unrecognized target labels: {bad}")
        y = y.astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("target must be binary")
    return X, y.rename("outcome")


class ZeroAsMissingTransformer(BaseEstimator, TransformerMixin):
    """Replace physiologically implausible zero measurements with NaN.

    Pregnancies and age are intentionally not transformed.
    """

    def __init__(self, columns=ZERO_AS_MISSING):
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZeroAsMissingTransformer expects a pandas DataFrame")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"missing zero-as-missing columns: {missing}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZeroAsMissingTransformer expects a pandas DataFrame")
        out = X.copy()
        for col in self.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace(0, np.nan)
        return out

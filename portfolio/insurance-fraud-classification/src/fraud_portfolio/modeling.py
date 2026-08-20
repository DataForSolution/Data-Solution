"""Leakage-safe Logistic Regression pipelines for mixed tabular claims data."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic_pipeline(
    frame: pd.DataFrame,
    *,
    C: float = 0.01,
    class_weight=None,
) -> Pipeline:
    """Build preprocessing from declared training-frame dtypes only."""
    if frame.empty:
        raise ValueError("training frame is empty")
    categorical = frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in frame.columns if c not in categorical]
    if not categorical and not numeric:
        raise ValueError("no predictor columns")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        C=float(C),
        class_weight=class_weight,
        solver="liblinear",
        max_iter=2000,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def candidate_configurations():
    """Small declared model-search space used on validation average precision."""
    return [
        {"C": 0.01, "class_weight": None},
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 0.01, "class_weight": "balanced"},
        {"C": 0.1, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 10.0, "class_weight": "balanced"},
    ]

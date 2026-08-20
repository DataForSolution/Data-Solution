"""Stratified out-of-fold evaluation for binary review sentiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import validate_text_labels
from .modeling import build_model_pipeline, model_names


def classification_metrics(y_true, y_pred) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError("aligned non-empty 1-D labels are required")
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision_positive": float(precision_score(y, p, zero_division=0)),
        "recall_positive": float(recall_score(y, p, zero_division=0)),
        "f1_positive": float(f1_score(y, p, zero_division=0)),
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def evaluate_models_cv(texts, labels, *, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    text, y = validate_text_labels(texts, labels)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    counts = y.value_counts()
    if counts.min() < n_splits:
        raise ValueError("each class must have at least n_splits examples")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name in model_names():
        pipeline = build_model_pipeline(name)
        pred = cross_val_predict(pipeline, text, y, cv=cv, method="predict")
        row = {"model": name, **classification_metrics(y, pred)}
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["f1_macro", "balanced_accuracy", "model"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

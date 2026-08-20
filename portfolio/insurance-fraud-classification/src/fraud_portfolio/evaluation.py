"""Binary metrics that keep the minority fraud class visible."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, scores, *, threshold: float = 0.5) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape or len(y) == 0:
        raise ValueError("y_true and scores must be aligned non-empty 1-D arrays")
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("binary y_true values 0/1 are required")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    pred = (s >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else None
    roc_auc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None
    pr_auc = float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": None if specificity is None else float(specificity),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

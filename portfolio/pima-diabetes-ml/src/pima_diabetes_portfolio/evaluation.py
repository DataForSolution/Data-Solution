"""Binary-classification metrics with healthcare-relevant denominators."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def classification_report_dict(y_true, y_pred, *, positive_scores=None) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and y_pred must be aligned non-empty 1-D arrays")
    labels = set(np.unique(y).tolist()) | set(np.unique(p).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("binary labels 0/1 are required")

    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    auc = None
    if positive_scores is not None and len(np.unique(y)) == 2:
        scores = np.asarray(positive_scores, dtype=float)
        if scores.shape != y.shape or not np.isfinite(scores).all():
            raise ValueError("positive_scores must be finite and aligned")
        auc = float(roc_auc_score(y, scores))

    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "sensitivity": None if sensitivity is None else float(sensitivity),
        "specificity": None if specificity is None else float(specificity),
        "precision_positive": float(precision_score(y, p, zero_division=0)),
        "f1_positive": float(f1_score(y, p, zero_division=0)),
        "roc_auc": auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

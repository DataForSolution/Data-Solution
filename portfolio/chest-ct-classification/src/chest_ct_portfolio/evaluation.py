"""Deterministic classification metrics for the chest CT experiment."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> dict[str, object]:
    """Compute compact metrics after validating prediction/label alignment."""
    truth = list(y_true)
    pred = list(y_pred)
    if not truth:
        raise ValueError("y_true must not be empty")
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")

    resolved_labels = list(labels) if labels is not None else sorted(set(truth) | set(pred))
    if not resolved_labels:
        raise ValueError("labels must not be empty")

    return {
        "n": len(truth),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=resolved_labels, average="macro", zero_division=0)),
        "labels": resolved_labels,
        "confusion_matrix": confusion_matrix(truth, pred, labels=resolved_labels).tolist(),
    }

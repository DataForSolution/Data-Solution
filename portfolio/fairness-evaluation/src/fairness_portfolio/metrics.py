"""Binary-classification fairness metrics with explicit group/denominator semantics."""
from __future__ import annotations

import numpy as np


def _as_binary(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0, 1, False, True}):
        raise ValueError(f"{name} must contain only binary values")
    return arr.astype(int)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def balanced_accuracy(y_true, y_pred) -> float | None:
    """Mean of TPR and TNR, or None when either class is absent."""
    y = _as_binary(y_true, "y_true")
    p = _as_binary(y_pred, "y_pred")
    if len(y) != len(p):
        raise ValueError("y_true and y_pred must have the same length")
    positives = y == 1
    negatives = y == 0
    tpr = _safe_rate(int(np.logical_and(positives, p == 1).sum()), int(positives.sum()))
    tnr = _safe_rate(int(np.logical_and(negatives, p == 0).sum()), int(negatives.sum()))
    return None if tpr is None or tnr is None else 0.5 * (tpr + tnr)


def _group_rates(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    n = int(mask.sum())
    selected = int(np.logical_and(mask, p == 1).sum())
    positives = np.logical_and(mask, y == 1)
    negatives = np.logical_and(mask, y == 0)
    tp = int(np.logical_and(positives, p == 1).sum())
    fp = int(np.logical_and(negatives, p == 1).sum())
    return {
        "n": n,
        "selection_rate": _safe_rate(selected, n),
        "true_positive_rate": _safe_rate(tp, int(positives.sum())),
        "false_positive_rate": _safe_rate(fp, int(negatives.sum())),
    }


def _difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def fairness_report(
    y_true,
    y_pred,
    protected,
    *,
    privileged_value=1,
    unprivileged_value=0,
) -> dict[str, object]:
    """Compute AIF360-style group fairness metrics on one fixed label vector.

    Differences are unprivileged minus privileged. Disparate impact is the
    unprivileged selection rate divided by the privileged selection rate.
    Undefined denominators fail closed to None.
    """
    y = _as_binary(y_true, "y_true")
    p = _as_binary(y_pred, "y_pred")
    g = np.asarray(protected)
    if g.ndim != 1 or len(g) != len(y) or len(p) != len(y):
        raise ValueError("y_true, y_pred, and protected must be aligned 1-D arrays")

    priv_mask = g == privileged_value
    unpriv_mask = g == unprivileged_value
    if not priv_mask.any() or not unpriv_mask.any():
        raise ValueError("both privileged and unprivileged groups must be present")

    priv = _group_rates(y, p, priv_mask)
    unpriv = _group_rates(y, p, unpriv_mask)

    spd = _difference(unpriv["selection_rate"], priv["selection_rate"])
    eod = _difference(unpriv["true_positive_rate"], priv["true_positive_rate"])
    fpr_diff = _difference(unpriv["false_positive_rate"], priv["false_positive_rate"])
    aod = None if eod is None or fpr_diff is None else 0.5 * (eod + fpr_diff)
    di = None
    if priv["selection_rate"] not in (None, 0):
        di = unpriv["selection_rate"] / priv["selection_rate"]

    return {
        "balanced_accuracy": balanced_accuracy(y, p),
        "statistical_parity_difference": spd,
        "disparate_impact": di,
        "equal_opportunity_difference": eod,
        "average_odds_difference": aod,
        "privileged": priv,
        "unprivileged": unpriv,
    }

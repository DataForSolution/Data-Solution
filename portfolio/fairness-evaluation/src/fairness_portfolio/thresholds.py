"""Threshold evaluation on fixed reference labels and protected groups."""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd

from .metrics import fairness_report


def threshold_curve(
    scores,
    y_true,
    protected,
    *,
    thresholds: Iterable[float] | None = None,
    privileged_value=1,
    unprivileged_value=0,
) -> pd.DataFrame:
    """Evaluate utility/fairness across thresholds without changing ground truth."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError("scores must be a finite one-dimensional array")
    y = np.asarray(y_true)
    g = np.asarray(protected)
    if len(s) != len(y) or len(s) != len(g):
        raise ValueError("scores, y_true, and protected must have the same length")
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for threshold in thresholds:
        t = float(threshold)
        if not np.isfinite(t):
            raise ValueError("thresholds must be finite")
        pred = (s > t).astype(int)
        report = fairness_report(
            y,
            pred,
            g,
            privileged_value=privileged_value,
            unprivileged_value=unprivileged_value,
        )
        rows.append(
            {
                "threshold": t,
                "balanced_accuracy": report["balanced_accuracy"],
                "statistical_parity_difference": report["statistical_parity_difference"],
                "disparate_impact": report["disparate_impact"],
                "equal_opportunity_difference": report["equal_opportunity_difference"],
                "average_odds_difference": report["average_odds_difference"],
            }
        )
    return pd.DataFrame(rows)


def best_balanced_accuracy_threshold(curve: pd.DataFrame) -> float:
    """Return the lowest threshold tied for maximum defined balanced accuracy."""
    required = {"threshold", "balanced_accuracy"}
    if not required.issubset(curve.columns) or curve.empty:
        raise ValueError("curve must contain threshold and balanced_accuracy rows")
    valid = curve.dropna(subset=["balanced_accuracy"])
    if valid.empty:
        raise ValueError("no threshold has defined balanced accuracy")
    best = valid[valid["balanced_accuracy"] == valid["balanced_accuracy"].max()]
    return float(best["threshold"].min())

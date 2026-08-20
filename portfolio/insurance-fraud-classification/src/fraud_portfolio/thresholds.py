"""Threshold evaluation on validation data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .evaluation import binary_metrics


def threshold_curve(y_true, scores, *, quantiles: int = 501) -> pd.DataFrame:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or len(s) == 0 or not np.isfinite(s).all():
        raise ValueError("scores must be a finite non-empty 1-D array")
    if quantiles < 3:
        raise ValueError("quantiles must be at least 3")
    thresholds = np.unique(np.quantile(s, np.linspace(0.0, 1.0, quantiles)))
    return pd.DataFrame([binary_metrics(y_true, s, threshold=float(t)) for t in thresholds])


def select_threshold(curve: pd.DataFrame, *, objective: str = "f1") -> float:
    """Select the highest objective, breaking ties by balanced accuracy then higher threshold."""
    if objective not in {"f1", "balanced_accuracy"}:
        raise ValueError("objective must be 'f1' or 'balanced_accuracy'")
    required = {objective, "balanced_accuracy", "threshold"}
    missing = required - set(curve.columns)
    if missing or curve.empty:
        raise ValueError(f"invalid threshold curve; missing={sorted(missing)}")
    ranked = curve.sort_values(
        [objective, "balanced_accuracy", "threshold"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["threshold"])

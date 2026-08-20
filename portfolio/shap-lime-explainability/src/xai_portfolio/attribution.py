"""Attribution helpers that keep SHAP and LIME class alignment explicit."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def bounded_background(
    X_train: pd.DataFrame, *, max_rows: int = 40, random_state: int = 42
) -> pd.DataFrame:
    """Return a deterministic bounded background sample."""
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if len(X_train) <= max_rows:
        return X_train.copy()
    return X_train.sample(n=max_rows, random_state=random_state)


def mean_abs_shap(
    values: Any,
    feature_names: Sequence[str],
    *,
    class_index: int,
) -> pd.Series:
    """Aggregate multi-class SHAP values for one explicit output class.

    Expected shape: (samples, features, classes).
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError("SHAP values must have shape (samples, features, classes)")
    if array.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match SHAP feature dimension")
    if class_index < 0 or class_index >= array.shape[2]:
        raise IndexError("class_index is out of range")

    importance = np.mean(np.abs(array[:, :, class_index]), axis=0)
    return pd.Series(importance, index=list(feature_names), name="mean_abs_shap").sort_values(
        ascending=False
    )


def lime_weights_by_feature(
    explanation: Any,
    feature_names: Sequence[str],
    *,
    class_index: int,
) -> pd.Series:
    """Read one LIME explanation from feature indices, not condition strings."""
    mapping = explanation.as_map()
    if class_index not in mapping:
        raise KeyError(f"LIME explanation does not contain class index {class_index}")

    weights = np.zeros(len(feature_names), dtype=float)
    for feature_index, weight in mapping[class_index]:
        if feature_index < 0 or feature_index >= len(feature_names):
            raise IndexError(f"invalid LIME feature index {feature_index}")
        weights[int(feature_index)] = float(weight)

    return pd.Series(weights, index=list(feature_names), name="lime_weight")


def mean_abs_lime(
    explanations: Iterable[Any],
    feature_names: Sequence[str],
    *,
    class_index: int,
) -> pd.Series:
    """Aggregate LIME local weights for one explicit class."""
    rows = [
        lime_weights_by_feature(exp, feature_names, class_index=class_index)
        for exp in explanations
    ]
    if not rows:
        raise ValueError("at least one LIME explanation is required")
    frame = pd.DataFrame(rows)
    return frame.abs().mean(axis=0).sort_values(ascending=False).rename("mean_abs_lime")


def top_k_overlap(
    left: Mapping[str, float] | pd.Series,
    right: Mapping[str, float] | pd.Series,
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Compare the top-k feature sets from two attribution rankings."""
    if k < 1:
        raise ValueError("k must be at least 1")

    def ranked_names(values: Mapping[str, float] | pd.Series) -> list[str]:
        items = list(values.items())
        return [name for name, _ in sorted(items, key=lambda item: abs(float(item[1])), reverse=True)]

    left_top = ranked_names(left)[:k]
    right_top = ranked_names(right)[:k]
    shared = [name for name in left_top if name in set(right_top)]
    union = set(left_top) | set(right_top)
    return {
        "left_top": left_top,
        "right_top": right_top,
        "shared": shared,
        "overlap_count": len(shared),
        "jaccard": len(set(shared)) / len(union) if union else 1.0,
    }


def shap_permutation_explanation(
    model: Any,
    background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> Any:
    """Run current SHAP permutation explanation lazily.

    SHAP is optional so the core package remains lightweight.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError('Install the optional XAI dependencies with `pip install -e ".[xai]"`') from exc

    explainer = shap.Explainer(model.predict_proba, background, algorithm="permutation")
    max_evals = 2 * X_explain.shape[1] + 1
    return explainer(X_explain, max_evals=max_evals)

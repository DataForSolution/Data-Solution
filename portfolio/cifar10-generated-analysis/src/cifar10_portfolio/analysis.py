"""Classifier-proxy analysis for batches of generated images."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd


def _validate_probabilities(probabilities, *, atol: float = 1e-5) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes>=2)")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain non-finite values")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be within [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=atol):
        raise ValueError("each probability row must sum to 1")
    return probs


def _validate_class_names(class_names: Sequence[str], n_classes: int) -> list[str]:
    names = [str(x) for x in class_names]
    if len(names) != n_classes or len(set(names)) != len(names):
        raise ValueError("class_names must be unique and match probability columns")
    return names


def prediction_table(probabilities, class_names: Sequence[str]) -> pd.DataFrame:
    """Return deterministic classifier-response diagnostics for generated samples."""
    probs = _validate_probabilities(probabilities)
    names = _validate_class_names(class_names, probs.shape[1])

    pred_idx = probs.argmax(axis=1)
    confidence = probs[np.arange(len(probs)), pred_idx]
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    safe = np.clip(probs, 1e-15, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1) / np.log(probs.shape[1])

    return pd.DataFrame(
        {
            "sample_index": np.arange(len(probs), dtype=int),
            "predicted_class_index": pred_idx.astype(int),
            "predicted_class": [names[i] for i in pred_idx],
            "classifier_confidence": confidence,
            "normalized_entropy": entropy,
            "top2_margin": margin,
        }
    )


def top_confidence_indices(probabilities, *, n: int = 5) -> np.ndarray:
    """Indices with highest classifier confidence; this is not a realism ranking."""
    probs = _validate_probabilities(probabilities)
    if n < 1 or n > len(probs):
        raise ValueError("n must be between 1 and n_samples")
    confidence = probs.max(axis=1)
    return np.lexsort((np.arange(len(probs)), -confidence))[:n]


def class_confidence_summary(probabilities, class_names: Sequence[str]) -> pd.DataFrame:
    """Summarize confidence among samples whose predicted class is each class."""
    probs = _validate_probabilities(probabilities)
    names = _validate_class_names(class_names, probs.shape[1])
    table = prediction_table(probs, names)

    rows = []
    for idx, name in enumerate(names):
        values = table.loc[
            table["predicted_class_index"] == idx, "classifier_confidence"
        ].to_numpy()
        rows.append(
            {
                "class_index": idx,
                "class_name": name,
                "count": int(len(values)),
                "min_confidence": float(values.min()) if len(values) else np.nan,
                "mean_confidence": float(values.mean()) if len(values) else np.nan,
                "max_confidence": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def class_coverage_summary(probabilities, class_names: Sequence[str]) -> dict[str, object]:
    """Summarize predicted-class coverage; coverage is not generative fidelity."""
    probs = _validate_probabilities(probabilities)
    names = _validate_class_names(class_names, probs.shape[1])
    pred = probs.argmax(axis=1)
    counts = np.bincount(pred, minlength=len(names))
    distribution = counts / counts.sum()
    nonzero = distribution[distribution > 0]
    distribution_entropy = float(
        -(nonzero * np.log(nonzero)).sum() / np.log(len(names))
    ) if len(names) > 1 else 0.0
    return {
        "n_samples": int(len(probs)),
        "classes_covered": int((counts > 0).sum()),
        "coverage_rate": float((counts > 0).mean()),
        "predicted_class_counts": dict(zip(names, counts.astype(int).tolist())),
        "predicted_distribution_entropy": distribution_entropy,
    }

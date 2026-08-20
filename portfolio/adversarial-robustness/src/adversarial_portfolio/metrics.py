"""Robustness metrics with explicit denominators."""
from __future__ import annotations

import numpy as np


def _validate_scores(scores: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must have shape (n_samples, n_classes>=2)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_labels(labels, n: int) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError("true_labels must be one-dimensional and match score rows")
    return arr


def classification_summary(
    clean_scores: np.ndarray,
    adversarial_scores: np.ndarray,
    true_labels,
) -> dict[str, float | int | None]:
    """Summarize attack effect with attack success conditioned on clean correctness."""
    clean = _validate_scores(clean_scores, "clean_scores")
    adv = _validate_scores(adversarial_scores, "adversarial_scores")
    if clean.shape != adv.shape:
        raise ValueError("clean_scores and adversarial_scores must have the same shape")

    y = _validate_labels(true_labels, len(clean))
    clean_pred = clean.argmax(axis=1)
    adv_pred = adv.argmax(axis=1)
    clean_correct = clean_pred == y
    adv_correct = adv_pred == y

    eligible = int(clean_correct.sum())
    successful_attacks = int(np.logical_and(clean_correct, ~adv_correct).sum())

    return {
        "n_samples": int(len(y)),
        "clean_accuracy": float(clean_correct.mean()),
        "adversarial_accuracy": float(adv_correct.mean()),
        "robust_accuracy": float(adv_correct.mean()),
        "attack_eligible": eligible,
        "attack_success_count": successful_attacks,
        "attack_success_rate": successful_attacks / eligible if eligible else None,
    }


def defence_summary(
    clean_scores: np.ndarray,
    adversarial_scores: np.ndarray,
    defended_clean_scores: np.ndarray,
    defended_adversarial_scores: np.ndarray,
    true_labels,
) -> dict[str, float | int | None]:
    """Measure robustness gain and clean-utility cost for a defence."""
    base = classification_summary(clean_scores, adversarial_scores, true_labels)
    defended = classification_summary(
        defended_clean_scores, defended_adversarial_scores, true_labels
    )

    y = np.asarray(true_labels)
    clean_pred = np.asarray(clean_scores).argmax(axis=1)
    adv_pred = np.asarray(adversarial_scores).argmax(axis=1)
    def_clean_pred = np.asarray(defended_clean_scores).argmax(axis=1)
    def_adv_pred = np.asarray(defended_adversarial_scores).argmax(axis=1)

    clean_before_correct = clean_pred == y
    attacked = np.logical_and(clean_before_correct, adv_pred != y)
    recovered = np.logical_and(attacked, def_adv_pred == y)
    retained = np.logical_and(clean_before_correct, def_clean_pred == y)

    attack_count = int(attacked.sum())
    clean_correct_count = int(clean_before_correct.sum())

    return {
        **base,
        "defended_clean_accuracy": defended["clean_accuracy"],
        "defended_adversarial_accuracy": defended["adversarial_accuracy"],
        "robust_accuracy_gain": float(
            defended["adversarial_accuracy"] - base["adversarial_accuracy"]
        ),
        "clean_accuracy_delta": float(
            defended["clean_accuracy"] - base["clean_accuracy"]
        ),
        "recovered_attack_count": int(recovered.sum()),
        "adversarial_recovery_rate": (
            int(recovered.sum()) / attack_count if attack_count else None
        ),
        "clean_retention_rate": (
            int(retained.sum()) / clean_correct_count if clean_correct_count else None
        ),
    }

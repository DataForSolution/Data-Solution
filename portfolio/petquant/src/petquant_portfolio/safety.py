"""Safety-oriented evaluation helpers for PETQuant portfolio examples.

These functions are intentionally model-agnostic. They support transparent
measurement of false reassurance and a conservative score gate. They are not
clinical decision rules and must not be used for patient care.
"""
from __future__ import annotations

from collections.abc import Iterable


def false_reliable_rate(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    unreliable_label: str = "unreliable",
    reliable_label: str = "reliable",
) -> dict[str, float | int | None]:
    """Return the false-reliable count/rate among reference-unreliable cases.

    A false-reliable case is a reference label of ``unreliable`` paired with a
    model prediction of ``reliable``. The function fails closed when there are
    no unreliable reference cases by returning ``None`` for the rate rather
    than reporting an artificial 0% rate.
    """
    truth = list(y_true)
    pred = list(y_pred)
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")

    unreliable_total = sum(v == unreliable_label for v in truth)
    false_reliable = sum(
        t == unreliable_label and p == reliable_label
        for t, p in zip(truth, pred, strict=True)
    )

    return {
        "unreliable_total": unreliable_total,
        "false_reliable_count": false_reliable,
        "false_reliable_rate": (
            false_reliable / unreliable_total if unreliable_total else None
        ),
    }


def apply_reliable_score_gate(
    labels: Iterable[str],
    scores: Iterable[float],
    *,
    min_reliable_score: float = 90.0,
    reliable_label: str = "reliable",
    fallback_label: str = "caution",
) -> list[str]:
    """Downgrade low-confidence ``reliable`` predictions to ``caution``.

    This reproduces the transparent historical safety-gate concept used during
    PETQuant development. It is included for engineering demonstration only;
    the threshold is not clinically validated.
    """
    labels_list = list(labels)
    scores_list = list(scores)
    if len(labels_list) != len(scores_list):
        raise ValueError("labels and scores must have the same length")

    calibrated: list[str] = []
    for label, score in zip(labels_list, scores_list, strict=True):
        if label == reliable_label and float(score) < min_reliable_score:
            calibrated.append(fallback_label)
        else:
            calibrated.append(label)
    return calibrated

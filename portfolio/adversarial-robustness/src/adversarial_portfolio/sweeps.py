"""Defence sweep harness that keeps preprocessing/prediction contracts consistent."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from .metrics import defence_summary


def run_defence_sweep(
    *,
    parameters: Iterable[Any],
    clean_inputs,
    adversarial_inputs,
    true_labels,
    predictor: Callable[[Any], Any],
    transform_factory: Callable[[Any], Callable[[Any], Any]],
    parameter_name: str = "parameter",
) -> pd.DataFrame:
    """Evaluate the same transform parameter on clean and adversarial inputs.

    `predictor` owns *all* model preprocessing. This prevents a defence sweep from
    accidentally bypassing the model's normal input contract.
    """
    clean_scores = predictor(clean_inputs)
    adversarial_scores = predictor(adversarial_inputs)

    rows: list[dict[str, Any]] = []
    for parameter in parameters:
        transform = transform_factory(parameter)
        defended_clean = transform(clean_inputs)
        defended_adv = transform(adversarial_inputs)
        metrics = defence_summary(
            clean_scores,
            adversarial_scores,
            predictor(defended_clean),
            predictor(defended_adv),
            true_labels,
        )
        rows.append({parameter_name: parameter, **metrics})

    return pd.DataFrame(rows)

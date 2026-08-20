import numpy as np
import pandas as pd
import pytest

from xai_portfolio.attribution import (
    bounded_background,
    lime_weights_by_feature,
    mean_abs_lime,
    mean_abs_shap,
    top_k_overlap,
)


class FakeLimeExplanation:
    def __init__(self, mapping):
        self._mapping = mapping

    def as_map(self):
        return self._mapping


def test_bounded_background_is_deterministic():
    frame = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    first = bounded_background(frame, max_rows=10, random_state=7)
    second = bounded_background(frame, max_rows=10, random_state=7)
    assert len(first) == 10
    assert first.index.tolist() == second.index.tolist()


def test_mean_abs_shap_is_class_specific():
    values = np.array(
        [
            [[1, 10], [2, 20], [3, 30]],
            [[3, 30], [2, 20], [1, 10]],
        ],
        dtype=float,
    )
    result = mean_abs_shap(values, ["a", "b", "c"], class_index=0)
    assert result.to_dict() == {"a": 2.0, "b": 2.0, "c": 2.0}

    with pytest.raises(IndexError):
        mean_abs_shap(values, ["a", "b", "c"], class_index=2)


def test_lime_uses_feature_indices_not_condition_strings():
    explanation = FakeLimeExplanation({1: [(2, -0.4), (0, 0.2)]})
    weights = lime_weights_by_feature(
        explanation, ["alcohol", "malic_acid", "proline"], class_index=1
    )
    assert weights.to_dict() == {
        "alcohol": 0.2,
        "malic_acid": 0.0,
        "proline": -0.4,
    }


def test_mean_abs_lime_and_overlap():
    explanations = [
        FakeLimeExplanation({0: [(0, 0.4), (1, -0.2)]}),
        FakeLimeExplanation({0: [(0, -0.2), (1, 0.6)]}),
    ]
    lime = mean_abs_lime(explanations, ["a", "b"], class_index=0)
    assert lime["b"] == pytest.approx(0.4)
    assert lime["a"] == pytest.approx(0.3)

    overlap = top_k_overlap({"a": 5, "b": 4, "c": 1}, {"b": 5, "a": 2, "d": 1}, k=2)
    assert overlap["overlap_count"] == 2
    assert overlap["jaccard"] == 1.0

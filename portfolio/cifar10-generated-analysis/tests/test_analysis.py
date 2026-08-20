import numpy as np
import pytest

from cifar10_portfolio.analysis import (
    prediction_table,
    class_confidence_summary,
    top_confidence_indices,
    class_coverage_summary,
)

NAMES = ["airplane", "automobile", "bird"]
PROBS = np.array([
    [0.80, 0.10, 0.10],
    [0.20, 0.70, 0.10],
    [0.10, 0.15, 0.75],
    [0.60, 0.20, 0.20],
])


def test_prediction_table_has_stable_class_confidence_entropy_and_margin():
    table = prediction_table(PROBS, NAMES)
    assert table["predicted_class"].tolist() == ["airplane", "automobile", "bird", "airplane"]
    assert table.loc[0, "classifier_confidence"] == pytest.approx(0.8)
    assert table.loc[0, "top2_margin"] == pytest.approx(0.7)
    assert table["normalized_entropy"].between(0, 1).all()


def test_random_scores_that_are_not_probabilities_are_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        prediction_table(np.array([[0.8, 0.8, 0.8]]), NAMES)


def test_top_confidence_is_deterministic_and_not_based_on_class_name():
    assert top_confidence_indices(PROBS, n=2).tolist() == [0, 2]


def test_class_summary_handles_zero_predicted_samples():
    probs = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]])
    summary = class_confidence_summary(probs, NAMES)
    bird = summary.loc[summary["class_name"] == "bird"].iloc[0]
    assert bird["count"] == 0
    assert np.isnan(bird["mean_confidence"])


def test_class_coverage_uses_predicted_classes_only():
    result = class_coverage_summary(PROBS, NAMES)
    assert result["classes_covered"] == 3
    assert result["coverage_rate"] == pytest.approx(1.0)
    assert result["predicted_class_counts"] == {"airplane": 2, "automobile": 1, "bird": 1}

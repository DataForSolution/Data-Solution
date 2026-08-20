import pytest

from chest_ct_portfolio.evaluation import classification_metrics


def test_classification_metrics():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
    assert result["n"] == 4
    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert 0 <= result["balanced_accuracy"] <= 1
    assert 0 <= result["macro_f1"] <= 1


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0])


def test_empty_truth_rejected():
    with pytest.raises(ValueError):
        classification_metrics([], [])

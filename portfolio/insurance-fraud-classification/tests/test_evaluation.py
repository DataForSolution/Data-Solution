import pytest
from fraud_portfolio.evaluation import binary_metrics


def test_binary_metrics_keep_positive_class_visible():
    r = binary_metrics([0,0,1,1], [0.1,0.8,0.9,0.2], threshold=0.5)
    assert r["true_positive"] == 1
    assert r["false_positive"] == 1
    assert r["recall"] == 0.5
    assert r["specificity"] == 0.5
    assert r["roc_auc"] is not None
    assert r["pr_auc"] is not None


def test_bad_alignment_rejected():
    with pytest.raises(ValueError):
        binary_metrics([0,1], [0.2])

import pytest

from pima_diabetes_portfolio.evaluation import classification_report_dict


def test_metrics_have_explicit_sensitivity_specificity():
    r = classification_report_dict(
        [0,0,1,1], [0,1,1,1], positive_scores=[0.1,0.7,0.8,0.9]
    )
    assert r["sensitivity"] == pytest.approx(1.0)
    assert r["specificity"] == pytest.approx(0.5)
    assert r["balanced_accuracy"] == pytest.approx(0.75)
    assert r["roc_auc"] == pytest.approx(1.0)


def test_binary_label_contract():
    with pytest.raises(ValueError, match="binary"):
        classification_report_dict([0,2], [0,1])

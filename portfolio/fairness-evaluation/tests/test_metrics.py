import pytest

from fairness_portfolio.metrics import fairness_report, balanced_accuracy


def test_balanced_accuracy_known_example():
    assert balanced_accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.75)


def test_group_fairness_metrics_known_example():
    # privileged first four, unprivileged last four
    y = [1, 1, 0, 0, 1, 1, 0, 0]
    p = [1, 1, 1, 0, 1, 0, 0, 0]
    g = [1, 1, 1, 1, 0, 0, 0, 0]
    r = fairness_report(y, p, g)
    assert r["privileged"]["selection_rate"] == pytest.approx(0.75)
    assert r["unprivileged"]["selection_rate"] == pytest.approx(0.25)
    assert r["statistical_parity_difference"] == pytest.approx(-0.5)
    assert r["disparate_impact"] == pytest.approx(1/3)
    assert r["equal_opportunity_difference"] == pytest.approx(-0.5)
    assert r["average_odds_difference"] == pytest.approx(-0.5)


def test_disparate_impact_fails_closed_on_zero_privileged_selection():
    r = fairness_report([1,0,1,0], [0,0,1,0], [1,1,0,0])
    assert r["disparate_impact"] is None


def test_missing_group_rejected():
    with pytest.raises(ValueError, match="both privileged"):
        fairness_report([1,0], [1,0], [1,1])

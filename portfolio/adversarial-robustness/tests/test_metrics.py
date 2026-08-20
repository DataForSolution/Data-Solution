import numpy as np
import pytest

from adversarial_portfolio.metrics import classification_summary, defence_summary


def test_attack_success_denominator_is_clean_correct_only():
    clean = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    adv = np.array([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2]])
    y = np.array([0, 1, 1])  # third sample is already wrong when clean
    result = classification_summary(clean, adv, y)
    assert result["clean_accuracy"] == pytest.approx(2 / 3)
    assert result["attack_eligible"] == 2
    assert result["attack_success_count"] == 1
    assert result["attack_success_rate"] == pytest.approx(0.5)


def test_zero_clean_correct_fails_closed_for_attack_success():
    clean = np.array([[0.1, 0.9]])
    adv = np.array([[0.2, 0.8]])
    result = classification_summary(clean, adv, [0])
    assert result["attack_eligible"] == 0
    assert result["attack_success_rate"] is None


def test_defence_reports_clean_cost_and_recovery():
    clean = np.array([[0.9, 0.1], [0.1, 0.9]])
    adv = np.array([[0.1, 0.9], [0.2, 0.8]])
    def_clean = np.array([[0.8, 0.2], [0.6, 0.4]])  # damages second clean case
    def_adv = np.array([[0.7, 0.3], [0.2, 0.8]])    # recovers first attack
    result = defence_summary(clean, adv, def_clean, def_adv, [0, 1])
    assert result["robust_accuracy_gain"] == pytest.approx(0.5)
    assert result["clean_accuracy_delta"] == pytest.approx(-0.5)
    assert result["adversarial_recovery_rate"] == pytest.approx(1.0)
    assert result["clean_retention_rate"] == pytest.approx(0.5)


def test_bad_score_shapes_rejected():
    with pytest.raises(ValueError):
        classification_summary(np.array([1, 2]), np.array([1, 2]), [0, 1])

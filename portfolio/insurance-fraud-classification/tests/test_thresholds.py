import pandas as pd
import pytest
from fraud_portfolio.thresholds import threshold_curve, select_threshold


def test_threshold_selection_is_deterministic():
    y = [0,0,0,1,1]
    scores = [0.1,0.2,0.4,0.6,0.9]
    curve = threshold_curve(y, scores, quantiles=11)
    t1 = select_threshold(curve, objective="f1")
    t2 = select_threshold(curve, objective="f1")
    assert t1 == t2


def test_invalid_objective_rejected():
    curve = pd.DataFrame({"threshold":[0.5], "f1":[1.0], "balanced_accuracy":[1.0]})
    with pytest.raises(ValueError):
        select_threshold(curve, objective="accuracy")

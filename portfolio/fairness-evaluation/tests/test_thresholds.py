import numpy as np
import pytest

from fairness_portfolio.thresholds import threshold_curve, best_balanced_accuracy_threshold


def test_threshold_curve_keeps_one_reference_label_vector():
    scores = [0.9, 0.8, 0.4, 0.1, 0.7, 0.6, 0.3, 0.2]
    y =      [1,   1,   0,   0,   1,   0,   1,   0]
    group =  [1,   1,   1,   1,   0,   0,   0,   0]
    curve = threshold_curve(scores, y, group, thresholds=[0.25, 0.5, 0.75])
    assert curve["threshold"].tolist() == [0.25, 0.5, 0.75]
    assert len(curve) == 3


def test_best_threshold_tie_breaks_to_lower_threshold():
    import pandas as pd
    curve = pd.DataFrame({"threshold":[0.2,0.3,0.4], "balanced_accuracy":[0.7,0.8,0.8]})
    assert best_balanced_accuracy_threshold(curve) == pytest.approx(0.3)


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError):
        threshold_curve([0.2, np.nan], [0,1], [0,1], thresholds=[0.5])

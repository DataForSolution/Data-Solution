import pytest
from sentiment_portfolio.evaluation import classification_metrics, evaluate_models_cv


TEXT = [
    "excellent meal", "wonderful service", "loved the dinner", "friendly staff",
    "terrible food", "awful service", "hated the meal", "rude staff",
    "great restaurant", "delicious dinner", "bad restaurant", "disgusting dinner",
]
Y = [1,1,1,1,0,0,0,0,1,1,0,0]


def test_metrics_keep_both_classes_visible():
    r = classification_metrics([0,0,1,1], [0,1,1,0])
    assert r["true_positive"] == 1
    assert r["true_negative"] == 1
    assert r["balanced_accuracy"] == 0.5


def test_cv_returns_one_row_per_declared_model():
    result = evaluate_models_cv(TEXT, Y, n_splits=3)
    assert set(result["model"]) == {"multinomial_nb", "logistic_regression", "linear_svm"}
    assert result["f1_macro"].between(0, 1).all()


def test_too_many_folds_rejected():
    with pytest.raises(ValueError):
        evaluate_models_cv(["good","bad"], [1,0], n_splits=2)

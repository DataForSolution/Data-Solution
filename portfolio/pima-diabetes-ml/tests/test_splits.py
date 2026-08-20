import pandas as pd

from pima_diabetes_portfolio.splits import stratified_train_validation_test_split


def test_split_is_disjoint_and_stratified():
    X = pd.DataFrame({"x": range(100), "z": range(100,200)})
    y = pd.Series([0,1] * 50)
    s = stratified_train_validation_test_split(X, y)
    assert (len(s.X_train), len(s.X_validation), len(s.X_test)) == (60,20,20)
    assert set(s.X_train.index).isdisjoint(s.X_validation.index)
    assert set(s.X_train.index).isdisjoint(s.X_test.index)
    assert set(s.X_validation.index).isdisjoint(s.X_test.index)
    assert s.y_train.mean() == 0.5
    assert s.y_validation.mean() == 0.5
    assert s.y_test.mean() == 0.5

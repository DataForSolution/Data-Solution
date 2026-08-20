from xai_portfolio.data import load_wine_frame, split_wine


def test_wine_contract():
    X, y, classes = load_wine_frame()
    assert X.shape == (178, 13)
    assert len(y) == 178
    assert classes == ("class_0", "class_1", "class_2")
    assert set(y.unique()) == set(classes)


def test_split_is_reproducible_and_stratified():
    first = split_wine()
    second = split_wine()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()
    assert first.y_test.value_counts().to_dict() == {
        "class_1": 14,
        "class_0": 12,
        "class_2": 10,
    }

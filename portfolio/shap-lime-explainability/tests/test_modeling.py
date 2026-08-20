from xai_portfolio.data import split_wine
from xai_portfolio.modeling import build_mlp, evaluate_classifier


def test_model_evaluation_uses_stable_labels():
    split = split_wine()
    model = build_mlp()
    model.fit(split.X_train, split.y_train)
    result = evaluate_classifier(model, split.X_test, split.y_test)

    assert result["labels"] == ["class_0", "class_1", "class_2"]
    assert result["accuracy"] >= 0.85
    assert result["balanced_accuracy"] >= 0.85
    assert result["macro_f1"] >= 0.85

    supports = {label: int(result["report"][label]["support"]) for label in result["labels"]}
    assert supports == {"class_0": 12, "class_1": 14, "class_2": 10}

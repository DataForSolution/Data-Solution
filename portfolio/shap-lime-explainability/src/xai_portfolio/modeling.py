"""Reproducible MLP training and evaluation."""
from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp(*, random_state: int = 42) -> Pipeline:
    """Build a scaled, deterministic small-sample MLP pipeline."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_classifier(model: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    """Evaluate using the model's own stable class ordering."""
    predicted = model.predict(X_test)
    labels = [str(value) for value in model.classes_]
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "labels": labels,
        "report": classification_report(
            y_test,
            predicted,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

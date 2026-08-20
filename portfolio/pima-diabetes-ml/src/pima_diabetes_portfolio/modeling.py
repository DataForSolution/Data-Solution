"""Leakage-safe SVM and MLP pipelines for retrospective comparison."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import ZeroAsMissingTransformer


def _steps(model):
    return [
        ("zero_as_missing", ZeroAsMissingTransformer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ]


def build_svm_pipeline(*, kernel: str = "linear", C: float = 1.0, random_state: int = 42):
    """Build an SVM pipeline whose preprocessing is learned from training data only."""
    if kernel not in {"linear", "rbf", "poly"}:
        raise ValueError("kernel must be one of: linear, rbf, poly")
    return Pipeline(
        _steps(
            SVC(
                kernel=kernel,
                C=C,
                probability=True,
                random_state=random_state,
            )
        )
    )


def build_mlp_pipeline(
    *,
    hidden_layer_sizes=(100,),
    activation: str = "relu",
    max_iter: int = 2000,
    solver: str = "adam",
    random_state: int = 42,
):
    """Build an MLP pipeline with train-only imputation and scaling."""
    return Pipeline(
        _steps(
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                max_iter=max_iter,
                solver=solver,
                random_state=random_state,
            )
        )
    )

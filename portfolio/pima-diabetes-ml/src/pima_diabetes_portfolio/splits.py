"""Deterministic, stratified train/validation/test splitting."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def stratified_train_validation_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Create disjoint stratified splits; sizes are fractions of full data."""
    if not (0 < test_size < 1 and 0 < validation_size < 1):
        raise ValueError("test_size and validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be < 1")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_fraction_of_trainval = validation_size / (1.0 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=random_state,
    )
    return DataSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )

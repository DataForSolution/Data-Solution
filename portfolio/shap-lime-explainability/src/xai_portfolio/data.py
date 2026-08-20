"""Deterministic data preparation for the Wine XAI example."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class WineSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    class_names: tuple[str, ...]


def load_wine_frame() -> tuple[pd.DataFrame, pd.Series, tuple[str, ...]]:
    """Return features, stable string labels, and class names."""
    raw = load_wine(as_frame=True)
    X = raw.data.copy()
    class_names = tuple(str(name) for name in raw.target_names)
    y = pd.Series(
        [class_names[int(index)] for index in raw.target],
        index=X.index,
        name="class",
        dtype="string",
    )
    return X, y, class_names


def split_wine(*, test_size: float = 0.20, random_state: int = 42) -> WineSplit:
    """Create a reproducible stratified split."""
    X, y, class_names = load_wine_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return WineSplit(X_train, X_test, y_train, y_test, class_names)

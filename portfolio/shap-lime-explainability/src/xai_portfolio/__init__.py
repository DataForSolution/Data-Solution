"""Utilities for the reconstructed SHAP/LIME portfolio project."""

from .attribution import (
    bounded_background,
    lime_weights_by_feature,
    mean_abs_lime,
    mean_abs_shap,
    top_k_overlap,
)
from .data import load_wine_frame, split_wine
from .modeling import build_mlp, evaluate_classifier

__all__ = [
    "bounded_background",
    "lime_weights_by_feature",
    "mean_abs_lime",
    "mean_abs_shap",
    "top_k_overlap",
    "load_wine_frame",
    "split_wine",
    "build_mlp",
    "evaluate_classifier",
]

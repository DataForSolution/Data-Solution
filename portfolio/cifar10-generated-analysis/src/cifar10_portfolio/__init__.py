"""Analysis helpers for generated CIFAR-10 image experiments."""

from .analysis import (
    prediction_table,
    class_confidence_summary,
    top_confidence_indices,
    class_coverage_summary,
)
from .images import rescale_to_unit_interval

__all__ = [
    "prediction_table",
    "class_confidence_summary",
    "top_confidence_indices",
    "class_coverage_summary",
    "rescale_to_unit_interval",
]

"""Image-range conversion helpers."""
from __future__ import annotations

import numpy as np


def rescale_to_unit_interval(images, *, source_min: float, source_max: float) -> np.ndarray:
    """Convert an explicitly declared source range to [0,1] exactly once."""
    if not np.isfinite([source_min, source_max]).all() or source_min >= source_max:
        raise ValueError("source_min must be finite and smaller than source_max")
    arr = np.asarray(images, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("images contain non-finite values")
    tol = 1e-6
    if arr.min() < source_min - tol or arr.max() > source_max + tol:
        raise ValueError("image values fall outside the declared source range")
    scaled = (arr - source_min) / (source_max - source_min)
    return np.clip(scaled, 0.0, 1.0)

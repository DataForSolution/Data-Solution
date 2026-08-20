import numpy as np
import pytest

from cifar10_portfolio.images import rescale_to_unit_interval


def test_rescale_minus_one_one_to_unit_interval():
    x = np.array([-1.0, 0.0, 1.0])
    out = rescale_to_unit_interval(x, source_min=-1.0, source_max=1.0)
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_already_unit_images_are_not_shifted_when_declared_unit_range():
    x = np.array([0.0, 0.25, 1.0])
    out = rescale_to_unit_interval(x, source_min=0.0, source_max=1.0)
    assert np.allclose(out, x)


def test_declared_range_mismatch_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        rescale_to_unit_interval(np.array([-0.1, 0.5]), source_min=0.0, source_max=1.0)

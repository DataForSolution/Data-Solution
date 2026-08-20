import pytest

from chest_ct_portfolio.modeling import build_resnet50_classifier


def test_invalid_class_count_rejected_before_tensorflow_is_needed():
    with pytest.raises(ValueError):
        build_resnet50_classifier(num_classes=1)

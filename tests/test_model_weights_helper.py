"""Tests for module model_weights_helper."""

import pytest

from orthoseg.model.model_weights_helper import (
    get_model_weights_path,
    get_weights_types_for_architecture,
)


@pytest.mark.parametrize(
    "architecture, weights_type",
    [
        ("mobilenetv2+linknet", "aerial"),
        ("mobilenetv2+linknet", "aerial-v1"),
        ("inceptionresnetv2+unet", "aerial"),
    ],
)
def test_get_model_weights_path(architecture: str, weights_type: str):
    weights_path = get_model_weights_path(
        architecture=architecture, weights_type=weights_type
    )
    assert weights_path is not None
    assert weights_path.exists()
    assert weights_path.is_file()
    assert weights_path.stat().st_size > 5 * 1024 * 1024  # > 5MB


def test_get_model_weights_path_localfile(tmp_path):
    """Test using a local weights file in the weights dir."""
    architecture = "mobilenetv2+linknet"
    weights_type = "custom-weights"
    local_weights_path = tmp_path / f"{architecture}_{weights_type}_notop.weights.h5"
    local_weights_path.touch()

    weights_path = get_model_weights_path(
        architecture=architecture, weights_type=weights_type, weights_dir=tmp_path
    )

    assert weights_path == local_weights_path


@pytest.mark.parametrize(
    "architecture, weights_type",
    [
        ("mobilenetv2+linknet", "invalid"),
        ("unknown+unet", "aerial"),
    ],
)
def test_get_model_weights_path_invalid_param(architecture, weights_type):
    with pytest.raises(ValueError, match="No weights available for"):
        _ = get_model_weights_path(architecture, weights_type)


def test_get_weights_types_for_architecture():
    assert get_weights_types_for_architecture("mobilenetv2+linknet") == ["aerial"]
    assert get_weights_types_for_architecture("inceptionresnetv2+unet") == ["aerial"]
    assert get_weights_types_for_architecture("unknown+unet") == []

"""
Tests for functionalities in orthoseg.model.model_factory.
"""
from typing import Optional

import pytest

from orthoseg.model import model_factory


@pytest.mark.parametrize(
    "architecture, input_width, input_height, expected_error",
    [
        ["test+linknet", 100, 100, "for decoder linknet"],
        ["test+linknet", 256, 256, None],
        ["test+pspnet", 144, 144, None],
        ["test+unet", 256, 256, None],
    ],
)
def test_check_image_size(
    architecture: str,
    input_width: int,
    input_height: int,
    expected_error: Optional[str],
):
    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            model_factory.check_image_size(
                architecture=architecture,
                input_width=input_width,
                input_height=input_height,
            )
    else:
        model_factory.check_image_size(
            architecture=architecture,
            input_width=input_width,
            input_height=input_height,
        )


@pytest.mark.parametrize(
    "architecture, input_width, input_height",
    [
        ["mobilenetv2+linknet", 256, 256],
        ["mobilenetv2+pspnet", 144, 144],
        ["mobilenetv2+unet", 256, 256],
    ],
)
def test_get_compile_save_load_model(
    tmp_path, architecture: str, input_width: int, input_height: int
):
    # Get model
    model = model_factory.get_model(
        architecture=architecture,
        input_width=input_width,
        input_height=input_height,
        nb_classes=5,
    )
    assert model is not None

    # Compile model
    model = model_factory.compile_model(
        model,
        optimizer="adam",
        optimizer_params={"learning_rate": 0.0001},
        loss="categorical_crossentropy",
    )
    assert model is not None

    # Now save model
    model_path = tmp_path / f"{architecture}.hdf5"
    model.save(str(model_path))
    model = None

    # Load model again
    model = model_factory.load_model(model_path)
    assert model is not None


@pytest.mark.parametrize("architecture", ["mobilenetv2+unknown"])
def test_get_model_unknown_decoder(architecture: str):
    with pytest.raises(ValueError, match="Unknown decoder architecture:"):
        _ = model_factory.get_model(
            architecture=architecture,
        )


@pytest.mark.parametrize("architecture", ["unknown+unet"])
def test_get_model_unknown_encoder(architecture: str):
    # Error is raised by segmentation_models library
    with pytest.raises(ValueError, match="No such model"):
        _ = model_factory.get_model(
            architecture=architecture,
        )

"""Tests for functionalities in orthoseg.model.model_factory."""

import os

import pytest

from orthoseg.model import model_factory as mf, model_helper as mh


@pytest.mark.parametrize("backend", [os.environ.get("KERAS_BACKEND", "tensorflow")])
def test_backend(backend):
    """The backend loaded should be the same as set in the environment variable."""
    import keras  # noqa: PLC0415

    assert keras.backend.backend() == backend


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
    expected_error: str | None,
):
    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            mf.check_image_size(
                architecture=architecture,
                input_width=input_width,
                input_height=input_height,
            )
    else:
        mf.check_image_size(
            architecture=architecture,
            input_width=input_width,
            input_height=input_height,
        )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
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
    model = mf.get_model(
        architecture=architecture,
        input_width=input_width,
        input_height=input_height,
        nb_classes=5,
    )
    assert model is not None

    # On windows, these give segmentation fault on github, so skip.
    if "GITHUB_ACTIONS" in os.environ and os.name == "nt":
        pytest.skip("crashes on github CI on windows")

    # Compile model
    model = mf.compile_model(
        model,
        optimizer="adam",
        optimizer_params={"learning_rate": 0.0001},
        loss="categorical_crossentropy",
        class_weights=None,
    )
    assert model is not None

    # Now save model + hyperparams.
    model_path = tmp_path / f"{architecture}.keras"
    model.save(str(model_path))
    model = None
    augmentations = {"rescale": 1 / 255.0, "fill_mode": "constant", "cval": 0}
    hyperparams = mh.HyperParams(
        architecture=mh.ArchitectureParams(architecture=architecture),
        train=mh.TrainParams(
            image_augmentations=augmentations, mask_augmentations=augmentations
        ),
    )
    hyperparams_filepath = tmp_path / f"{model_path.stem}_hyperparams.json"
    hyperparams_filepath.write_text(hyperparams.toJSON())

    # Load model again
    model = mf.load_model(model_path)
    assert model is not None


@pytest.mark.parametrize("architecture", ["mobilenetv2+unknown"])
def test_get_model_unknown_decoder(architecture: str):
    with pytest.raises(ValueError, match="Unknown decoder architecture:"):
        _ = mf.get_model(architecture=architecture)


@pytest.mark.parametrize("architecture", ["unknown+unet"])
def test_get_model_unknown_encoder(architecture: str):
    # Error is raised by segmentation_models library
    with pytest.raises(
        ValueError, match="Backbone with name 'unknown' is not supported"
    ):
        _ = mf.get_model(architecture=architecture)

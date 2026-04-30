"""Tests for functionalities in orthoseg.model.model_factory."""

import os

import pytest
import segmodels_keras as smk

from orthoseg._compat import KERAS_GTE_3
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
        ["inceptionresnetv2+unet", 256, 256],
    ],
)
def test_get_compile_save_load_model(
    tmp_path, architecture: str, input_width: int, input_height: int
):
    # Get model
    classes = ["a", "b", "c", "d", "e"]
    model, _model_preprocess_input = mf.get_model(
        architecture=architecture,
        input_width=input_width,
        input_height=input_height,
        nb_classes=len(classes),
        weights_type=None,
    )
    assert model is not None

    # On windows, these give segmentation fault on github, so skip.
    if "GITHUB_ACTIONS" in os.environ and os.name == "nt":
        pytest.skip("crashes on github CI on windows")

    # Compile model
    model = mf.compile_model(
        model,
        optimizer="AdamW",
        optimizer_params={"learning_rate": 0.0001},
        loss="categorical_crossentropy",
        class_weights=None,
    )
    assert model is not None

    # Now save model + hyperparams.
    model_path = tmp_path / "subject_01.keras"
    model.save(str(model_path))

    augmentations = {"rescale": 1 / 255.0, "fill_mode": "constant", "cval": 0}
    hyperparams = mh.HyperParams(
        architecture=mh.ArchitectureParams(architecture=architecture, classes=classes),
        train=mh.TrainParams(
            image_augmentations=augmentations, mask_augmentations=augmentations
        ),
    )
    hyperparams_filepath = tmp_path / f"{model_path.stem}_hyperparams.json"
    hyperparams_filepath.write_text(hyperparams.toJSON())

    # Load model again
    model, _model_preprocess_input = mf.load_model(model_path)
    assert model is not None

    # Load only model weights from the saved file
    smk.utils.load_weights(model, model_path)
    assert model is not None


@pytest.mark.parametrize(
    "architecture, weights_type",
    [
        ["mobilenetv2+linknet", None],
        ["mobilenetv2+linknet", "imagenet"],
        ["mobilenetv2+linknet", "aerial"],
        ["inceptionresnetv2+unet", "aerial"],
    ],
)
def test_get_model(architecture: str, weights_type: str):
    model, preprocess_input = mf.get_model(
        architecture=architecture, weights_type=weights_type
    )
    assert model is not None
    assert preprocess_input is not None


@pytest.mark.parametrize(
    "architecture, err_msg",
    [
        ("mobilenetv2+unknown", "Unknown decoder architecture"),
        ("unknown+unet", "Backbone with name 'unknown' is not supported"),
        ("unknown", "Unsupported architecture"),
    ],
)
def test_get_model_architecture_unknown(architecture: str, err_msg: str):
    with pytest.raises(ValueError, match=err_msg):
        _ = mf.get_model(architecture=architecture)


def test_get_model_weights_type_unknown():
    architecture = "mobilenetv2+linknet"
    weights_type = "unknown"
    with pytest.raises(ValueError, match="No weights available for"):
        _ = mf.get_model(
            architecture=architecture,
            input_width=256,
            input_height=256,
            nb_classes=5,
            weights_type=weights_type,
        )


@pytest.mark.parametrize(
    "loss, class_weights",
    [
        ("categorical_crossentropy", None),
        ("categorical_focal_crossentropy", None),
        ("categorical_focal_crossentropy", [1, 2, 3, 4, 5]),
    ],
)
def test_get_loss_func(loss, class_weights):
    if not KERAS_GTE_3:
        pytest.skip("loss function is not supported in keras < 3: {loss}")

    loss_func = mf._get_loss_func(loss, class_weights=class_weights)
    assert loss_func is not None


def test_get_loss_func_error():
    loss = "weighted_categorical_crossentropy"
    with pytest.raises(ValueError, match=f"With {loss=}, class_weights cannot be None"):
        _ = mf._get_loss_func(loss, class_weights=None)


@pytest.mark.parametrize("optimizer", ["Adam", "AdamW", "SGD"])
def test_get_optimizer_func(optimizer):
    opt = mf._get_optimizer_func(optimizer, params={"learning_rate": 0.0001})
    assert opt is not None


def test_get_optimizer_func_unknown():
    optimizer = "unknown"
    with pytest.raises(ValueError, match=f"Optimizer {optimizer} not found"):
        _ = mf._get_optimizer_func(optimizer, params={"learning_rate": 0.0001})


def test_load_model_hyperparams_invalid_path(tmp_path):
    invalid_path = tmp_path / "subject_1_0.86_5.keras"
    with pytest.raises(FileNotFoundError, match="No hyperparams file found for model"):
        _ = mf.load_model_hyperparams(invalid_path)

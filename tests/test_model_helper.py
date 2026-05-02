"""Tests for the model_helper module."""

import re
from pathlib import Path

import pytest

from orthoseg._compat import KERAS_GTE_3
from orthoseg.helpers import config_helper as conf
from orthoseg.model import model_helper


@pytest.mark.parametrize(
    "input_names, expected_name",
    [
        (
            ["subj_1_0.8_10.hdf5", "subj_2_0.9_20.hdf5", "subj_3_0.85_15.hdf5"],
            "subj_3_0.85_15.hdf5",
        ),
        (
            ["subj_1_0.8_10.hdf5", "subj_3_0.9_20.hdf5", "subj_3_0.85_15.hdf5"],
            "subj_3_0.9_20.hdf5",
        ),
        (
            ["subj_1_0.8_10.hdf5", "subj_2_0.9_20.keras", "subj_2_0.9_20.hdf5"],
            "subj_2_0.9_20.keras",
        ),
        (
            ["subj_1_0.8_10.hdf5", "subj_2_0.9_20.hdf5", "subj_2_0.9_20_tf"],
            "subj_2_0.9_20_tf",
        ),
        (
            ["subj_1_0.8_10.hdf5", "subj_2_0.9_20.hdf5", "subj_2_0.9_21.hdf5"],
            "subj_2_0.9_20.hdf5",
        ),
        (["subj_1_0.8_10.invalid"], None),
        (["subj_1_0.8_10_invalid"], None),
    ],
)
def test_get_best_model(tmp_path, input_names, expected_name):
    # Create dummy model files/directories.
    for name in input_names:
        path = tmp_path / name
        if path.suffix in (".keras", ".hdf5", ".invalid"):
            path.touch()
        else:
            path.mkdir()

    best_model = model_helper.get_best_model(tmp_path)

    if expected_name is None:
        assert best_model is None
    else:
        if not KERAS_GTE_3 and expected_name.endswith(".keras"):
            expected_name = expected_name.replace(".keras", ".hdf5")
        assert Path(best_model["filepath"]).name == expected_name


@pytest.mark.parametrize(
    "filename, expected_info",
    [
        (
            "subj_1_0.8_10.hdf5",
            {
                "traindata_id": 1,
                "monitor_metric_accuracy": 0.8,
                "epoch": 10,
                "save_format": "h5",
            },
        ),
        (
            "subj_2_0.9_20.keras",
            {
                "traindata_id": 2,
                "monitor_metric_accuracy": 0.9,
                "epoch": 20,
                "save_format": "keras",
            },
        ),
        (
            "subj_3_0.85_15_tf",
            {
                "traindata_id": 3,
                "monitor_metric_accuracy": 0.85,
                "epoch": 15,
                "save_format": "tf",
            },
        ),
    ],
)
def test_parse_model_filename(tmp_path, filename, expected_info):
    # Create dummy model file/directory.
    tmp_file = tmp_path / filename
    if tmp_file.suffix in (".keras", ".hdf5", ".invalid"):
        tmp_file.touch()
    else:
        tmp_file.mkdir()

    model_info = model_helper.parse_model_filename(tmp_file)
    if expected_info is None:
        assert model_info is None
    else:
        for key, value in expected_info.items():
            assert model_info[key] == value


@pytest.mark.parametrize(
    "filename, err_msg",
    [
        (
            "subj_3_0.85_15.invalid",
            "Model file should have .h5, .hdf5, or .keras as suffix",
        ),
        (
            "subj_3_0.85_15_invalid",
            "Not a valid path for a model, dir needs to end on _tf",
        ),
    ],
)
def test_parse_model_filename_invalid(tmp_path, filename, err_msg):
    # Create dummy model file/directory.
    tmp_file = tmp_path / filename
    if tmp_file.suffix in (".keras", ".hdf5", ".invalid"):
        tmp_file.touch()
    else:
        tmp_file.mkdir()

    with pytest.raises(ValueError, match=re.escape(err_msg)):
        _ = model_helper.parse_model_filename(tmp_file)


def test_trainparams_backwards_compatibility():
    """Test if the TrainParams class can still load old hyperparams.

    In orthoseg < 0.8, `weights_type` was stored in a parameter called `weights`. Check
    if this parameter can still be used for backwards compatibility.
    """
    old_hyperparams = {
        "trainparams_id": 1,
        "image_augmentations": {"cval": 0, "fill_mode": "constant", "rescale": 1},
        "mask_augmentations": {"cval": 0, "fill_mode": "constant", "rescale": 1},
        "weights": "old_weights_type",
        "class_weights": None,
        "batch_size": 32,
    }
    params = model_helper.TrainParams(**old_hyperparams)
    assert params.weights_type == "old_weights_type"


@pytest.mark.parametrize("class_weights", [None, {0: 1.0, 1: 2.0}])
def test_trainparams_defaults(class_weights):
    params = model_helper.TrainParams(
        image_augmentations={"cval": 0, "fill_mode": "constant", "rescale": 1},
        mask_augmentations={"cval": 0, "fill_mode": "constant", "rescale": 1},
        class_weights=class_weights,
    )

    assert params.trainparams_id == 0
    assert params.weights_type == "aerial"
    assert params.save_format == "keras" if KERAS_GTE_3 else "h5"
    if KERAS_GTE_3:
        expected_loss_function = "categorical_focal_crossentropy"
    else:
        if class_weights is not None:
            expected_loss_function = "weighted_categorical_crossentropy"
        else:
            expected_loss_function = "categorical_crossentropy"
    assert params.loss_function == expected_loss_function


@pytest.mark.parametrize(
    "image_augmentations, mask_augmentations",
    [
        (
            {"fill_mode": "constant", "cval": 255, "rescale": 0.001},
            {"fill_mode": "constant", "cval": 0, "rescale": 1},
        ),
    ],
)
def test_validate_augmentations(image_augmentations, mask_augmentations):
    model_helper._validate_augmentations(
        dict(image_augmentations), dict(mask_augmentations)
    )


def test_validate_augmentations_defaults(tmp_path):
    tmp_ini_path = tmp_path / "tmp.ini"
    tmp_ini_path.touch()
    conf.read_orthoseg_config(
        tmp_ini_path,
        overrules=[
            "general.segment_subject = test",
            f"files.image_layers_config_filepath = {tmp_ini_path.as_posix()}",
        ],
    )
    image_augmentations = conf.train.getdict("image_augmentations")
    mask_augmentations = conf.train.getdict("mask_augmentations")
    model_helper._validate_augmentations(image_augmentations, mask_augmentations)


@pytest.mark.parametrize(
    "image_augmentations, mask_augmentations, expected_error",
    [
        (
            {"fill_mode": "constant", "cval": 255, "rescale": 0.001},
            {"fill_mode": "constant", "cval": 1, "rescale": 1},
            "['cval for mask_augmentations should be 0, not 1']",
        ),
        (
            {"brightness_range": [0.9, 1.1]},
            {"brightness_range": [0.9, 1.1]},
            "brightness_range for mask_augmentations should be",
        ),
        (
            {"zoom_range": 0.1},
            {"zoom_range": 0.2},
            "['when zoom_range is used, it should be in image_augmentations and "
            "mask_augmentations with the same value'",
        ),
        (
            {"zoom_range": 9},
            {},
            "the same augmentations should be specified in the same order",
        ),
    ],
)
def test_validate_augmentations_error(
    image_augmentations, mask_augmentations, expected_error
):
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        model_helper._validate_augmentations(
            dict(image_augmentations), dict(mask_augmentations)
        )

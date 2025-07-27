"""Tests for the model_helper module."""

import re

import pytest

from orthoseg.helpers import config_helper as conf
from orthoseg.model import model_helper


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


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
)
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

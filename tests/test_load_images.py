"""Tests for the load_images module."""

import os
import shutil
from pathlib import Path

import pytest

import orthoseg
from orthoseg import load_images
from orthoseg.helpers import config_helper as conf
from orthoseg.load_images import _load_images_args
from tests import test_helper


@pytest.mark.parametrize(
    "args",
    [
        (
            [
                "--config",
                "X:/Monitoring/OrthoSeg/test/test.ini",
                "predict.image_layer=LT-2023",
            ]
        )
    ],
)
def test_load_images_args(args):
    valid_args = _load_images_args(args=args)
    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None


def test_load_images_error_handling():
    """Force an error so the general error handler in predict is tested."""
    with pytest.raises(
        RuntimeError,
        match="ERROR in load_images for footballfields_BEFL-2019",
    ):
        load_images(
            config_path=test_helper.SampleProjectFootball.predict_config_path,
            config_overrules=["predict.image_pixel_x_size=INVALID_TYPE"],
        )


def test_load_images_image_layer_list(tmp_path):
    config_path = test_helper.SampleProjectFootball.predict_config_path
    predict_image_input_base_dir = tmp_path / "cache"
    predict_image_input_dir = predict_image_input_base_dir / "{predict_image_layer}"

    load_images(
        config_path=config_path,
        config_overrules=[
            "predict.image_layer=BEFL-2019,BEFL-2020",
            f"dirs.predict_image_input_dir={predict_image_input_dir.as_posix()}",
        ],
    )

    # Check if the right output directories were created
    assert predict_image_input_base_dir.exists()
    files = predict_image_input_base_dir.glob("**/*.jpg")
    assert len(list(files)) == 4  # 2 images per layer

    assert (predict_image_input_base_dir / "BEFL-2019").exists()
    assert (predict_image_input_base_dir / "BEFL-2020").exists()


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
)
@pytest.mark.parametrize(
    "overrules, exp_image_count",
    [
        (["predict.image_layer=BEFL-2019-WMTS"], 2),
        (["predict.image_layer=OSM-XYZ"], 2),
        (["predict.image_layer=BEFL-2019"], 2),
    ],
)
def test_load_images(tmp_path, overrules, exp_image_count):
    # Use footballfields sample project for these end to end tests
    testprojects_dir = tmp_path / "sample_projects"
    footballfields_dir = testprojects_dir / "footballfields"
    image_cache_dir = testprojects_dir / "_image_cache"
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)

    # Run task to load images
    orthoseg.load_images(
        footballfields_dir / "footballfields_BEFL-2019.ini",
        config_overrules=overrules,
    )

    # Check if the right number of files was loaded
    assert image_cache_dir.exists()
    files = list(image_cache_dir.glob("**/*.jpg"))
    assert len(files) == exp_image_count

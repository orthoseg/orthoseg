"""Tests for the load_images module."""

import pytest

from orthoseg import load_images
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
        match="ERROR in load_images for footballfields_BEFL-2019_test",
    ):
        load_images(
            config_path=test_helper.SampleProjectFootball.predict_config_path,
            config_overrules=["predict.image_pixel_width=INVALID_TYPE"],
        )


import shutil

import pytest

import orthoseg


@pytest.mark.parametrize(
    "overrules",
    [
        ["predict.image_layer=BEFL-2019-WMTS"],
        ["predict.image_layer=BEFL-2019-XYZ"],
        [],
    ],
)
def test_load_images(tmp_path, overrules):
    # Use footballfields sample project for these end to end tests
    testprojects_dir = tmp_path / "sample_projects"
    footballfields_dir = testprojects_dir / "footballfields"
    image_cache_dir = testprojects_dir / "_image_cache"
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)

    # Run task to load images
    orthoseg.load_images(
        footballfields_dir / "footballfields_BEFL-2019_test.ini",
        config_overrules=overrules,
    )

    # Check if the right number of files was loaded
    assert image_cache_dir.exists()
    files = list(image_cache_dir.glob("**/*.jpg"))
    assert len(files) == 6

"""Tests for the load_images module."""

import os
import shutil

import pytest

import orthoseg
from orthoseg import load_images
from orthoseg.load_images import _load_images_args
from tests import test_helper
from tests.test_helper import SportsFields


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
    with pytest.raises(RuntimeError, match="ERROR in load_images for sportsfields"):
        load_images(
            config_path=SportsFields.config_path,
            config_overrules=["predict.image_pixel_width=INVALID_TYPE"],
        )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
)
@pytest.mark.parametrize(
    "image_layer_overrule, exp_image_count",
    [
        ("predict.image_layer=BEFL-2025-footballfield-WMTS", 2),
        ("predict.image_layer=OSM-XYZ-footballfield", 2),
        ("predict.image_layer=BEFL-2025-sportsfields", 8),
    ],
)
def test_load_images(tmp_path, image_layer_overrule, exp_image_count):
    # Use sportsfields sample project for these end to end tests
    if (
        image_layer_overrule == "predict.image_layer=BEFL-2025-footballfield-WMTS"
        and "GITHUB_ACTIONS" in os.environ
    ):
        pytest.skip("Skipping this test on GITHUB_ACTIONS as it is very slow there.")

    testprojects_dir = tmp_path / "sample_projects"
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)
    project_dir = testprojects_dir / SportsFields.subject
    _, image_layer = image_layer_overrule.split("=")
    image_cache_dir = testprojects_dir / "_image_cache" / image_layer

    # Add extra overrules to make images smaller and improve test speed.
    all_overrules = [
        image_layer_overrule,
        "predict.image_pixel_width=128",
        "predict.image_pixel_height=128",
        "predict.image_pixel_x_size=2",
        "predict.image_pixel_y_size=2",
        "predict.image_pixels_overlap=16",
    ]

    # Run task to load images
    orthoseg.load_images(
        project_dir / SportsFields.config_path.name, config_overrules=all_overrules
    )

    # Check if the right number of files was loaded
    assert image_cache_dir.exists()
    files = list(image_cache_dir.glob("**/*.jpg"))
    assert len(files) == exp_image_count

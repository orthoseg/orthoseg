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

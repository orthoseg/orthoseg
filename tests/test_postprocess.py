"""Tests for module postprocess."""

import os

import pytest

from orthoseg import postprocess
from orthoseg.postprocess import _postprocess_args
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
def test_postprocess_args(args):
    valid_args = _postprocess_args(args=args)
    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
)
def test_postprocess_error_handling():
    """Force an error so the general error handler in postprocess is tested."""
    with pytest.raises(
        RuntimeError,
        match="ERROR in postprocess for footballfields_BEFL-2019_test on UNEXISTING",
    ):
        postprocess(
            config_path=test_helper.SampleProjectFootball.predict_config_path,
            config_overrules=["predict.image_layer=UNEXISTING"],
        )

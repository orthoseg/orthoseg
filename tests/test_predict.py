from contextlib import nullcontext
from pathlib import Path

import pytest

from orthoseg import predict
from orthoseg.predict import _predict_args


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
def test_predict_args(args):
    valid_args = _predict_args(args=args)
    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None


@pytest.mark.parametrize("config_path, exp_error", [("INVALID", True)])
def test_predict(config_path, exp_error):
    if exp_error:
        handler = pytest.raises(ValueError)
    else:
        handler = nullcontext()
    with handler:
        predict(config_path=Path("INVALID"))

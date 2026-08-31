"""Tests for module train."""

import importlib
from contextlib import nullcontext
from pathlib import Path

import pytest

from orthoseg import train
from orthoseg.train import _train_args
from tests import test_helper

train_module = importlib.import_module("orthoseg.train")


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
def test_train_args(args):
    valid_args = _train_args(args=args)
    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None


@pytest.mark.parametrize("config_path, exp_error", [(Path("INVALID"), True)])
def test_train(config_path, exp_error):
    if exp_error:
        handler = pytest.raises(ValueError)
    else:
        handler = nullcontext()
    with handler:
        train(config_path=config_path)


def test_train_error_handling():
    """Force an error so the general error handler in train is tested."""
    with pytest.raises(RuntimeError, match="ERROR in train for sportsfields"):
        train(
            config_path=test_helper.SportsFields.config_path,
            config_overrules=["train.force_model_traindata_id=INVALID_TYPE"],
        )


@pytest.mark.parametrize(
    "dtype_policy_raw, expected_calls",
    [
        ("float32", ["float32"]),
        ("", ["float32"]),
        (None, ["float32"]),
    ],
)
def test_set_dtype_policy_for_train(monkeypatch, dtype_policy_raw, expected_calls):
    calls = []

    monkeypatch.setattr(
        train_module.mf,
        "set_dtype_policy",
        lambda policy: calls.append(policy),
    )

    train_module._set_dtype_policy_for_train(dtype_policy_raw)

    assert calls == expected_calls

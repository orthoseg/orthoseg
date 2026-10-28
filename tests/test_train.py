"""Tests for module train."""

import importlib
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

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


def test_train_error_handling(monkeypatch, tmp_path):
    """Force an error so the general error handler in train is tested."""

    def fail_get_train_label_infos():
        raise ValueError("forced training error")

    fake_conf = SimpleNamespace(
        read_orthoseg_config=lambda *_args, **_kwargs: None,
        dirs=SimpleNamespace(getpath=lambda key: tmp_path / key),
        logging_conf=SimpleNamespace(getint=lambda _key: 1),
        train=SimpleNamespace(get=lambda _key, default=None: default),
        get_train_label_infos=fail_get_train_label_infos,
        pformat_config=lambda: "",
        remove_run_tmp_dir=lambda: None,
    )
    monkeypatch.setattr(train_module, "conf", fake_conf)
    monkeypatch.setattr(train_module.log_util, "clean_log_dir", lambda **_kwargs: None)
    monkeypatch.setattr(
        train_module.log_util,
        "main_log_init",
        lambda *_args: logging.getLogger(__name__),
    )
    monkeypatch.setattr(train_module.mf, "set_dtype_policy", lambda _policy: None)
    monkeypatch.setattr(
        train_module.email_helper,
        "sendmail",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="ERROR in train for sportsfields"):
        train(
            config_path=test_helper.SportsFields.config_path,
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

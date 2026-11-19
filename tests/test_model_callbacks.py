"""Tests for the model training callbacks."""

import pytest

from orthoseg.model import model_callbacks


def test_model_checkpoint_ext_on_epoch_end_reports_monitor_value(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        model_callbacks, "save_and_clean_models", lambda **kwargs: calls.append(kwargs)
    )
    callback = _create_checkpoint(tmp_path, monitor_metric="{accuracy}+{val_accuracy}")
    callback.set_model(object())

    callback.on_epoch_end(3, {"accuracy": 0.8, "val_accuracy": 0.6})

    assert calls[0]["new_model_monitor_value"] == pytest.approx(1.4)
    assert calls[0]["new_model_epoch"] == 3


def test_model_checkpoint_ext_rejects_invalid_mode(tmp_path):
    with pytest.raises(ValueError, match="Invalid value for mode"):
        _create_checkpoint(tmp_path, monitor_metric_mode="invalid")


def test_model_checkpoint_ext_rejects_invalid_save_format(tmp_path):
    with pytest.raises(ValueError, match="Invalid value for save_format"):
        _create_checkpoint(tmp_path, save_format="invalid")


def test_model_checkpoint_ext_rejects_tf_format_on_non_tensorflow_backend(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        model_callbacks, "get_keras_backend", lambda _keras_module: "torch"
    )

    with pytest.raises(
        ValueError, match="TensorFlow SavedModel output requires the TensorFlow"
    ):
        _create_checkpoint(tmp_path, save_format="tf")


def test_model_checkpoint_ext_uses_metric_value_when_logs_are_missing(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        model_callbacks, "save_and_clean_models", lambda **kwargs: calls.append(kwargs)
    )
    callback = _create_checkpoint(tmp_path, monitor_metric="accuracy")
    callback.set_model(object())

    callback.on_epoch_end(0)

    assert calls[0]["new_model_monitor_value"] is None


def _create_checkpoint(tmp_path, **overrides):
    kwargs = {
        "model_save_dir": tmp_path,
        "segment_subject": "subject",
        "traindata_id": 1,
        "monitor_metric": "accuracy",
        "monitor_metric_mode": "max",
    }
    kwargs.update(overrides)
    return model_callbacks.ModelCheckpointExt(**kwargs)

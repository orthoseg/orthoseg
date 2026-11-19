import importlib
import importlib.metadata
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from orthoseg import _compat


def test_compat_handles_missing_keras_distribution(monkeypatch):
    original_version = importlib.metadata.version

    def version(distribution):
        if distribution == "keras":
            raise PackageNotFoundError(distribution)
        return original_version(distribution)

    monkeypatch.setattr(importlib.metadata, "version", version)
    try:
        importlib.reload(_compat)

        assert _compat.KERAS_VERSION is None
        assert not _compat.KERAS_GTE_3
    finally:
        monkeypatch.undo()
        importlib.reload(_compat)


@pytest.mark.parametrize(
    ("backend", "available", "keras_gte_3", "error", "message"),
    [
        ("invalid", True, True, ValueError, "Unsupported Keras backend"),
        ("torch", False, True, ImportError, "is not installed"),
        ("torch", True, False, ImportError, "require Keras 3"),
    ],
)
def test_configure_backend_rejects_invalid_settings(
    monkeypatch, backend, available, keras_gte_3, error, message
):
    monkeypatch.setenv("KERAS_BACKEND", backend)
    monkeypatch.setattr(_compat, "KERAS_GTE_3", keras_gte_3)
    monkeypatch.setattr(_compat, "_module_is_available", lambda _name: available)

    with pytest.raises(error, match=message):
        _compat.configure_keras_backend()


def test_configure_backend_requires_installed_backend(monkeypatch):
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.setattr(_compat, "_module_is_available", lambda _name: False)

    with pytest.raises(ImportError, match="A Keras backend is required"):
        _compat.configure_keras_backend()


def test_configure_backend_selects_available_non_tensorflow_backend(monkeypatch):
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.setattr(_compat, "KERAS_GTE_3", True)
    monkeypatch.setattr(_compat, "_module_is_available", lambda name: name == "torch")

    assert _compat.configure_keras_backend() == "torch"
    assert os.environ["KERAS_BACKEND"] == "torch"


@pytest.mark.parametrize(
    ("keras_gte_3", "keras_module", "expected"),
    [
        (False, None, "tensorflow"),
        (
            True,
            SimpleNamespace(backend=SimpleNamespace(backend=lambda: "torch")),
            "torch",
        ),
    ],
)
def test_get_keras_backend(monkeypatch, keras_gte_3, keras_module, expected):
    monkeypatch.setattr(_compat, "KERAS_GTE_3", keras_gte_3)

    assert _compat.get_keras_backend(keras_module) == expected


@pytest.mark.skipif(
    not _compat.KERAS_GTE_3 or not _compat._module_is_available("torch"),
    reason="Requires Keras 3 and the torch backend",
)
def test_model_factory_configures_backend_before_importing_segmodels():
    script = """
import importlib.util
import os
import sys

os.environ.pop("KERAS_BACKEND", None)
original_find_spec = importlib.util.find_spec

def find_spec(name, *args, **kwargs):
    if name == "tensorflow":
        return None
    return original_find_spec(name, *args, **kwargs)

importlib.util.find_spec = find_spec

import orthoseg.model.model_factory
import keras

assert keras.backend.backend() == "torch"
assert not any(
    name == "tensorflow" or name.startswith("tensorflow.")
    for name in sys.modules
)
"""
    repository_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=repository_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("error", [ImportError, ValueError])
def test_module_is_available_handles_lookup_errors(monkeypatch, error):
    def raise_error(_module_name):
        raise error

    monkeypatch.setattr(_compat.importlib.util, "find_spec", raise_error)

    assert _compat._module_is_available("optional_backend") is False


def test_non_ml_imports_without_keras_or_tensorflow():
    script = """
import builtins
import sys

original_import = builtins.__import__

def reject_optional_backend(name, *args, **kwargs):
    if name.split(".", maxsplit=1)[0] in {"keras", "segmodels_keras", "tensorflow"}:
        raise AssertionError(f"unexpected optional backend import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_optional_backend
import orthoseg
from orthoseg import load_images, postprocess, validate

assert orthoseg.__version__
assert callable(load_images)
assert callable(postprocess)
assert callable(validate)
assert not any(
    name.split(".", maxsplit=1)[0] in {"keras", "segmodels_keras", "tensorflow"}
    for name in sys.modules
)
"""
    repository_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=repository_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )

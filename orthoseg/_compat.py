"""Compatibility helpers for optional Keras backends."""

import importlib.util
import os
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

try:
    KERAS_VERSION: str | None = version("keras")
except PackageNotFoundError:
    KERAS_VERSION = None

KERAS_GTE_3 = (
    KERAS_VERSION is not None and int(KERAS_VERSION.split(".", maxsplit=1)[0]) >= 3
)


def _get_version() -> str:
    version_path = Path(__file__).resolve().parent / "version.txt"
    with version_path.open() as file:
        return file.readline()


__version__ = _get_version()


def _module_is_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def configure_keras_backend() -> str:
    """Select an installed Keras backend, preferring TensorFlow for compatibility."""
    backend = os.environ.get("KERAS_BACKEND")
    if backend is None:
        backend = next(
            (
                candidate
                for candidate in ("tensorflow", "torch", "jax")
                if _module_is_available(candidate)
            ),
            None,
        )
        if backend is None:
            raise ImportError(
                "A Keras backend is required for model operations. Install one with "
                "'orthoseg[tensorflow]', 'orthoseg[torch]', or 'orthoseg[jax]'."
            )
        os.environ["KERAS_BACKEND"] = backend

    if backend not in {"tensorflow", "torch", "jax"}:
        raise ValueError(
            f"Unsupported Keras backend {backend!r}; choose tensorflow, torch, or jax."
        )
    if backend in {"tensorflow", "torch", "jax"} and not _module_is_available(backend):
        raise ImportError(
            f"The configured Keras backend {backend!r} is not installed. Install "
            f"'orthoseg[{backend}]' or select an installed backend with "
            "KERAS_BACKEND."
        )
    if backend != "tensorflow" and not KERAS_GTE_3:
        raise ImportError(
            "Keras backends other than TensorFlow require Keras 3. Install "
            "'orthoseg[torch]' or 'orthoseg[jax]' with Keras 3."
        )
    if backend == "tensorflow":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    return backend


def configure_tensorflow_warnings() -> None:
    """Configure warning filters needed for the TensorFlow backend."""
    if os.environ.get("KERAS_BACKEND") == "tensorflow" and sys.version_info >= (3, 13):
        warnings.filterwarnings(
            "ignore",
            message=r".*\.__init__ got an unexpected keyword argument "
            r"'___pyct_anno'",
            category=DeprecationWarning,
        )


def get_keras_backend(keras_module: Any) -> str:
    """Return the active backend, including the TensorFlow-only Keras 2 case."""
    if not KERAS_GTE_3:
        return "tensorflow"
    return keras_module.backend.backend()

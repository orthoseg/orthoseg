"""OrthoSeg makes it easy to train neural networks to segment orthophotos."""

from importlib import import_module
from typing import TYPE_CHECKING

from orthoseg._compat import __version__

if TYPE_CHECKING:
    from orthoseg.load_images import load_images
    from orthoseg.postprocess import postprocess
    from orthoseg.predict import predict
    from orthoseg.train import train

# Keep the public function bound when the validate submodule is imported elsewhere.
from orthoseg.validate import validate

_PUBLIC_API_MODULES = {
    "load_images": "orthoseg.load_images",
    "postprocess": "orthoseg.postprocess",
    "predict": "orthoseg.predict",
    "train": "orthoseg.train",
}

__all__ = (
    "__version__",
    "load_images",
    "postprocess",
    "predict",
    "train",
    "validate",
)


def __getattr__(name: str):
    """Load public API modules on demand, leaving non-ML imports backend-free."""
    module_name = _PUBLIC_API_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

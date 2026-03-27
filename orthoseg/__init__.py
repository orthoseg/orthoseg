"""OrthoSeg makes it easy to train neural networks to segment orthophotos."""

import os
from pathlib import Path

# Default to using tensorflow as keras backend if not specified.
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"

# ruff: noqa: F401
from orthoseg.load_images import load_images
from orthoseg.postprocess import postprocess
from orthoseg.predict import predict
from orthoseg.train import train
from orthoseg.validate import validate

print(f"Using Keras backend: {os.environ['KERAS_BACKEND']}")


def _get_version():
    version_path = Path(__file__).resolve().parent / "version.txt"
    with version_path.open() as file:
        return file.readline()


__version__ = _get_version()

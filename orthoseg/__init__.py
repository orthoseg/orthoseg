"""OrthoSeg makes it easy to train neural networks to segment orthophotos."""

from pathlib import Path

# Import tensorflow first to avoid CI segmentation faults on Windows.
import tensorflow as tf

# ruff: noqa: F401
from orthoseg.load_images import load_images
from orthoseg.postprocess import postprocess
from orthoseg.predict import predict
from orthoseg.train import train
from orthoseg.validate import validate


def _get_version():
    version_path = Path(__file__).resolve().parent / "version.txt"
    with version_path.open() as file:
        return file.readline()


__version__ = _get_version()

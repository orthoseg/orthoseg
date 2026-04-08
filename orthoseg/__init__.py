"""OrthoSeg makes it easy to train neural networks to segment orthophotos."""

import os
from pathlib import Path

# Default to using tensorflow as keras backend if not specified.
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

# ruff: noqa: F401
from orthoseg._compat import __version__
from orthoseg.load_images import load_images
from orthoseg.postprocess import postprocess
from orthoseg.predict import predict
from orthoseg.train import train
from orthoseg.validate import validate

print(f"Using Keras backend: {keras.backend.backend()}")

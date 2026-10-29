"""Shared pytest configuration."""

import gc
import os


def pytest_runtest_teardown(item, nextitem):  # noqa: ARG001
    """Run garbage collection after each test on Windows.

    This prevents C-extension segfaults caused by accumulated cyclic references.
    """
    if os.name == "nt":
        gc.collect()


if (
    os.name == "nt"
    and "GITHUB_ACTIONS" in os.environ
    and os.environ.get("KERAS_BACKEND", "tensorflow") == "tensorflow"
):
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import keras
    import tensorflow as tf

    # Import the lazy optimizer API before native-extension tests run.
    _ = tf.keras.optimizers
    if not keras.__version__.startswith("3."):
        from tensorflow.keras.optimizers import experimental  # noqa: F401

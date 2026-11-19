"""Load Keras only when model functionality is used."""

from orthoseg._compat import configure_keras_backend

configure_keras_backend()

import keras  # noqa: E402

__all__ = ["keras"]

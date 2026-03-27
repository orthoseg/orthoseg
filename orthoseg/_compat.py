"""Module to centralise version checks."""

import keras

KERAS_GTE_3 = keras.__version__.startswith("3.")

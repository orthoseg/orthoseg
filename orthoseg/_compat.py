"""Module to centralise version checks."""

from pathlib import Path

import keras

KERAS_GTE_3 = keras.__version__.startswith("3.")


def _get_version():
    version_path = Path(__file__).resolve().parent / "version.txt"
    with version_path.open() as file:
        return file.readline()


__version__ = _get_version()

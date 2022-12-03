# Import the high level API
# flake8: noqa: F401
from pathlib import Path

from orthoseg.load_images import load_images
from orthoseg.train import train
from orthoseg.train import _search_label_files
from orthoseg.predict import predict
from orthoseg.postprocess import postprocess


def _get_version():
    version_path = Path(__file__).resolve().parent / "version.txt"
    with open(version_path, mode="r") as file:
        return file.readline()


__version__ = _get_version()

# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.train.
"""

from pathlib import Path
import sys

# Add path so the local orthoseg packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import orthoseg
from tests import test_helper


def test_search_label_files():
    labeldata_template = (
        test_helper.get_testdata_dir() / "footballfields_{image_layer}_data.gpkg"
    )
    labellocation_template = (
        test_helper.get_testdata_dir() / "footballfields_{image_layer}_locations.gpkg"
    )
    result = orthoseg._search_label_files(labeldata_template, labellocation_template)

    assert len(result) == 2

# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.train.
"""

from pathlib import Path
import sys
import pytest

# Add path so the local orthoseg packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import orthoseg
from tests.test_helper import TestData


def test_search_label_files():
    labeldata_template = (
        TestData.testdata_dir / "footballfields_{image_layer}_data.gpkg"
    )
    labellocation_template = (
        TestData.testdata_dir / "footballfields_{image_layer}_locations.gpkg"
    )
    image_layers = {
        "BEFL-2019": {"pixel_x_size": 1, "pixel_y_size": 2},
        "BEFL-2020": {},
    }
    results = orthoseg._search_label_files(
        labeldata_template, labellocation_template, image_layers=image_layers
    )

    assert len(results) == 2
    for result in results:
        if result.image_layer == "BEFL-2019":
            assert result.pixel_x_size == 1
            assert result.pixel_y_size == 2
        else:
            assert result.pixel_x_size is None
            assert result.pixel_y_size is None


def test_search_label_files_invalid_layer():
    labeldata_template = (
        TestData.testdata_dir / "footballfields_{image_layer}_data.gpkg"
    )
    labellocation_template = (
        TestData.testdata_dir / "footballfields_{image_layer}_locations.gpkg"
    )
    image_layers = {"BEFL-2019": {"pixel_x_size": 1, "pixel_y_size": 1}}
    with pytest.raises(ValueError, match="image layer not found: BEFL-2020"):
        _ = orthoseg._search_label_files(
            labeldata_template, labellocation_template, image_layers=image_layers
        )

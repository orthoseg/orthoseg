"""
Tests for functionalities in orthoseg.train.
"""

import orthoseg
from tests.test_helper import TestData


def test_search_label_files():
    labeldata_template = (
        TestData.testdata_dir / "footballfields_{image_layer}_data.gpkg"
    )
    labellocation_template = (
        TestData.testdata_dir / "footballfields_{image_layer}_locations.gpkg"
    )
    result = orthoseg._search_label_files(labeldata_template, labellocation_template)

    assert len(result) == 2

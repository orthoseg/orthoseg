import pytest

from orthoseg.helpers import config_helper as conf
from tests.test_helper import SampleProjectFootball
from tests.test_helper import TestData


def test_read_orthoseg_config_predict():
    # Load project config to init some vars.
    conf.read_orthoseg_config(SampleProjectFootball.predict_config_path)

    layer = conf.image_layers.get("BEFL-2019")
    assert layer is not None
    pixel_x_size = layer.get("pixel_x_size")
    assert pixel_x_size == 0.25
    pixel_y_size = layer.get("pixel_y_size")
    assert pixel_y_size == 0.25


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
    results = conf._search_label_files(
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
        _ = conf._search_label_files(
            labeldata_template, labellocation_template, image_layers=image_layers
        )

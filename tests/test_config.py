import pytest

from orthoseg.helpers import config_helper as conf
from orthoseg.lib.prepare_traindatasets import LabelInfo
from tests.test_helper import SampleProjectFootball
from tests.test_helper import TestData


def test_read_orthoseg_config_predict():
    # Load project config to init some vars.
    conf.read_orthoseg_config(SampleProjectFootball.predict_config_path)

    layer = conf.image_layers.get("BEFL-2019")
    assert layer is not None


def test_prepare_train_label_infos():
    labelpolygons_pattern = TestData.dir / "footballfields_{image_layer}_data.gpkg"
    labellocations_pattern = (
        TestData.dir / "footballfields_{image_layer}_locations.gpkg"
    )
    image_layers = {"BEFL-2019": {}, "BEFL-2020": {}, "BEFL-2021": {}, "BEFL-2022": {}}
    label_datasources = {
        "label_ds1": {
            "locations_path": str(
                TestData.dir / "footballfields_BEFL-2019_locations.gpkg"
            ),
            "polygons_path": str(TestData.dir / "footballfields_BEFL-2019_data.gpkg"),
            "pixel_x_size": 1,
            "pixel_y_size": 2,
            "image_layer": "BEFL-2021",
        },
        "label_ds2": {
            "locations_path": str(
                TestData.dir / "footballfields_BEFL-2022_locations.gpkg"
            ),
            "data_path": str(TestData.dir / "footballfields_BEFL-2022_data.gpkg"),
            "pixel_x_size": 5,
            "pixel_y_size": 6,
            "image_layer": "BEFL-2022",
        },
    }

    label_infos_result = conf._prepare_train_label_infos(
        labelpolygons_pattern,
        labellocations_pattern,
        label_datasources=label_datasources,
        image_layers=image_layers,
    )
    assert len(label_infos_result) == 3
    for result in label_infos_result:
        if result.image_layer == "BEFL-2021":
            assert result.pixel_x_size == 1
            assert result.pixel_y_size == 2
        elif result.image_layer == "BEFL-2022":
            assert result.pixel_x_size == 5
            assert result.pixel_y_size == 6
        else:
            assert result.pixel_x_size is None
            assert result.pixel_y_size is None


def test_prepare_train_label_infos_invalid_layer():
    labeldata_pattern = TestData.dir / "footballfields_{image_layer}_data.gpkg"
    labellocation_pattern = TestData.dir / "footballfields_{image_layer}_locations.gpkg"
    image_layers = {"BEFL-2019": {}}

    with pytest.raises(ValueError, match="invalid image_layer in <LabelInfo"):
        _ = conf._prepare_train_label_infos(
            labeldata_pattern,
            labellocation_pattern,
            label_datasources={},
            image_layers=image_layers,
        )


def test_search_label_files():
    labelpolygons_pattern = TestData.dir / "footballfields_{image_layer}_data.gpkg"
    labellocation_pattern = TestData.dir / "footballfields_{image_layer}_locations.gpkg"
    results = conf._search_label_files(labelpolygons_pattern, labellocation_pattern)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, LabelInfo)


def test_search_label_files_invalid_dir():
    invalid_dir = TestData.dir / "unexisting"
    labelpolygons_pattern = TestData.dir / "footballfields_{image_layer}_data.gpkg"
    labellocation_pattern = invalid_dir / "footballfields_{image_layer}_locations.gpkg"

    with pytest.raises(ValueError, match="Label dir doesn't exist"):
        _ = conf._search_label_files(labelpolygons_pattern, labellocation_pattern)

    labelpolygons_pattern = invalid_dir / "footballfields_{image_layer}_data.gpkg"
    labellocation_pattern = TestData.dir / "footballfields_{image_layer}_locations.gpkg"

    with pytest.raises(ValueError, match="Label dir doesn't exist"):
        _ = conf._search_label_files(labelpolygons_pattern, labellocation_pattern)


def test_unformat():
    result = conf._unformat(
        "footballfields_BEFL-2018_data.gpkg",
        pattern="footballfields_{image_layer}_data.gpkg",
    )
    assert result == {"image_layer": "BEFL-2018"}


def test_unformat_error():
    with pytest.raises(
        ValueError, match="pattern fields_{image_layer}_data.gpkg not found"
    ):
        _ = conf._unformat(
            "fields_BEFL-2018_polygons.gpkg",
            pattern="fields_{image_layer}_data.gpkg",
        )

import re
import tempfile
from pathlib import Path

import pytest

from orthoseg.helpers import config_helper as conf
from orthoseg.lib.prepare_traindatasets import LabelInfo
from tests.test_helper import SampleProjectFootball, TestData


@pytest.mark.parametrize(
    "overrules, exp_error, message",
    [
        (
            ["general.segment_subject=MUST_OVERRIDE"],
            ValueError,
            "Projectconfig parameter general.segment_subject needs to be overruled",
        ),
        (
            ["general.segment_subject=Football_field"],
            ValueError,
            "should not contain any of the following chars",
        ),
    ],
)
def test_read_orthoseg_config_error_segment_subject(overrules, exp_error, message):
    with pytest.raises(
        exp_error,
        match=message,
    ):
        conf.read_orthoseg_config(
            SampleProjectFootball.predict_config_path, overrules=overrules
        )


def test_read_orthoseg_config_image_layers():
    # Load project config to init some vars.
    conf.read_orthoseg_config(SampleProjectFootball.predict_config_path)

    layer = conf.image_layers.get("BEFL-2019")
    assert layer is not None


@pytest.mark.parametrize(
    "overrules, expected_image_layer",
    [
        (None, "BEFL-2019"),
        ([], "BEFL-2019"),
        (["predict.image_layer=BEFL-2020"], "BEFL-2020"),
    ],
)
def test_read_orthoseg_config_predict_overrules(overrules, expected_image_layer):
    # Load project config to test overrules.
    kwargs = {}
    if overrules is not None:
        kwargs["overrules"] = overrules
    conf.read_orthoseg_config(SampleProjectFootball.predict_config_path, **kwargs)

    image_layer = conf.predict.get("image_layer")
    assert image_layer == expected_image_layer


@pytest.mark.parametrize(
    "overrules",
    [["predictimage_layer=BEFL-2020"], ["predict.image_layerBEFL-2020"]],
)
def test_read_orthoseg_config_predict_overrules_invalid(overrules):
    """Test with invalid overrules: one without '=' and one without '.'."""
    # Load project config to test overrules.
    with pytest.raises(ValueError, match="invalid config overrule found"):
        conf.read_orthoseg_config(
            SampleProjectFootball.predict_config_path, overrules=overrules
        )


@pytest.mark.parametrize(
    "overrules, expected_image_layer",
    [
        (None, None),
        ([], None),
        (["train.image_layer=BEFL-2020"], "BEFL-2020"),
    ],
)
def test_read_orthoseg_config_train_overrules(overrules, expected_image_layer):
    # Load project config to test overrules.
    kwargs = {}
    if overrules is not None:
        kwargs["overrules"] = overrules
    conf.read_orthoseg_config(SampleProjectFootball.train_config_path, **kwargs)

    image_layer = conf.train.get("image_layer")
    if expected_image_layer is None:
        assert image_layer is None
    else:
        assert image_layer == expected_image_layer


def test_prepare_train_label_infos():
    labelpolygons_pattern = TestData.dir / "footballfields_{image_layer}_data.gpkg"
    labellocations_pattern = (
        TestData.dir / "footballfields_{image_layer}_locations.gpkg"
    )
    image_layers = {"BEFL-2019": {}, "BEFL-2020": {}, "BEFL-2021": {}, "BEFL-2022": {}}
    label_datasources = {
        "label_ds1_resolution1": {
            "locations_path": str(
                TestData.dir / "footballfields_BEFL-2019_locations.gpkg"
            ),
            "pixel_x_size": 1,
            "pixel_y_size": 2,
            "image_layer": "BEFL-2021",
        },
        "label_ds1_resolution2_None": {
            "locations_path": str(
                TestData.dir / "footballfields_BEFL-2019_locations.gpkg"
            ),
            "data_path": str(TestData.dir / "footballfields_BEFL-2019_data.gpkg"),
        },
        "label_ds2": {
            "locations_path": str(
                TestData.dir / "footballfields_BEFL-2022_locations.gpkg"
            ),
            "polygons_path": str(TestData.dir / "footballfields_BEFL-2022_data.gpkg"),
            "image_layer": "BEFL-2022",
            "pixel_x_size": 5,
            "pixel_y_size": 6,
        },
    }

    label_infos_result = conf._prepare_train_label_infos(
        labelpolygons_pattern,
        labellocations_pattern,
        label_datasources=label_datasources,
        image_layers=image_layers,
    )
    assert len(label_infos_result) == 4
    for index, result in enumerate(label_infos_result):
        if index == 0:
            # label_ds1_resolution1
            # polygons_path is not overruled, so the pattern-discovered file is used
            exp = TestData.dir / "footballfields_BEFL-2019_data.gpkg"
            assert result.polygons_path.resolve().as_posix() == exp.resolve().as_posix()
            assert result.image_layer == "BEFL-2021"
            assert result.pixel_x_size == 1
            assert result.pixel_y_size == 2
        elif index == 1:
            # label_ds1_resolution2_None
            # data_path should also work for backwards compatibility
            exp = Path(label_datasources["label_ds1_resolution2_None"]["data_path"])
            assert result.polygons_path.resolve().as_posix() == exp.resolve().as_posix()
            # image_layer was not specified, so is reused from the pattern version
            assert result.image_layer == "BEFL-2019"
            assert result.pixel_x_size is None
            assert result.pixel_y_size is None
        elif index == 2:
            # label_ds2
            exp = Path(label_datasources["label_ds2"]["polygons_path"])
            assert result.polygons_path.resolve().as_posix() == exp.resolve().as_posix()
            assert result.image_layer == "BEFL-2022"
            assert result.pixel_x_size == 5
            assert result.pixel_y_size == 6
        else:
            # BEFL-2020 file is pattern-discovered but not overruled
            assert result.image_layer == "BEFL-2020"
            assert result.polygons_path is not None
            assert result.pixel_x_size is None
            assert result.pixel_y_size is None


@pytest.mark.parametrize(
    "label_datasources, expected_error",
    [
        ({"label_ds1": {"pixel_x_size": 1}}, "locations_path is mandatory"),
        ({"label_ds1": {"locations_path": None}}, "locations_path is mandatory"),
        (
            {"label_ds1": {"locations_path": "/tmp/tst.gpkg"}},
            "polygons_path not specified for",
        ),
    ],
)
def test_prepare_train_label_infos_error(label_datasources, expected_error):
    labelpolygons_pattern = TestData.dir / "footballfields_{image_layer}_data.gpkg"
    labellocations_pattern = (
        TestData.dir / "footballfields_{image_layer}_locations.gpkg"
    )
    image_layers = {"BEFL-2019": {}, "BEFL-2020": {}, "BEFL-2021": {}, "BEFL-2022": {}}

    with pytest.raises(
        ValueError,
        match=expected_error,
    ):
        _ = conf._prepare_train_label_infos(
            labelpolygons_pattern,
            labellocations_pattern,
            label_datasources=label_datasources,
            image_layers=image_layers,
        )


def test_prepare_train_label_infos_invalid_layer():
    labeldata_pattern = TestData.dir / "footballfields_{image_layer}_data.gpkg"
    labellocation_pattern = TestData.dir / "footballfields_{image_layer}_locations.gpkg"
    image_layers = {"BEFL-2019": {}}

    with pytest.raises(
        ValueError, match=re.escape("invalid image_layer in LabelInfo(image_layer=")
    ):
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


def test_tmpdir():
    tmpdir = conf._set_tmp_dir("orthoseg")

    assert tmpdir.exists()
    assert tmpdir == Path(tempfile.gettempdir())

# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""
import os
from pathlib import Path
import sys
from typing import List, Optional, Union

import pytest
from shapely import geometry as sh_geom

# Make hdf5 version warning non-blocking
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

import geopandas as gpd
import geofileops as gfo

# Add path so the local orthoseg packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.helpers import config_helper
from orthoseg.lib import prepare_traindatasets as prep_traindata
from orthoseg.lib.prepare_traindatasets import ValidationError
from tests.test_helper import TestData


# Helper functions to prepare test data
# -------------------------------------


def _prepare_locations_file(
    tmp_path, locations: Optional[Union[dict, gpd.GeoDataFrame]]
) -> Path:
    locations_path = tmp_path / "locations.gpkg"
    if locations is None:
        locations = TestData.locations_gdf
    if isinstance(locations, dict):
        locations = gpd.GeoDataFrame(locations, crs="EPSG:31370")  # type: ignore

    gfo.to_file(locations, locations_path)
    return locations_path


def _prepare_polygons_file(
    tmp_path, polygons: Optional[Union[dict, gpd.GeoDataFrame]]
) -> Path:
    polygons_path = tmp_path / "polygons.gpkg"
    if polygons is None:
        polygons = TestData.polygons_gdf
    if isinstance(polygons, dict):
        polygons = gpd.GeoDataFrame(polygons, crs="EPSG:31370")  # type: ignore

    gfo.to_file(polygons, polygons_path)
    return polygons_path


def _prepare_labelinfos(
    tmp_path,
    locations: Optional[Union[dict, gpd.GeoDataFrame]] = None,
    polygons: Optional[Union[dict, gpd.GeoDataFrame]] = None,
) -> List[prep_traindata.LabelInfo]:
    locations_path = _prepare_locations_file(tmp_path, locations)
    polygons_path = _prepare_polygons_file(tmp_path, polygons)

    label_info = prep_traindata.LabelInfo(
        locations_path=locations_path,
        polygons_path=polygons_path,
        image_layer="BEFL-2019",
    )
    return [label_info]


# Actual tests
# ------------


@pytest.mark.parametrize(
    "geometry, traindata_type, expected_len_locations",
    [
        [TestData.location, "train", 4],
        [None, "validation", 2],
        [sh_geom.Polygon(), "validation", 2],
    ],
)
def test_prepare_labeldata_locations(geometry, traindata_type, expected_len_locations):
    # Prepare test data
    # Add some extra data to make sure it also works for multiple rows
    # Prepare test data, by wrapping the parametrized invalid test data by proper data
    # to make sure the checks work on multiple rows,...
    locations_gdf = gpd.GeoDataFrame(
        data={
            "geometry": [TestData.location, geometry, geometry, TestData.location],
            "traindata_type": ["train", traindata_type, traindata_type, "validation"],
            "path": "/tmp/locations.gdf",
        },
        crs="EPSG:31370",  # type: ignore
    )
    label_info = prep_traindata.LabelInfo(
        locations_path=Path("locations"),
        polygons_path=Path("polygons"),
        image_layer="BEFL-2019",
        locations_gdf=locations_gdf,
        polygons_gdf=TestData.polygons_gdf,
    )

    # Test!
    labeldata = prep_traindata.prepare_labeldata(
        label_infos=[label_info],
        classes=TestData.classes,
        labelname_column="classname",
        image_pixel_x_size=TestData.image_pixel_x_size,
        image_pixel_y_size=TestData.image_pixel_y_size,
        image_pixel_width=TestData.image_pixel_width,
        image_pixel_height=TestData.image_pixel_height,
    )
    labellocations_gdf, labelpolygons_gdf = labeldata[0]

    assert len(labelpolygons_gdf) == len(TestData.polygons_gdf)
    assert len(labellocations_gdf) == expected_len_locations


@pytest.mark.parametrize(
    "expected_error, geometry, traindata_type",
    [
        ["Invalid geometry ", TestData.location_invalid, "train"],
        ["Invalid traindata_type ", TestData.location, "invalid_traindata_type"],
        ["Location geometry too small ", TestData.location.buffer(-5), "train"],
    ],
)
def test_prepare_labeldata_locations_invalid(expected_error, geometry, traindata_type):
    # Prepare test data, by wrapping the parametrized invalid test data by proper data
    # to make sure the checks work on multiple rows,...
    locations_gdf = gpd.GeoDataFrame(
        data={
            "geometry": [TestData.polygon, geometry, geometry, TestData.polygon],
            "traindata_type": ["train", traindata_type, traindata_type, "validation"],
            "path": "/tmp/locations.gdf",
        },
        crs="EPSG:31370",  # type: ignore
    )
    label_info = prep_traindata.LabelInfo(
        locations_path=Path("locations"),
        polygons_path=Path("polygons"),
        image_layer="BEFL-2019",
        locations_gdf=locations_gdf,
        polygons_gdf=TestData.polygons_gdf,
    )

    # Test!
    with pytest.raises(ValidationError, match="Errors found in label data") as ex:
        _ = prep_traindata.prepare_labeldata(
            label_infos=[label_info],
            classes=TestData.classes,
            labelname_column="classname",
            image_pixel_x_size=TestData.image_pixel_x_size,
            image_pixel_y_size=TestData.image_pixel_y_size,
            image_pixel_width=TestData.image_pixel_width,
            image_pixel_height=TestData.image_pixel_height,
        )

    assert ex.value.errors is not None
    assert len(ex.value.errors) == 2
    assert ex.value.errors[0].startswith(expected_error)
    assert ex.value.errors[1].startswith(expected_error)


@pytest.mark.parametrize(
    "geometry, classname, expected_len_polygons",
    [
        [TestData.polygon, "testlabel1", 4],
        [None, "testlabel1", 2],
        [sh_geom.Polygon(), "testlabel1", 2],
    ],
)
def test_prepare_labeldata_polygons(geometry, classname, expected_len_polygons):
    # Prepare test data
    # Add some extra data to make sure it also works for multiple rows
    polygons_gdf = gpd.GeoDataFrame(
        data={
            "geometry": [TestData.polygon, geometry, geometry, TestData.polygon],
            "classname": ["testlabel1", classname, classname, "testlabel2"],
            "path": "/tmp/polygons.gdf",
        },
        crs="EPSG:31370",  # type: ignore
    )
    label_info = prep_traindata.LabelInfo(
        locations_path=Path("locations"),
        polygons_path=Path("polygons"),
        image_layer="BEFL-2019",
        locations_gdf=TestData.locations_gdf,
        polygons_gdf=polygons_gdf,
    )

    # Test!
    labeldata = prep_traindata.prepare_labeldata(
        label_infos=[label_info],
        classes=TestData.classes,
        labelname_column="classname",
        image_pixel_x_size=TestData.image_pixel_x_size,
        image_pixel_y_size=TestData.image_pixel_y_size,
        image_pixel_width=TestData.image_pixel_width,
        image_pixel_height=TestData.image_pixel_height,
    )
    labellocations_gdf, labelpolygons_gdf = labeldata[0]

    assert len(labelpolygons_gdf) == expected_len_polygons
    assert len(labellocations_gdf) == len(TestData.locations_gdf)


@pytest.mark.parametrize(
    "expected_error, geometry, classname",
    [
        ["Invalid geometry ", TestData.polygon_invalid, "testlabel1"],
        ["Invalid classname ", TestData.polygon, "unknown"],
    ],
)
def test_prepare_labeldata_polygons_invalid(expected_error, geometry, classname):
    # Prepare test data, by wrapping the parametrized invalid test data by proper data
    # to make sure the checks work on multiple rows,...
    polygons_gdf = gpd.GeoDataFrame(
        data={
            "geometry": [TestData.polygon, geometry, geometry, TestData.polygon],
            "classname": ["testlabel1", classname, classname, "testlabel2"],
            "path": "/tmp/polygons.gdf",
        },
        crs="EPSG:31370",  # type: ignore
    )
    label_info = prep_traindata.LabelInfo(
        locations_path=Path("locations"),
        polygons_path=Path("polygons"),
        image_layer="BEFL-2019",
        locations_gdf=TestData.locations_gdf,
        polygons_gdf=polygons_gdf,
    )

    # Test!
    with pytest.raises(ValidationError, match="Errors found in label data") as ex:
        _ = prep_traindata.prepare_labeldata(
            label_infos=[label_info],
            classes=TestData.classes,
            labelname_column="classname",
            image_pixel_x_size=TestData.image_pixel_x_size,
            image_pixel_y_size=TestData.image_pixel_y_size,
            image_pixel_width=TestData.image_pixel_width,
            image_pixel_height=TestData.image_pixel_height,
        )

    assert ex.value.errors is not None
    assert len(ex.value.errors) == 2
    assert ex.value.errors[0].startswith(expected_error)
    assert ex.value.errors[1].startswith(expected_error)


def test_prepare_labeldata_polygons_columnname_backw_compat(tmp_path):
    # Test bacwards compatibility for old label column name
    # Prepare test data
    polygons_gdf = gpd.GeoDataFrame(
        data={
            "geometry": [TestData.polygon, TestData.polygon],
            "label_name": ["testlabel1", "testlabel2"],
            "path": "/tmp/polygons.gdf",
        },
        crs="EPSG:31370",  # type: ignore
    )
    label_info = prep_traindata.LabelInfo(
        locations_path=Path("locations"),
        polygons_path=Path("polygons"),
        image_layer="BEFL-2019",
        locations_gdf=TestData.locations_gdf,
        polygons_gdf=polygons_gdf,
    )

    # Test!
    labeldata = prep_traindata.prepare_labeldata(
        label_infos=[label_info],
        classes=TestData.classes,
        labelname_column="test_columnname",
        image_pixel_x_size=TestData.image_pixel_x_size,
        image_pixel_y_size=TestData.image_pixel_y_size,
        image_pixel_width=TestData.image_pixel_width,
        image_pixel_height=TestData.image_pixel_height,
    )
    locations_gdf, polygons_to_burn_gdf = labeldata[0]

    assert len(polygons_to_burn_gdf) == 2


def test_prepare_traindata_full(tmp_path):
    # Prepare test data
    classes = TestData.classes
    image_layers_config_path = TestData.testprojects_dir / "imagelayers.ini"
    image_layers = config_helper.read_layer_config(image_layers_config_path)
    label_infos = _prepare_labelinfos(tmp_path)

    # Test with the default data...
    training_dir = tmp_path / "training_dir"
    training_dir, traindata_id = prep_traindata.prepare_traindatasets(
        label_infos=label_infos,
        classes=classes,
        image_layers=image_layers,
        training_dir=training_dir,
        image_pixel_x_size=TestData.image_pixel_x_size,
        image_pixel_y_size=TestData.image_pixel_y_size,
        image_pixel_width=TestData.image_pixel_width,
        image_pixel_height=TestData.image_pixel_height,
    )

    assert training_dir.exists() is True

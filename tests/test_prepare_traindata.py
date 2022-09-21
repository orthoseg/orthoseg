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


def test_default(tmp_path):
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
        image_pixel_x_size=0.25,
        image_pixel_y_size=0.25,
        image_pixel_width=512,
        image_pixel_height=512,
    )

    assert training_dir.exists() is True


def test_label_name_column_backw_compat(tmp_path):
    # Test bacwards compatibility for old label column name
    # Prepare test data
    classes = TestData.classes
    polygons = {
        "geometry": [
            sh_geom.box(150030, 170030, 150060, 170060),
            sh_geom.box(150030, 180030, 150060, 180060),
        ],
        "label_name": ["testlabel", "testlabel"],
    }
    label_infos = _prepare_labelinfos(tmp_path, polygons=polygons)

    # Test!
    locations_gdf, polygons_to_burn_gdf = prep_traindata.read_labeldata(
        label_infos=label_infos,
        classes=classes,
        labelname_column="test_columnname"
    )

    assert len(polygons_to_burn_gdf) == 2


def test_invalid_labelnames(tmp_path):
    # Test with None and incorrect label names
    # Prepare test data
    classes = TestData.classes
    polygons = {
        "geometry": [
            sh_geom.box(150030, 170030, 150060, 170060),
            sh_geom.box(150030, 180030, 150060, 180060),
            sh_geom.box(150030, 180030, 150060, 180060),
        ],
        "classname": ["testlabelwrong", None, "testlabel"],
    }
    label_infos = _prepare_labelinfos(tmp_path, polygons=polygons)

    # Test!
    with pytest.raises(
        ValidationError,
        match="Errors found in label data",
    ) as ex:
        _ = prep_traindata.read_labeldata(
            label_infos=label_infos,
            classes=classes,
        )

    assert ex.value.errors is not None
    assert len(ex.value.errors) == 2
    assert ex.value.errors[0].startswith("Invalid classname in ")


def test_invalid_geoms_polygons(tmp_path):
    # Test with None and incorrect label names
    # Prepare test data
    classes = TestData.classes
    invalid_poly = sh_geom.Polygon(
        [
            (150000, 190000),
            (150128, 190000),
            (150128, 190128),
            (150000, 190128),
            (150002, 190130),
            (150000, 190000),
        ]
    )
    polygons = {
        "geometry": [
            sh_geom.box(150030, 170030, 150060, 170060),
            invalid_poly,
            invalid_poly,
        ],
        "classname": ["testlabel", "testlabel", "testlabel"],
    }
    label_infos = _prepare_labelinfos(tmp_path, polygons=polygons)

    # Test!
    with pytest.raises(ValidationError, match="Errors found in label data") as ex:
        _, _ = prep_traindata.read_labeldata(
            label_infos=label_infos,
            classes=classes,
        )

    assert ex.value.errors is not None
    assert len(ex.value.errors) == 2
    assert ex.value.errors[0].startswith("Invalid geom")

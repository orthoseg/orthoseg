# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""
import os
from pathlib import Path
import sys
from typing import List, Optional

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
from tests import test_helper


# Default test data
# -----------------

_test_classes = {
    "background": {
        "labelnames": ["ignore_for_train", "background"],
        "weight": 1,
        "burn_value": 0,
    },
    "test": {"labelnames": ["testlabel"], "weight": 1, "burn_value": 1},
}

_locations_data = {
    "geometry": [
        sh_geom.box(150000, 170000, 150128, 170128),
        sh_geom.box(150000, 180000, 150128, 180128),
        sh_geom.box(150000, 190000, 150128, 190128),
    ],
    "traindata_type": ["train", "train", "validation"],
}

_polygons_data = {
    "geometry": [
        sh_geom.box(150030, 170030, 150060, 170060),
        sh_geom.box(150030, 180030, 150060, 180060),
    ],
    "label_name": ["testlabel", "testlabel"],
}


# Helper functions to prepare test data
# -------------------------------------


def _prepare_locations_file(tmp_path, locations_data: Optional[dict]) -> Path:
    locations_path = tmp_path / "locations.gpkg"
    if locations_data is None:
        locations_data = _locations_data

    locations_gdf = gpd.GeoDataFrame(locations_data, crs="epsg:31370")  # type: ignore
    gfo.to_file(locations_gdf, locations_path)
    return locations_path


def _prepare_polygons_file(tmp_path, polygons_data: Optional[dict]) -> Path:
    polygons_path = tmp_path / "polygons.gpkg"
    if polygons_data is None:
        polygons_data = _polygons_data

    polygons_gdf = gpd.GeoDataFrame(polygons_data, crs="epsg:31370")  # type: ignore
    gfo.to_file(polygons_gdf, polygons_path)
    return polygons_path


def _prepare_labelinfos(
    tmp_path,
    locations_data: Optional[dict] = None,
    polygons_data: Optional[dict] = None,
) -> List[prep_traindata.LabelInfo]:
    locations_path = _prepare_locations_file(tmp_path, locations_data)
    polygons_path = _prepare_polygons_file(tmp_path, polygons_data)

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
    classes = _test_classes
    image_layers_config_path = test_helper.get_testprojects_dir() / "imagelayers.ini"
    image_layers = config_helper.read_layer_config(image_layers_config_path)
    label_infos = _prepare_labelinfos(tmp_path)

    # Test with the default data...
    training_dir = tmp_path / "training_dir"
    training_dir, traindata_id = prep_traindata.prepare_traindatasets(
        label_infos=label_infos,
        classes=classes,
        image_layers=image_layers,
        training_dir=training_dir,
        labelname_column="label_name",
        image_pixel_x_size=0.25,
        image_pixel_y_size=0.25,
        image_pixel_width=512,
        image_pixel_height=512,
    )

    assert training_dir.exists() is True


def test_invalid_labelnames(tmp_path):
    # Test with None and incorrect label names
    # Prepare test data
    classes = _test_classes
    polygons_data = {
        "geometry": [
            sh_geom.box(150030, 170030, 150060, 170060),
            sh_geom.box(150030, 180030, 150060, 180060),
        ],
        "label_name": ["testlabelwrong", None],
    }
    label_infos = _prepare_labelinfos(tmp_path, polygons_data=polygons_data)

    # Test!
    with pytest.raises(
        ValueError, match=r"Unknown labelnames \(not in config\) were found in"
    ):
        _, _ = prep_traindata.read_labeldata(
            label_infos=label_infos,
            classes=classes,
            labelname_column="label_name",
        )


def test_invalid_geoms_polygons(tmp_path):
    # Test with None and incorrect label names
    # Prepare test data
    classes = _test_classes
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
    polygons_data = {
        "geometry": [
            sh_geom.box(150030, 170030, 150060, 170060),
            invalid_poly,
            invalid_poly,
        ],
        "label_name": ["testlabel", "testlabel", "testlabel"],
    }
    label_infos = _prepare_labelinfos(tmp_path, polygons_data=polygons_data)

    # Test!
    with pytest.raises(ValidationError, match=r"Errors found in label data") as ex:
        _, _ = prep_traindata.read_labeldata(
            label_infos=label_infos,
            classes=classes,
            labelname_column="label_name",
        )
    print(f"ValidationError: {ex}")
    assert ex.value.errors is not None
    assert len(ex.value.errors) == 2
    assert ex.value.errors[0].startswith("Invalid geom")

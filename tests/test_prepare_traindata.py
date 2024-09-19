"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""

import filecmp
import math
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import geofileops as gfo
import geopandas as gpd
import pytest
from PIL import Image
from shapely import geometry as sh_geom

from tests import test_helper

# Make hdf5 version warning non-blocking
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

from orthoseg.helpers import config_helper
from orthoseg.lib import prepare_traindatasets as prep_traindata
from orthoseg.lib.prepare_traindatasets import ValidationError
from tests.test_helper import TestData


@pytest.fixture
def empty_image(tmp_path):
    path = tmp_path / "empty_image.png"
    if not path.exists():
        img = Image.new(
            mode="RGB",
            size=(TestData.image_pixel_width, TestData.image_pixel_height),
        )
        img.save(path, "png")

    return path


def _prepare_locations_file(
    tmp_path, locations: Optional[Union[dict, gpd.GeoDataFrame]]
) -> Path:
    locations_path = tmp_path / "locations.gpkg"
    if locations is None:
        locations = TestData.locations_gdf
    if isinstance(locations, dict):
        locations = gpd.GeoDataFrame(locations, crs="EPSG:31370")

    gfo.to_file(locations, locations_path)
    return locations_path


def _prepare_polygons_file(
    tmp_path, polygons: Optional[Union[dict, gpd.GeoDataFrame]]
) -> Path:
    polygons_path = tmp_path / "polygons.gpkg"
    if polygons is None:
        polygons = TestData.polygons_gdf
    if isinstance(polygons, dict):
        polygons = gpd.GeoDataFrame(polygons, crs="EPSG:31370")

    gfo.to_file(polygons, polygons_path)
    return polygons_path


def _prepare_labelinfos(
    tmp_path,
    locations: Optional[Union[dict, gpd.GeoDataFrame]] = None,
    polygons: Optional[Union[dict, gpd.GeoDataFrame]] = None,
) -> list[prep_traindata.LabelInfo]:
    locations_path = _prepare_locations_file(tmp_path, locations)
    polygons_path = _prepare_polygons_file(tmp_path, polygons)

    label_info = prep_traindata.LabelInfo(
        locations_path=locations_path,
        polygons_path=polygons_path,
        image_layer="BEFL-2019",
    )
    return [label_info]


@pytest.mark.parametrize(
    "geometry, traindata_type, expected_len_locations",
    [
        (TestData.location, "train", 4),
        (None, "validation", 2),
        (sh_geom.Polygon(), "validation", 2),
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
        crs="EPSG:31370",
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
    "expected_errors, geometries, traindata_types",
    [
        (["Invalid geometry "] * 4, [TestData.location_invalid] * 4, None),
        (
            "Invalid traindata_type ",
            None,
            ["train", "train", "validation", "invalid_traindata_type"],
        ),
        (
            ["Location geometry skewed or too small "] * 4,
            [TestData.location.buffer(-5)] * 4,
            None,
        ),
        ("No labellocations with traindata_type == 'validation' ", None, ["train"] * 4),
        ("No labellocations with traindata_type == 'train' ", None, ["validation"] * 4),
    ],
)
def test_prepare_labeldata_locations_invalid(
    expected_errors, geometries, traindata_types
):
    # Prepare test data, by wrapping the parametrized invalid test data by proper data
    # to make sure the checks work on multiple rows,...
    if isinstance(expected_errors, str):
        expected_errors = [expected_errors]
    if geometries is None:
        geometries = [TestData.location] * 4
    if traindata_types is None:
        traindata_types = ["train", "train", "train", "validation"]
    locations_gdf = gpd.GeoDataFrame(
        data={
            "geometry": geometries,
            "traindata_type": traindata_types,
            "path": "/tmp/locations.gdf",
        },
        crs="EPSG:31370",
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
    assert len(ex.value.errors) == len(expected_errors)
    for idx, expected_error in enumerate(expected_errors):
        assert ex.value.errors[idx].startswith(expected_error)


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
        crs="EPSG:31370",
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
        ("Invalid geometry ", TestData.polygon_invalid, "testlabel1"),
        ("Invalid classname ", TestData.polygon, "unknown"),
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
        crs="EPSG:31370",
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


def test_prepare_labeldata_polygons_columnname_backw_compat():
    # Test bacwards compatibility for old label column name
    # Prepare test data
    polygons_gdf = gpd.GeoDataFrame(
        data={
            "geometry": [TestData.polygon, TestData.polygon],
            "label_name": ["testlabel1", "testlabel2"],
            "path": "/tmp/polygons.gdf",
        },
        crs="EPSG:31370",
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


locations = {
    "loc1": (150100, 150100, "150100_150100_150228_150228_512_512_OMWRGB19VL.png"),
    "loc2": (150200, 150200, "150200_150200_150328_150328_512_512_OMWRGB19VL.png"),
    "loc3": (150300, 150300, "150300_150300_150428_150428_512_512_OMWRGB19VL.png"),
}


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt", reason="crashes on windows"
)
@pytest.mark.parametrize(
    "descr, prev_locations, new_locations",
    [
        ("reuse1_new1", ["loc1"], ["loc1", "loc2"]),
        ("remove1", ["loc1", "loc2", "loc3"], ["loc1", "loc2"]),
        ("reuse_all", ["loc1", "loc2"], ["loc1", "loc2"]),
        ("reuse_none", ["loc3"], ["loc1", "loc2"]),
    ],
)
def test_prepare_traindatasets_reuse_prev_images(
    tmp_path, empty_image, descr, prev_locations, new_locations
):
    """Test if images of previous training versions are reused properly."""
    # Prepare test data
    classes = TestData.classes
    image_layers_config_path = test_helper.sampleprojects_dir / "imagelayers.ini"
    image_layers = config_helper._read_layer_config(image_layers_config_path)
    label_infos = _prepare_labelinfos(tmp_path)
    training_dir = tmp_path / "training"

    # Prepare images to be reused from previous training version
    prev_training_dir = training_dir / "01"
    traindata_types = ["train", "validation", "test"]

    for traindata_type in traindata_types:
        dir = prev_training_dir / traindata_type
        (dir / "image").mkdir(parents=True, exist_ok=True)
        (dir / "mask").mkdir(parents=True, exist_ok=True)
        for previous_location in prev_locations:
            _, _, image = locations[previous_location]
            image_path = dir / "image" / image
            shutil.copy(empty_image, image_path)
            image_path.with_suffix(".pgw").touch()

    # Prepare gdf with new locations
    new_locations_list = []
    crs_width = TestData.image_pixel_width * TestData.image_pixel_x_size
    crs_height = TestData.image_pixel_height * TestData.image_pixel_y_size
    for new_location in new_locations:
        xmin, ymin, _ = locations[new_location]
        geometry = sh_geom.box(xmin, ymin, xmin + crs_width, ymin + crs_height)
        for traindata_type in traindata_types:
            new_locations_list.append(
                {
                    "geometry": geometry,
                    "traindata_type": traindata_type,
                    "path": "/tmp/locations.gdf",
                }
            )

    new_locations_gdf = gpd.GeoDataFrame(new_locations_list, crs=31370)

    # Run prepare traindatasets
    label_infos = _prepare_labelinfos(tmp_path=tmp_path, locations=new_locations_gdf)

    new_training_dir, _ = prep_traindata.prepare_traindatasets(
        label_infos=label_infos,
        classes=classes,
        image_layers=image_layers,
        training_dir=training_dir,
    )

    # Check if files exist in new output folder and if they have the correct size
    # if size = 1kb then copied else if size = 2kb downloaded
    for traindata_type in traindata_types:
        for loc in ["loc1", "loc2", "loc3"]:
            filename = locations[loc][2]
            prev_path = prev_training_dir / traindata_type / "image" / filename
            new_path = new_training_dir / traindata_type / "image" / filename

            if loc in new_locations and loc in prev_locations:
                # File is reused, so should be the same and small
                assert filecmp.cmp(prev_path, new_path, shallow=False) is True
                assert math.ceil(new_path.stat().st_size) <= 1024
            elif loc in new_locations:
                # File is new, so should exist and be larger than 1 kb
                assert new_path.exists()
                assert math.ceil(new_path.stat().st_size) > 1024
            elif loc in prev_locations:
                # File is only supposed to be in previous locations, not in new
                assert not new_path.exists()
                assert prev_path.exists()

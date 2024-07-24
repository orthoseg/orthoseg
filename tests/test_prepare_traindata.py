"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""

import filecmp
import math
import os
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


def test_prepare_labeldata_polygons_columnname_backw_compat(tmp_path):
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


image_1 = "150100_150100_150228_150228_512_512_OMWRGB19VL.png"
image_2 = "150200_150200_150328_150328_512_512_OMWRGB19VL.png"
image_3 = "150300_150300_150428_150428_512_512_OMWRGB19VL.png"


@pytest.mark.parametrize(
    "copy_images, match, mismatch, error, filesize_kb",
    [
        ["add_one", [image_1, image_2], [], [image_3], (1, 1, 2)],
        ["remove_one", [image_1], [], [image_2, image_3], (1, None, None)],
        ["copy_all", [image_1, image_2], [], [image_3], (1, 1, None)],
        ["copy_none", [], [], [image_1, image_2, image_3], (None, None, 2)],
    ],
    ids=[
        "add one new location",
        "remove one location",
        "all the same locations",
        "all different locations",
    ],
)
def test_prepare_traindatasets(
    tmp_path, copy_images: str, match, mismatch, error, filesize_kb
):
    # Prepare test data
    classes = TestData.classes
    image_layers_config_path = test_helper.sampleprojects_dir / "imagelayers.ini"
    image_layers = config_helper._read_layer_config(image_layers_config_path)
    label_infos = _prepare_labelinfos(tmp_path)

    training_dir = tmp_path / "training"
    previous_training_version_dir = training_dir / "01"
    traindata_types = ["train", "validation", "test"]

    for traindata_type in traindata_types:
        dir = previous_training_version_dir / traindata_type
        (dir / "image").mkdir(parents=True, exist_ok=True)
        (dir / "mask").mkdir(parents=True, exist_ok=True)
        for image in [image_1, image_2]:
            img = Image.new(
                mode="RGB",
                size=(TestData.image_pixel_width, TestData.image_pixel_height),
            )
            img.save(dir / "image" / image, "png")
            (dir / "image" / image.replace(".png", ".pgw")).touch()

    locations_gdf = create_locations_dataframe(copy_images=copy_images)

    # Run prepare traindatasets
    label_infos = _prepare_labelinfos(tmp_path=tmp_path, locations=locations_gdf)
    output_dir, _ = prep_traindata.prepare_traindatasets(
        label_infos=label_infos,
        classes=classes,
        image_layers=image_layers,
        training_dir=training_dir,
    )

    # Check if files exist in new output folder and if they have the correct size
    # if size = 1kb then copied else if size = 2kb downloaded
    for traindata_type in traindata_types:
        images_to_compare = [image_1, image_2, image_3]
        assert filecmp.cmpfiles(
            previous_training_version_dir / traindata_type / "image",
            output_dir / traindata_type / "image",
            images_to_compare,
            shallow=False,
        ) == (match, mismatch, error)

        for i, image in enumerate(images_to_compare):
            if filesize_kb[i] is None:
                assert not (output_dir / traindata_type / "image" / image).exists()
            else:
                assert (output_dir / traindata_type / "image" / image).exists()
                assert (
                    math.ceil(
                        os.path.getsize(output_dir / traindata_type / "image" / image)
                        / 1024
                    )
                    == filesize_kb[i]
                )


def create_location(
    crs_xmin: int,
    crs_ymin: int,
):
    return sh_geom.box(
        crs_xmin,
        crs_ymin,
        crs_xmin + (TestData.image_pixel_width * TestData.image_pixel_x_size),
        crs_ymin + (TestData.image_pixel_height * TestData.image_pixel_y_size),
    )


def create_locations_dataframe(copy_images: str) -> gpd.GeoDataFrame:
    location_1 = create_location(crs_xmin=150100, crs_ymin=150100)
    location_2 = create_location(crs_xmin=150200, crs_ymin=150200)
    location_3 = create_location(crs_xmin=150300, crs_ymin=150300)

    if copy_images == "add_one":
        locations_gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    location_1,
                    location_1,
                    location_1,
                    location_2,
                    location_2,
                    location_2,
                    location_3,
                    location_3,
                    location_3,
                ],
                "traindata_type": [
                    "train",
                    "validation",
                    "test",
                    "train",
                    "validation",
                    "test",
                    "train",
                    "validation",
                    "test",
                ],
                "path": "/tmp/locations.gdf",
            },
            crs="epsg:31370",
        )

    elif copy_images == "remove_one":
        locations_gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    location_1,
                    location_1,
                    location_1,
                ],
                "traindata_type": ["train", "validation", "test"],
                "path": "/tmp/locations.gdf",
            },
            crs="epsg:31370",
        )
    elif copy_images == "copy_all":
        locations_gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    location_1,
                    location_1,
                    location_1,
                    location_2,
                    location_2,
                    location_2,
                ],
                "traindata_type": [
                    "train",
                    "validation",
                    "test",
                    "train",
                    "validation",
                    "test",
                ],
                "path": "/tmp/locations.gdf",
            },
            crs="epsg:31370",
        )
    elif copy_images == "copy_none":
        locations_gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    location_3,
                    location_3,
                    location_3,
                ],
                "traindata_type": ["train", "validation", "test"],
                "path": "/tmp/locations.gdf",
            },
            crs="epsg:31370",
        )

    return locations_gdf

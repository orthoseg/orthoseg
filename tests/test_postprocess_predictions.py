"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""

import os
import shutil
from pathlib import Path

import geofileops as gfo
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import rasterio.transform as rio_transform
import shapely

from orthoseg.lib import postprocess_predictions as postp
from tests import test_helper
from tests.test_helper import TestData

# Make hdf5 version warning non-blocking
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"


def test_read_prediction_file():
    # Read + polygonize raster prediction file
    pred_raster_path = TestData.dir / "129568_185248_130592_186272_4096_4096_1_pred.tif"
    pred_raster_gdf = postp.read_prediction_file(pred_raster_path)
    # gfo.to_file(pred_raster_gdf, get_testdata_dir() / f"{pred_raster_path.stem}.gpkg")

    # Read the comparison file, that contains the result of the polygonize
    pred_comparison_path = TestData.dir / f"{pred_raster_path.stem}.gpkg"
    pred_comparison_gdf = gfo.read_file(pred_comparison_path)

    # Now compare they are the same
    assert pred_raster_gdf is not None
    assert len(pred_raster_gdf) == len(pred_comparison_gdf)


def test_clean_vectordata(tmpdir):
    temp_dir = Path(tmpdir)

    # Clean data
    input1_path = TestData.dir / "129568_184288_130592_185312_4096_4096_1_pred.gpkg"
    input2_path = TestData.dir / "129568_185248_130592_186272_4096_4096_1_pred.gpkg"
    input1_gdf = gfo.read_file(input1_path)
    input2_gdf = gfo.read_file(input2_path)
    input_gdf = pd.concat([input1_gdf, input2_gdf])
    assert input1_gdf.crs == input_gdf.crs
    input_path = temp_dir / "vector_input.gpkg"
    gfo.to_file(input_gdf, input_path)
    output_path = temp_dir / input_path.name
    postp.postprocess_predictions(
        input_path=input_path,
        output_path=output_path,
        dissolve=True,
        dissolve_tiles_path=None,
        force=True,
    )

    # Read result and check
    geoms_simpl_filepath = (
        output_path.parent / f"{output_path.stem}_dissolve{output_path.suffix}"
    )
    result_gdf = gfo.read_file(geoms_simpl_filepath)

    assert len(result_gdf) == 616


def create_project_dir(tmp_path: Path, subject: str) -> Path:
    testproject_dir = tmp_path / "orthoseg_test_postprocess"
    project_dir = testproject_dir / subject

    shutil.rmtree(path=testproject_dir, ignore_errors=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "project_template/projectfile.ini",
        dst=project_dir / f"{subject}.ini",
    )
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "imagelayers.ini",
        dst=testproject_dir / "imagelayers.ini",
    )
    return project_dir


def create_prediction_file(output_vector_dir: Path, subject: str) -> Path:
    imagelayer = "BEFL-2019"
    prediction_dir = output_vector_dir / imagelayer
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_vector_path = prediction_dir / f"{subject}_01_201_{imagelayer}.gpkg"

    predictions = {
        "classname": [subject] * 5,
        "geometry": [
            shapely.wkt.loads(
                "MULTIPOLYGON (((175056.375 176371.125, 175056.375 176370.875, 175056.625 176370.875, 175056.625 176371.125, 175056.375 176371.125)))"  # noqa: E501
            ),
            shapely.wkt.loads(
                "MULTIPOLYGON (((175054.875 176370.875, 175054.875 176370.625, 175055.375 176370.625, 175055.375 176370.875, 175054.875 176370.875)))"  # noqa: E501
            ),
            shapely.wkt.loads(
                "MULTIPOLYGON (((175047.125 176370.875, 175047.125 176370.125, 175047.375 176370.625, 175047.125 176370.875)))"  # noqa: E501
            ),
            shapely.wkt.loads(
                "MULTIPOLYGON (((175046.625 176370.125, 175046.625 176369.875, 175046.875 176369.875, 175046.875 176370.125, 175046.625 176370.125)))"  # noqa: E501
            ),
            shapely.wkt.loads(
                "MULTIPOLYGON (((175054.625 176369.625, 175054.625 176368.875, 175055.625 176369.375, 175054.625 176369.625)))"  # noqa: E501
            ),
        ],
    }
    predictions_gdf = gpd.GeoDataFrame(predictions, geometry="geometry", crs=31370)
    gfo.to_file(gdf=predictions_gdf, path=output_vector_path)
    return output_vector_path


def write_test_raster(
    output_path: Path,
    image_arr: np.ndarray,
    image_transform,
    image_crs: str = "EPSG:31370",
):
    with rio.open(
        output_path,
        "w",
        driver="GTiff",
        compress="lzw",
        height=image_arr.shape[0],
        width=image_arr.shape[1],
        count=1,
        dtype=rio.uint8,
        crs=image_crs,
        transform=image_transform,
    ) as ds:
        ds.write(image_arr, 1)


def _create_prediction_test_data(tmp_path: Path, image_transform):
    image_pred_arr = np.zeros((4, 4, 2), dtype=np.float32)
    image_pred_arr[1:3, 1:3, 1] = 1.0

    input_image_dir = tmp_path / "images"
    input_mask_dir = tmp_path / "masks"
    input_image_dir.mkdir(parents=True, exist_ok=True)
    input_mask_dir.mkdir(parents=True, exist_ok=True)

    input_image_filepath = input_image_dir / "input_tile.tif"
    input_mask_filepath = input_mask_dir / "input_tile.tif"

    image_arr = np.zeros((4, 4), dtype=np.uint8)
    write_test_raster(
        output_path=input_image_filepath,
        image_arr=image_arr,
        image_transform=image_transform,
    )

    mask_arr = np.zeros((4, 4), dtype=np.uint8)
    mask_arr[1:3, 1:3] = 1
    write_test_raster(
        output_path=input_mask_filepath,
        image_arr=mask_arr,
        image_transform=image_transform,
    )

    return (
        image_pred_arr,
        input_image_dir,
        input_mask_dir,
        input_image_filepath,
        mask_arr,
    )


@pytest.mark.parametrize("keep_original_file", [False, True])
@pytest.mark.parametrize("keep_intermediary_files", [False, True])
def test_postprocess_predictions(
    tmp_path: Path,
    keep_original_file: bool,
    keep_intermediary_files: bool,
):
    # Create test project
    subject = "test-subject"
    project_dir = create_project_dir(tmp_path=tmp_path, subject=subject)

    # Creating dummy files
    output_vector_dir = project_dir / "output_vector"
    output_vector_path = create_prediction_file(
        output_vector_dir=output_vector_dir, subject=subject
    )
    output_orig_path = (
        output_vector_path.parent / f"{output_vector_path.stem}_orig.gpkg"
    )
    output_dissolve_path = (
        output_vector_path.parent / f"{output_vector_path.stem}_dissolve.gpkg"
    )
    output_reclass_path = (
        output_vector_path.parent / f"{output_vector_path.stem}_reclass.gpkg"
    )

    # Go!
    postp.postprocess_predictions(
        input_path=output_vector_path,
        output_path=output_vector_path,
        keep_original_file=keep_original_file,
        keep_intermediary_files=keep_intermediary_files,
        dissolve=True,
        reclassify_to_neighbour_query="(area < 5)",
        force=False,
    )

    # Check results
    if not keep_original_file and not keep_intermediary_files:
        assert len(list(output_vector_path.parent.iterdir())) == 1
        assert output_vector_path.exists()
    if keep_original_file and not keep_intermediary_files:
        assert len(list(output_vector_path.parent.iterdir())) == 2
        assert output_vector_path.exists()
        assert output_orig_path.exists()
    if not keep_original_file and keep_intermediary_files:
        assert len(list(output_vector_path.parent.iterdir())) == 3
        assert output_vector_path.exists()
        assert output_dissolve_path.exists()
        assert output_reclass_path.exists()
    if keep_original_file and keep_intermediary_files:
        assert len(list(output_vector_path.parent.iterdir())) == 4
        assert output_vector_path.exists()
        assert output_orig_path.exists()
        assert output_dissolve_path.exists()
        assert output_reclass_path.exists()


def test_postprocess_predictions_output_style_added(tmp_path: Path):
    output_vector_dir = tmp_path / "output_vector"
    subject = "test-subject"
    output_vector_path = create_prediction_file(
        output_vector_dir=output_vector_dir, subject=subject
    )
    output_style_path = tmp_path / f"{subject}.qml"
    output_style_path.write_text("<qgis></qgis>", encoding="utf-8")

    postp.postprocess_predictions(
        input_path=output_vector_path,
        output_path=output_vector_path,
        dissolve=False,
        output_style_path=output_style_path,
    )

    styles = gfo.get_layerstyles(output_vector_path)
    assert len(styles) == 1
    assert styles.iloc[0]["styleName"] == output_style_path.stem
    layer_name = gfo.get_only_layer(output_vector_path)
    assert styles.iloc[0]["f_table_name"] == layer_name


def test_postprocess_predictions_output_style_missing(tmp_path: Path):
    output_vector_dir = tmp_path / "output_vector"
    subject = "test-subject"
    output_vector_path = create_prediction_file(
        output_vector_dir=output_vector_dir, subject=subject
    )
    output_style_path = tmp_path / f"{subject}.qml"

    with pytest.raises(FileNotFoundError, match="output_style_path doesn't exist"):
        postp.postprocess_predictions(
            input_path=output_vector_path,
            output_path=output_vector_path,
            dissolve=False,
            output_style_path=output_style_path,
        )


def test_postprocess_predictions_output_style_non_gpkg(tmp_path: Path):
    subject = "test-subject"
    output_vector_path = tmp_path / f"{subject}.geojson"
    gdf = gpd.GeoDataFrame(
        {
            "classname": ["footballfields"],
            "geometry": [
                shapely.wkt.loads(
                    "MULTIPOLYGON (((175056.375 176371.125, 175056.375 176370.875, 175056.625 176370.875, 175056.625 176371.125, 175056.375 176371.125)))"  # noqa: E501
                )
            ],
        },
        geometry="geometry",
        crs=31370,
    )
    gfo.to_file(gdf=gdf, path=output_vector_path)

    output_style_path = tmp_path / f"{subject}.qml"
    output_style_path.write_text("<qgis></qgis>", encoding="utf-8")

    with pytest.raises(ValueError, match="output is not a GeoPackage"):
        postp.postprocess_predictions(
            input_path=output_vector_path,
            output_path=output_vector_path,
            dissolve=False,
            output_style_path=output_style_path,
        )


@pytest.mark.parametrize(
    "max_similarity_to_save, expect_saved",
    [(0.999, False), (1.0, True)],
)
def test_postprocess_for_evaluation(
    tmp_path: Path, max_similarity_to_save: float, expect_saved: bool
):
    # Prepare test data
    image_transform = rio_transform.from_origin(175000, 176000, 0.25, 0.25)
    _, input_image_dir, input_mask_dir, input_image_filepath, mask_arr = (
        _create_prediction_test_data(tmp_path=tmp_path, image_transform=image_transform)
    )

    image_pred_uint8_cleaned_bin = mask_arr * 255
    image_pred_filepath = tmp_path / "input_tile_test-subject_pred.tif"
    write_test_raster(
        output_path=image_pred_filepath,
        image_arr=image_pred_uint8_cleaned_bin,
        image_transform=image_transform,
    )

    # Go!
    postp.postprocess_for_evaluation(
        image_filepath=input_image_filepath,
        image_crs="EPSG:31370",
        image_transform=image_transform,
        image_pred_filepath=image_pred_filepath,
        image_pred_uint8_cleaned_bin=image_pred_uint8_cleaned_bin,
        class_id=1,
        class_name="test-subject",
        nb_classes=2,
        output_dir=tmp_path,
        output_suffix="_test-subject",
        input_image_dir=input_image_dir,
        input_mask_dir=input_mask_dir,
        max_similarity_to_save=max_similarity_to_save,
        border_pixels_to_ignore=0,
    )

    # Check results
    eval_pred_paths = sorted(tmp_path.glob("*_input_tile_test-subject_pred.tif"))
    eval_input_paths = sorted(tmp_path.glob("*_input_tile_test-subject.tif"))
    eval_mask_paths = sorted(tmp_path.glob("*_input_tile_test-subject_mask.tif"))

    if expect_saved:
        assert len(eval_pred_paths) == 1
        assert len(eval_input_paths) == 1
        assert len(eval_mask_paths) == 1

        with rio.open(eval_input_paths[0]) as src_copy_ds:
            assert src_copy_ds.crs.to_string() == "EPSG:31370"
            assert src_copy_ds.transform == image_transform

        with rio.open(eval_pred_paths[0]) as pred_ds:
            pred_arr = pred_ds.read(1)
            assert pred_ds.crs.to_string() == "EPSG:31370"
            assert pred_ds.transform == image_transform

        with rio.open(eval_mask_paths[0]) as mask_eval_ds:
            mask_eval_arr = mask_eval_ds.read(1)
            assert mask_eval_ds.crs.to_string() == "EPSG:31370"
            assert mask_eval_ds.transform == image_transform

        assert pred_arr.shape == (4, 4)
        assert pred_arr[1, 1] == 255
        assert pred_arr[2, 2] == 255
        assert pred_arr[0, 0] == 0
        assert np.array_equal(mask_eval_arr, mask_arr * 255)
    else:
        assert len(eval_pred_paths) == 0
        assert len(eval_input_paths) == 0
        assert len(eval_mask_paths) == 0
        assert not image_pred_filepath.exists()

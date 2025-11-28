"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""

import os
import shutil
from pathlib import Path

import geofileops as gfo
import geopandas as gpd
import pandas as pd
import pytest
import shapely

from orthoseg.lib import postprocess_predictions as postp
from tests import test_helper

# Make hdf5 version warning non-blocking
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

from orthoseg.lib import postprocess_predictions as post_pred
from tests.test_helper import TestData


def test_read_prediction_file():
    # Read + polygonize raster prediction file
    pred_raster_path = TestData.dir / "129568_185248_130592_186272_4096_4096_1_pred.tif"
    pred_raster_gdf = post_pred.read_prediction_file(pred_raster_path)
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
    post_pred.postprocess_predictions(
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


def create_projects_dir(tmp_path: Path) -> Path:
    testproject_dir = tmp_path / "orthoseg_test_postprocess"
    project_dir = testproject_dir / "footballfields"

    shutil.rmtree(path=testproject_dir, ignore_errors=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "project_template/projectfile.ini",
        dst=project_dir / "footballfields.ini",
    )
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "imagelayers.ini",
        dst=testproject_dir / "imagelayers.ini",
    )
    return project_dir


def create_prediction_file(output_vector_dir: Path) -> Path:
    imagelayer = "BEFL-2019"
    prediction_dir = output_vector_dir / imagelayer
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_vector_path = prediction_dir / f"footballfields_01_201_{imagelayer}.gpkg"

    footballfields = {
        "classname": [
            "footballfields",
            "footballfields",
            "footballfields",
            "footballfields",
            "footballfields",
        ],
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
    footballfields_gdf = gpd.GeoDataFrame(footballfields, geometry="geometry")
    gfo.to_file(gdf=footballfields_gdf, path=output_vector_path)
    return output_vector_path


@pytest.mark.parametrize("keep_original_file", [False, True])
@pytest.mark.parametrize("keep_intermediary_files", [False, True])
def test_postprocess_predictions(
    tmp_path: Path,
    keep_original_file: bool,
    keep_intermediary_files: bool,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    output_vector_dir = project_dir / "output_vector"
    output_vector_path = create_prediction_file(output_vector_dir=output_vector_dir)
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

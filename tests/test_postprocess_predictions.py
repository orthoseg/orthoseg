# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""
import os
from pathlib import Path
import sys

# Make hdf5 version warning non-blocking
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

import geopandas as gpd
import geofileops as gfo

# Add path so the local orthoseg packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.lib import postprocess_predictions as post_pred
from tests import test_helper


def test_read_prediction_file():
    # Read + polygonize raster prediction file
    pred_raster_path = (
        test_helper.get_testdata_dir()
        / "129568_185248_130592_186272_4096_4096_1_pred.tif"
    )
    pred_raster_gdf = post_pred.read_prediction_file(
        pred_raster_path, border_pixels_to_ignore=128
    )
    # gfo.to_file(pred_raster_gdf, get_testdata_dir() / f"{pred_raster_path.stem}.gpkg")

    # Read the comparison file, that contains the result of the polygonize
    pred_comparison_path = (
        test_helper.get_testdata_dir() / f"{pred_raster_path.stem}.gpkg"
    )
    pred_comparison_gdf = gfo.read_file(pred_comparison_path)

    # Now compare they are the same
    assert pred_raster_gdf is not None
    assert len(pred_raster_gdf) == len(pred_comparison_gdf)


def test_clean_vectordata(tmpdir):
    temp_dir = Path(tmpdir)

    # Clean data
    input1_path = (
        test_helper.get_testdata_dir()
        / "129568_184288_130592_185312_4096_4096_1_pred.gpkg"
    )
    input2_path = (
        test_helper.get_testdata_dir()
        / "129568_185248_130592_186272_4096_4096_1_pred.gpkg"
    )
    input1_gdf = gfo.read_file(input1_path)
    input2_gdf = gfo.read_file(input2_path)
    input_gdf = gpd.GeoDataFrame(
        input1_gdf.append(input2_gdf), crs=input1_gdf.crs  # type: ignore
    )
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

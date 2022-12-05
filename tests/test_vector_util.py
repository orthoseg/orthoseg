# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""
import os
from pathlib import Path
import sys

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import geofileops as gfo
import pandas as pd
import pytest
from shapely import geometry as sh_geom

# Make hdf5 version warning non-blocking
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

# Add path so the local orthoseg packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.util import vector_util


@pytest.mark.parametrize("onborder_column_name", ["onborder_custom", "default_value"])
def test_calc_onborder(onborder_column_name: str):
    # Prepare test data
    border_bounds = (0.0, 0, 20, 20)
    expected_onborder_column_name = "expected_onborder"
    assert expected_onborder_column_name != onborder_column_name
    columns = ["descr", "geometry", expected_onborder_column_name]
    testdata = [
        ["Geom None", None, 0],
        ["Geom None", None, 0],
        ["to be removed", None, 0],
        ["Polygon EMPTY", sh_geom.Polygon(), 0],
        [
            "polygon on border",
            sh_geom.Polygon(shell=[(10, 10), (10, 20), (20, 20), (20, 10), (10, 10)]),
            1,
        ],
        [
            "polygon not on border",
            sh_geom.Polygon(shell=[(10, 10), (10, 11), (11, 11), (11, 10), (10, 10)]),
            0,
        ],
    ]
    df = pd.DataFrame(testdata, columns=columns)
    testdata_gdf = gpd.GeoDataFrame(df, geometry="geometry")  # type: ignore
    # Remove a row to test if the index is properly maintained
    testdata_gdf = testdata_gdf.drop([2], axis=0)
    assert isinstance(testdata_gdf, gpd.GeoDataFrame)

    if onborder_column_name == "default_value":
        result_gdf = vector_util.calc_onborder(
            testdata_gdf, border_bounds=border_bounds
        )
        onborder_column_name = "onborder"
    else:
        result_gdf = vector_util.calc_onborder(
            testdata_gdf,
            border_bounds=border_bounds,
            onborder_column_name=onborder_column_name,
        )

    # Compare result with expected result
    assert onborder_column_name in result_gdf.columns
    assert onborder_column_name not in testdata_gdf.columns
    result_gdf[expected_onborder_column_name] = result_gdf[onborder_column_name]
    result_gdf = result_gdf.drop(columns=[onborder_column_name])
    assert_geodataframe_equal(testdata_gdf, result_gdf)

    # Test empty GeoDataFrame
    # --------------------
    result_gdf = vector_util.calc_onborder(
        gpd.GeoDataFrame(),
        border_bounds=border_bounds,
        onborder_column_name=onborder_column_name,
    )
    assert result_gdf is not None
    assert onborder_column_name in result_gdf.columns
    assert len(result_gdf) == 0


def test_simplify_topo_orthoseg():
    # Test with several different types of geometries
    # -----------------------------------------------
    # Prepare test data
    columns = ["descr", "wkt"]
    testdata = [
        ["Geom None", None],
        ["Geom None", None],
        ["to be removed", None],
        ["Polygon EMPTY", "POLYGON(EMPTY)"],
        [
            "result=geometrycollection",
            "POLYGON ((151819.625 213605.125, 151820.625 213605.125, 151820.625 213604.875, 151820.875 213604.875, 151820.875 213604.625, 151821.875 213604.625, 151821.875 213604.375, 151823.125 213604.375, 151823.125 213604.125, 151825.875 213604.125, 151825.875 213603.875, 151828.125 213603.875, 151828.125 213603.625, 151829.375 213603.625, 151829.375 213603.375, 151830.625 213603.375, 151830.625 213603.125, 151831.875 213603.125, 151831.875 213602.875, 151832.625 213602.875, 151832.625 213602.625, 151833.375 213602.625, 151833.375 213602.375, 151834.125 213602.375, 151834.125 213602.125, 151834.375 213602.125, 151834.375 213601.875, 151834.625 213601.875, 151834.625 213601.125, 151834.875 213601.125, 151834.875 213599.875, 151834.625 213599.875, 151834.625 213597.625, 151834.375 213597.625, 151834.375 213594.875, 151834.125 213594.875, 151834.125 213592.125, 151834.375 213592.125, 151834.375 213591.125, 151834.125 213591.125, 151834.125 213590.125, 151833.875 213590.125, 151833.875 213589.375, 151833.625 213589.375, 151833.625 213589.125, 151833.375 213589.125, 151833.375 213588.625, 151833.125 213588.625, 151833.125 213587.875, 151832.875 213587.875, 151832.875 213587.125, 151832.625 213587.125, 151832.625 213585.125, 151832.375 213585.125, 151832.375 213584.625, 151832.125 213584.625, 151832.125 213584.125, 151831.875 213584.125, 151831.875 213583.875, 151831.625 213583.875, 151831.625 213583.125, 151831.375 213583.125, 151831.375 213582.625, 151831.125 213582.625, 151831.125 213582.125, 151830.875 213582.125, 151830.875 213581.875, 151829.375 213581.875, 151829.375 213582.125, 151829.125 213582.125, 151829.125 213582.375, 151828.875 213582.375, 151828.875 213583.125, 151828.625 213583.125, 151828.625 213583.375, 151828.375 213583.375, 151828.375 213583.125, 151827.875 213583.125, 151827.875 213582.875, 151827.625 213582.875, 151827.625 213582.625, 151827.375 213582.625, 151827.375 213582.375, 151826.875 213582.375, 151826.875 213582.125, 151826.625 213582.125, 151826.625 213581.875, 151826.375 213581.875, 151826.375 213581.375, 151826.625 213581.375, 151826.625 213581.125, 151826.875 213581.125, 151826.875 213580.875, 151827.375 213580.875, 151827.375 213581.125, 151829.375 213581.125, 151829.375 213580.875, 151829.875 213580.875, 151829.875 213580.625, 151830.125 213580.625, 151830.125 213580.375, 151830.875 213580.375, 151830.875 213577.875, 151830.625 213577.875, 151830.625 213577.375, 151829.625 213577.375, 151829.625 213577.875, 151829.375 213577.875, 151829.375 213578.125, 151828.875 213578.125, 151828.875 213578.625, 151829.125 213578.625, 151829.125 213579.125, 151828.375 213579.125, 151828.375 213579.375, 151827.875 213579.375, 151827.875 213579.125, 151827.625 213579.125, 151827.625 213578.625, 151827.375 213578.625, 151827.375 213578.375, 151827.125 213578.375, 151827.125 213577.125, 151827.375 213577.125, 151827.375 213576.875, 151827.625 213576.875, 151827.625 213576.625, 151827.875 213576.625, 151827.875 213576.375, 151828.125 213576.375, 151828.125 213576.125, 151828.875 213576.125, 151828.875 213575.875, 151829.125 213575.875, 151829.125 213575.125, 151829.375 213575.125, 151829.375 213574.875, 151828.875 213574.875, 151828.875 213574.625, 151828.375 213574.625, 151828.375 213574.375, 151827.875 213574.375, 151827.875 213574.125, 151827.625 213574.125, 151827.625 213573.875, 151827.375 213573.875, 151827.375 213572.375, 151826.375 213572.375, 151826.375 213572.625, 151825.875 213572.625, 151825.875 213572.875, 151825.625 213572.875, 151825.625 213573.625, 151825.375 213573.625, 151825.375 213574.375, 151824.625 213574.375, 151824.625 213574.125, 151824.125 213574.125, 151824.125 213573.875, 151823.875 213573.875, 151823.875 213574.125, 151823.625 213574.125, 151823.625 213574.375, 151823.125 213574.375, 151823.125 213574.125, 151822.875 213574.125, 151822.875 213573.875, 151822.375 213573.875, 151822.375 213574.125, 151821.125 213574.125, 151821.125 213574.375, 151820.875 213574.375, 151820.875 213574.625, 151819.625 213574.625, 151819.625 213574.875, 151819.125 213574.875, 151819.125 213575.375, 151819.375 213575.375, 151819.375 213575.875, 151819.125 213575.875, 151819.125 213576.125, 151818.375 213576.125, 151818.375 213575.875, 151817.875 213575.875, 151817.875 213575.625, 151817.125 213575.625, 151817.125 213575.375, 151816.625 213575.375, 151816.625 213575.125, 151816.125 213575.125, 151816.125 213575.375, 151815.875 213575.375, 151815.875 213575.125, 151815.625 213575.125, 151815.625 213575.375, 151814.625 213575.375, 151814.625 213575.625, 151814.375 213575.625, 151814.375 213575.875, 151814.625 213575.875, 151814.625 213576.875, 151814.875 213576.875, 151814.875 213578.625, 151815.125 213578.625, 151815.125 213579.375, 151815.375 213579.375, 151815.375 213581.125, 151815.625 213581.125, 151815.625 213583.625, 151815.875 213583.625, 151815.875 213585.125, 151816.125 213585.125, 151816.125 213586.875, 151816.375 213586.875, 151816.375 213588.125, 151816.625 213588.125, 151816.625 213589.375, 151816.875 213589.375, 151816.875 213590.625, 151817.125 213590.625, 151817.125 213592.125, 151817.375 213592.125, 151817.375 213594.125, 151817.625 213594.125, 151817.625 213595.625, 151817.875 213595.625, 151817.875 213597.125, 151818.125 213597.125, 151818.125 213598.375, 151818.375 213598.375, 151818.375 213599.375, 151818.625 213599.375, 151818.625 213599.875, 151818.875 213599.875, 151818.875 213600.375, 151819.125 213600.375, 151819.125 213600.625, 151819.625 213600.625, 151819.625 213600.875, 151819.875 213600.875, 151819.875 213601.375, 151819.625 213601.375, 151819.625 213602.125, 151819.375 213602.125, 151819.375 213602.375, 151819.125 213602.375, 151819.125 213603.625, 151819.375 213603.625, 151819.375 213604.875, 151819.625 213604.875, 151819.625 213605.125), (151828.125 213584.625, 151828.125 213584.375, 151827.625 213584.375, 151827.625 213583.875, 151827.875 213583.875, 151827.875 213583.625, 151828.875 213583.625, 151828.875 213584.375, 151828.375 213584.375, 151828.375 213584.625, 151828.125 213584.625), (151826.375 213573.875, 151826.375 213573.625, 151826.625 213573.625, 151826.625 213573.875, 151826.375 213573.875))",
        ],
        [
            "result=linestring",
            "POLYGON ((6.375 4.375, 6.375 3.875, 6.625 3.875, 6.625 4.375, "
            "6.375 4.375))",
        ],
        [
            "result=linestring",
            "POLYGON ((6.625 3.875, 6.625 3.375, 6.875 3.375, 6.875 3.125, "
            "7.375 3.125, 7.375 3.375, 7.125 3.375, 7.125 3.625, 6.875 3.625, "
            "6.875 3.875, 6.625 3.875))",
        ],
    ]
    df = pd.DataFrame(testdata, columns=columns)
    gs = gpd.GeoSeries.from_wkt(df["wkt"])
    df = df.drop(columns="wkt")
    testdata_gdf = gpd.GeoDataFrame(df, geometry=gs)  # type: ignore
    # Remove a row to test if the index is properly maintained in view_angles
    testdata_gdf = testdata_gdf.drop([2], axis=0)

    simplified_gdf = testdata_gdf.copy()
    assert isinstance(testdata_gdf.geometry, gpd.GeoSeries)
    simplified_gdf.geometry = vector_util.simplify_topo_orthoseg(
        testdata_gdf.geometry, tolerance=0.375, algorithm=gfo.SimplifyAlgorithm.LANG
    )
    for index, _, geometry in testdata_gdf.itertuples():
        if geometry is None or geometry.is_empty or index > 4:
            assert simplified_gdf["geometry"][index] is None
        else:
            if index == 4:
                assert geometry.type == simplified_gdf["geometry"][index].type

    # Test empty geoseries
    # --------------------
    result = vector_util.simplify_topo_orthoseg(
        gpd.GeoSeries(), tolerance=0.375, algorithm=gfo.SimplifyAlgorithm.LANG
    )
    assert result is not None
    assert len(result) == 0

    # Test with only empty geometries
    # -------------------------------
    # Prepare test data
    columns = ["descr", "wkt"]
    testdata = [
        ["Geom None", None],
        ["Geom None", None],
        ["to be removed", None],
        ["Polygon EMPTY", "POLYGON(EMPTY)"],
    ]
    df = pd.DataFrame(testdata, columns=columns)
    gs = gpd.GeoSeries.from_wkt(df["wkt"])
    df = df.drop(columns="wkt")
    testdata_gdf = gpd.GeoDataFrame(df, geometry=gs)  # type: ignore
    # Remove a row to test if the index is properly maintained in view_angles
    testdata_gdf = testdata_gdf.drop([2], axis=0)

    simplified_gdf = testdata_gdf.copy()
    assert isinstance(testdata_gdf.geometry, gpd.GeoSeries)
    simplified_gdf.geometry = vector_util.simplify_topo_orthoseg(
        testdata_gdf.geometry, tolerance=0.375, algorithm=gfo.SimplifyAlgorithm.LANG
    )
    assert_geoseries_equal(testdata_gdf.geometry, simplified_gdf.geometry)

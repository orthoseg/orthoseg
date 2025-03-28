"""
Tests for functionalities in vector_util.
"""

import os

import geopandas as gpd
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from shapely import geometry as sh_geom

# Make hdf5 version warning non-blocking
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

from orthoseg.util import vector_util


@pytest.mark.parametrize("onborder_column_name", ["onborder_custom", "default_value"])
def test_is_onborder(onborder_column_name: str):
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
    testdata_gdf = gpd.GeoDataFrame(df, geometry="geometry")
    # Remove a row to test if the index is properly maintained
    testdata_gdf = testdata_gdf.drop([2], axis=0)
    assert isinstance(testdata_gdf, gpd.GeoDataFrame)

    if onborder_column_name == "default_value":
        result_gdf = vector_util.is_onborder(testdata_gdf, border_bounds=border_bounds)
        onborder_column_name = "onborder"
    else:
        result_gdf = vector_util.is_onborder(
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
    result_gdf = vector_util.is_onborder(
        gpd.GeoDataFrame(),
        border_bounds=border_bounds,
        onborder_column_name=onborder_column_name,
    )
    assert result_gdf is not None
    assert len(result_gdf) == 0


def test_reclassify_neighbours():
    # Prepare test data
    reclassify_column = "class_name"
    query = "area <= 20 and onborder == 0"
    border_bounds = (0.0, 0, 20, 20)
    columns = ["test_id", "descr", "expected_result", "geometry", reclassify_column]
    testdata = [
        ["None1", "Geom None", "disappears", None, "class_1"],
        ["None2", "Geom None", "disappears", None, "class_1"],
        ["None3", "to be removed", "disappears", None, "class_1"],
        ["Empty1", "Polygon EMPTY", "disappears", sh_geom.Polygon(), "class_1"],
        [
            "poly1",
            "large polygon (25m²)",
            "smaller polygons are added to it",
            sh_geom.Polygon(shell=[(5, 5), (10, 5), (10, 10), (5, 10), (5, 5)]),
            "class_1_large",
        ],
        [
            "poly2",
            "medium (5m²), neighbour of polygon1, onborder",
            "is onborder, so isn't merged with other things",
            sh_geom.Polygon(shell=[(0, 5), (0, 6), (5, 6), (5, 5), (0, 5)]),
            "class_2_medium",
        ],
        [
            "poly3",
            "small (1m²), neighbour of poly1 and poly4, not onborder",
            "is merged with poly1",
            sh_geom.Polygon(shell=[(10, 5), (11, 5), (11, 6), (10, 6), (10, 5)]),
            "class_3_small",
        ],
        [
            "poly4",
            "small (1m²), neighbour of poly3, not onborder",
            "is merged with poly1 and poly3",
            sh_geom.Polygon(shell=[(11, 5), (12, 5), (12, 6), (11, 6), (11, 5)]),
            "class_4_small",
        ],
        [
            "poly5",
            "medium polygon (5m²), larger than neighbour poly6, not onborder",
            "is merged with poly6 but retains class",
            sh_geom.Polygon(shell=[(5, 15), (10, 15), (10, 16), (5, 16), (5, 15)]),
            "class_5_medium",
        ],
        [
            "poly6",
            "small (1m²), smaller than neighbour poly5, not onborder",
            "is merged with poly6 and gets its class",
            sh_geom.Polygon(shell=[(4, 15), (5, 15), (5, 16), (4, 16), (4, 15)]),
            "class_6_small",
        ],
    ]
    df = pd.DataFrame(testdata, columns=columns)
    testdata_gdf = gpd.GeoDataFrame(df, geometry="geometry")
    # Remove a row to test if the index is properly maintained
    testdata_gdf = testdata_gdf.drop([2], axis=0)

    expected_result_df = pd.DataFrame(
        [
            ["poly1, poly3, poly4", "class_1_large"],
            ["poly2", "class_2_medium"],
            ["poly5, poly6", "class_5_medium"],
        ],
        columns=["test_id", reclassify_column],
    )
    # Test with extra column, values are concatenated
    # -----------------------------------------------
    # reclassify only supported on gdf with only the reclassify column
    testdata_prepared_gdf = testdata_gdf[["test_id", "geometry", reclassify_column]]
    assert isinstance(testdata_prepared_gdf, gpd.GeoDataFrame)
    result_gdf = vector_util.reclassify_neighbours(
        testdata_prepared_gdf, reclassify_column, query, border_bounds
    )
    assert result_gdf is not None
    result_df = pd.DataFrame(result_gdf[["test_id", reclassify_column]])
    assert_frame_equal(result_df, expected_result_df)

    # Test with no extra column
    # -------------------------
    expected_result_df = expected_result_df[[reclassify_column]]

    # reclassify only supported on gdf with only the reclassify column
    testdata_prepared_gdf = testdata_gdf[["geometry", reclassify_column]]
    assert isinstance(testdata_prepared_gdf, gpd.GeoDataFrame)
    result_gdf = vector_util.reclassify_neighbours(
        testdata_prepared_gdf, reclassify_column, query, border_bounds
    )
    assert result_gdf is not None
    result_df = pd.DataFrame(result_gdf[[reclassify_column]])
    assert_frame_equal(result_df, expected_result_df)

"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""

import os

import geopandas as gpd
import geofileops as gfo
import pandas as pd
from shapely import geometry as sh_geom

# Make hdf5 version warning non-blocking
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

from orthoseg.helpers import vectorfile_helper


def test_reclassify_neighbours(tmp_path):
    # Prepare test data
    reclassify_column = "class_name"
    query = "area <= 20"
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
            "neighbour of polygon1, small (1m²), not onborder",
            "is merged with poly1",
            sh_geom.Polygon(shell=[(10, 5), (11, 5), (11, 6), (10, 6), (10, 5)]),
            "class_3_small",
        ],
        [
            "poly4",
            "neighbour of poly3, small (1m²), not onborder",
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
    testdata_path = tmp_path / "testdata.gpkg"
    gfo.to_file(testdata_gdf, testdata_path)

    # Test
    result_path = tmp_path / "result.gpkg"
    vectorfile_helper.reclassify_neighbours(
        input_path=testdata_path,
        reclassify_column=reclassify_column,
        query=query,
        output_path=result_path,
    )
    result_gdf = gfo.read_file(result_path)
    assert result_gdf is not None
    assert len(result_gdf) == 2
    assert reclassify_column in result_gdf.columns
    assert "area" in result_gdf.columns
    assert "nbcoords" in result_gdf.columns

"""
Modile with generic Utility functions for vector manipulations.
"""

import logging
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
import shapely

logger = logging.getLogger(__name__)


def is_onborder(
    gdf: gpd.GeoDataFrame,
    border_bounds: Tuple[float, float, float, float],
    onborder_column_name: str = "onborder",
) -> gpd.GeoDataFrame:
    """
    Add/update the is_onborder column to the GeoDataFrame.

    The column containes these values:
        * 0 if the polygon isn't on the border and
        * 1 if it is.

    Args:
        gdf: input GeoDataFrame
        border_bounds: the bounds (tupple with (xmin, ymin, xmax, ymax)
                       to check against to determine onborder
        onborder_column_name: the column name of the onborder column
    """
    if gdf is None or len(gdf.index) == 0:
        return gdf

    result_gdf = gdf.copy()
    result_gdf[onborder_column_name] = 0

    # Check
    for i, geom_row in result_gdf.iterrows():
        if geom_row["geometry"] is not None and not geom_row["geometry"].is_empty:
            # Check if the geom is on the border of the tile
            geom_bounds = geom_row["geometry"].bounds
            if (
                geom_bounds[0] <= border_bounds[0]
                or geom_bounds[1] <= border_bounds[1]
                or geom_bounds[2] >= border_bounds[2]
                or geom_bounds[3] >= border_bounds[3]
            ):
                result_gdf.loc[i, onborder_column_name] = 1

    assert isinstance(result_gdf, gpd.GeoDataFrame)
    return result_gdf


def is_valid_reason(geoseries: gpd.GeoSeries) -> pd.Series:
    """
    Get the reason for invalidity of all geometries in the GeoSeries.

    Args:
        geoseries (gpd.GeoSeries): the GeoSeries to check.

    Returns:
        pd.Series: _description_
    """
    # Get result and keep geoseries indexes
    return pd.Series(
        data=shapely.is_valid_reason(geoseries.array.data),
        index=geoseries.index,
    )


def reclassify_neighbours(
    gdf: gpd.GeoDataFrame,
    reclassify_column: str,
    query: str,
    border_bounds: Optional[Tuple[float, float, float, float]],
    class_background: str = "background",
) -> gpd.GeoDataFrame:
    """
    Reclassify features to the class of neighbouring features.

    For features that comply to the query, if they have a neighbour (touch/overlap),
    change their classname to that of the neighbour with the longest intersection and
    dissolve to merge them.

    The query should follow the syntax of pandas.query queries and can us the following
    fields:
        - onborder (0 or 1): does the feature touch/overlap the border_bounds. If this
          field is used in the query, the border_bounds parameter is mandatory.
        - area (float): the area of the geometry.
        - perimeter (float): the perimeter of the geometry.

    Args:
        gdf (gpd.GeoDataFrame): input
        reclassify_column (str): column to reclassify.
        query (str): th query to find the features to reclassify.
        border_bounds (Optional[Tuple[float, float, float, float]]): the bounds of the
            border to use for th onborder field in the query.
        class_background (str, optional): the classname to treat as background.
            Defaults to "background".
        aggfunc ()

    Raises:
        ValueError: raised if incompatible parameters are passed.

    Returns:
        gpd.GeoDataFrame: the reclassified result
    """
    # Init column info
    columns_orig = list(gdf.columns)

    # Init query + needed data
    query = query.replace("\n", " ")

    def _add_needed_columns(
        inner_gdf: gpd.GeoDataFrame,
        inner_query: str,
        inner_border_bounds: Optional[Tuple[float, float, float, float]],
    ) -> gpd.GeoDataFrame:
        gdf_result = inner_gdf.copy()

        # Always add area to know which neighbour is largest
        gdf_result["area"] = gdf_result.geometry.area
        if "perimeter" in inner_query:
            gdf_result["perimeter"] = gdf_result.geometry.length
        if "onborder" in inner_query:
            if inner_border_bounds is None:
                raise ValueError(
                    "query contains onborder, but border_bounds parameter is None"
                )
            gdf_result = is_onborder(gdf_result, inner_border_bounds)

        # Return result
        assert isinstance(gdf_result, gpd.GeoDataFrame)
        return gdf_result

    result_gdf = _add_needed_columns(gdf, query, border_bounds)

    # First remove background polygons that don't match the reclassify query
    nobackground_query = (
        f"{reclassify_column} != '{class_background}' or "
        f"({reclassify_column} == '{class_background}' and "
        f"({query}))"
    )
    # Use copy() to avoid view-versus-copy warnings
    result_gdf = result_gdf.query(nobackground_query).copy()

    # Keep looking for polygons that comply with query and give the same class
    # as neighbour till no changes can be made anymore.
    # Stop after 5 iterations to be sure never to end up in endless loop
    reclassify_max = 5
    result_gdf["no_neighbours"] = 0
    query = f"no_neighbours == 0 and ({query})"
    for reclassify_counter in range(reclassify_max):
        # Loop till no features were changed anymore
        if reclassify_counter > 0:
            assert isinstance(result_gdf, gpd.GeoDataFrame)
            result_gdf = _add_needed_columns(result_gdf, query, border_bounds)

        # Use copy() to avoid view-versus-copy warnings
        result_reclass_gdf = result_gdf.query(query).copy()
        if len(result_reclass_gdf) == 0:
            break
        # Order by area to treat the features found from small to large
        result_reclass_gdf = result_reclass_gdf.sort_values(by=["area"])

        for row in result_reclass_gdf.itertuples():
            # Find neighbours (query returns iloc's, not indexes!)
            neighbours_ilocs = result_gdf.geometry.sindex.query(
                row.geometry, predicate="intersects"
            ).tolist()
            if len(neighbours_ilocs) <= 1:
                result_gdf.loc[[row.Index], ["no_neighbours"]] = 1
                continue

            # Remove yourself
            row_loc = result_gdf.index.get_loc(row.Index)
            neighbours_ilocs.remove(row_loc)

            # Find neighbour with longest intersection
            neighbours_gdf = result_gdf.iloc[neighbours_ilocs]
            inters_geoseries = neighbours_gdf.geometry.intersection(row.geometry)
            max_length_index = inters_geoseries.length.idxmax()

            # Change the class of the smallest one to the oher's class
            class_curr = result_gdf.at[row.Index, reclassify_column]
            class_neighbour = result_gdf.at[max_length_index, reclassify_column]
            if class_curr != class_neighbour:
                # If the neighbour is not a reclass feature or if its area is larger
                # than the current feature, use its class
                if (
                    max_length_index not in result_reclass_gdf.index
                    or result_gdf.at[max_length_index, "area"] >= row.area
                ):
                    result_gdf.loc[[row.Index], [reclassify_column]] = class_neighbour
                else:
                    result_gdf.loc[[max_length_index], [reclassify_column]] = class_curr

        # Remove temp columns + dissolve
        result_gdf = result_gdf[columns_orig + ["no_neighbours"]]
        dissolve_columns = [reclassify_column, "no_neighbours"]

        # If there are extra columns, use aggfunc join to concatenate values
        assert isinstance(result_gdf, gpd.GeoDataFrame)
        if len(columns_orig) > 2:
            result_gdf = result_gdf.dissolve(
                by=dissolve_columns, as_index=False, aggfunc=", ".join
            )
        else:
            result_gdf = result_gdf.dissolve(by=dissolve_columns, as_index=False)

        result_gdf = result_gdf.explode(ignore_index=True)

    # Finalize + make sure there is no background in the output
    result_gdf = result_gdf[columns_orig]
    # Use copy() to avoid view-versus-copy warnings
    result_gdf = result_gdf.query(f"{reclassify_column} != '{class_background}'").copy()
    assert isinstance(result_gdf, gpd.GeoDataFrame)
    return result_gdf

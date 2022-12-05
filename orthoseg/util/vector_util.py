# -*- coding: utf-8 -*-
"""
Modile with generic Utility functions for vector manipulations.
"""

import copy
import logging
from typing import Optional, Tuple

import geofileops as gfo
from geofileops.util import geometry_util
from geofileops.util import geoseries_util
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeos
from shapely import geometry as sh_geom
import topojson
import topojson.ops

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def calc_onborder(
    geoms_gdf: gpd.GeoDataFrame,
    border_bounds: Tuple[float, float, float, float],
    onborder_column_name: str = "onborder",
) -> gpd.GeoDataFrame:
    """
    Add/update a column to the GeoDataFrame with:
        * 0 if the polygon isn't on the border and
        * 1 if it is.

    Args
        geoms_gdf: input GeoDataFrame
        border_bounds: the bounds (tupple with (xmin, ymin, xmax, ymax)
                       to check against to determine onborder
        onborder_column_name: the column name of the onborder column

    """
    # Split geoms that need unioning versus geoms that don't
    # -> They are on the edge of a tile
    geoms_gdf = geoms_gdf.copy()  # type: ignore
    geoms_gdf[onborder_column_name] = 0

    if geoms_gdf is not None and len(geoms_gdf.index) > 0:
        # Check
        for i, geom_row in geoms_gdf.iterrows():
            if geom_row["geometry"] is not None and not geom_row["geometry"].is_empty:
                # Check if the geom is on the border of the tile
                geom_bounds = geom_row["geometry"].bounds  # type: ignore
                if (
                    geom_bounds[0] <= border_bounds[0]
                    or geom_bounds[1] <= border_bounds[1]
                    or geom_bounds[2] >= border_bounds[2]
                    or geom_bounds[3] >= border_bounds[3]
                ):
                    geoms_gdf.loc[i, onborder_column_name] = 1  # type: ignore

    return geoms_gdf


def is_valid_reason(geoseries) -> pd.Series:
    # Remark: should be moved to geofileops (till available in geopandas)!!!
    return pd.Series(
        data=pygeos.is_valid_reason(geoseries.array.data), index=geoseries.index
    )  # type: ignore


def simplify_topo_orthoseg(
    geoseries: gpd.GeoSeries,
    tolerance: float,
    algorithm: gfo.SimplifyAlgorithm = gfo.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
    lookahead: int = 8,
    keep_points_on: Optional[sh_geom.base.BaseGeometry] = None,
) -> gpd.GeoSeries:
    """
    Applies simplify while retaining common boundaries between geometries in the
    geoseries.

    This is a specific version for orthoseg that, when algorithm is LANG, applies rdp
    simplify with halve the tolerance first.

    Args:
        geoseries (gpd.GeoSeries): the geoseries to simplify.
        algorithm (SimplifyAlgorithm): algorithm to use.
        tolerance (float): tolerance to use for simplify
        lookahead (int, optional): lookahead value for algorithms that use this.
            Defaults to 8.
        keep_points_on (Optional[sh_geom.base.BaseGeometry], optional): points that
            intersect with this geometry won't be removed by the simplification.
            Defaults to None.

    Returns:
        gpd.GeoSeries: the simplified geoseries
    """
    # Just return empty geoseries
    if len(geoseries) == 0:
        return geoseries

    # Copy geoseries
    geoseries_copy = geoseries.copy()  # type: ignore

    # Set empty geometries to None, if no geometries are left, return
    empty_idxs = pygeos.is_empty(geoseries_copy.array.data).nonzero()  # type: ignore
    if len(empty_idxs) > 0:
        geoseries_copy.iloc[empty_idxs] = None
    if len(geoseries_copy[geoseries_copy != np.array(None)]) == 0:
        return geoseries

    topo = topojson.Topology(geoseries_copy, prequantize=False, shared_coords=False)
    # 46889.5, 211427.5
    # 46932.5, 211391.5

    def simplify_topolines(
        _topolines: list,
        _algorithm: gfo.SimplifyAlgorithm,
        _tolerance: float,
        _lookahead: int,
        _keep_points_on: Optional[sh_geom.base.BaseGeometry],
    ) -> list:
        """
        Internal function that takes care of some plumbing specific to simplifying
        the lines in the topologies.

        Args:
            _topolines (list): _description_
            _algorithm (gfo.SimplifyAlgorithm): _description_
            _tolerance (float): _description_
            _lookahead (int): _description_
            _keep_points_on (Optional[sh_geom.base.BaseGeometry]): _description_

        Returns:
            list: _description_
        """
        topolines_geom = sh_geom.MultiLineString(_topolines)
        topolines_simpl = geometry_util.simplify_ext(
            geometry=topolines_geom,
            tolerance=_tolerance,
            algorithm=_algorithm,
            lookahead=_lookahead,
            keep_points_on=_keep_points_on,
            preserve_topology=True,
        )
        assert topolines_simpl is not None

        # Copy the results of the simplified lines
        if _algorithm == gfo.SimplifyAlgorithm.LANG:
            # For LANG, a simple copy is OK
            if isinstance(topolines_simpl, sh_geom.LineString):
                return [list(topolines_simpl.coords)]
            else:
                assert isinstance(topolines_simpl, sh_geom.MultiLineString)
                return [list(geom.coords) for geom in topolines_simpl.geoms]
        else:
            # For RDP, only overwrite the lines that have a valid result
            topolines_copy = copy.deepcopy(_topolines)
            for index in range(len(topolines_copy)):
                # Get the coordinates of the simplified version
                if isinstance(topolines_simpl, sh_geom.base.BaseMultipartGeometry):
                    topoline_simpl = topolines_simpl.geoms[index].coords  # type: ignore
                else:
                    topoline_simpl = topolines_simpl.coords
                # If the result of the simplify is a point, keep original
                if len(topoline_simpl) < 2:
                    continue
                elif (
                    list(topoline_simpl[0]) != topolines_copy[index][0]
                    or list(topoline_simpl[-1]) != topolines_copy[index][-1]
                ):
                    # Start or end point of the simplified version is not the same
                    # anymore
                    continue
                else:
                    topolines_copy[index] = list(topoline_simpl)

            return topolines_copy

    if algorithm == gfo.SimplifyAlgorithm.LANG:
        # For lang, first pre-simplify using rdp
        topolines_simplified = simplify_topolines(
            _topolines=topo.output["arcs"],
            _algorithm=gfo.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
            _tolerance=tolerance / 2,
            _lookahead=lookahead,
            _keep_points_on=keep_points_on,
        )
        for line_id in range(len(topo.output["arcs"])):
            topo.output["arcs"][line_id] = topolines_simplified[line_id]

        topolines_simplified = simplify_topolines(
            _topolines=topo.output["arcs"],
            _algorithm=algorithm,
            _tolerance=tolerance,
            _lookahead=lookahead,
            _keep_points_on=keep_points_on,
        )
        for line_id in range(len(topo.output["arcs"])):
            topo.output["arcs"][line_id] = topolines_simplified[line_id]
    else:
        topolines_simplified = simplify_topolines(
            _topolines=topo.output["arcs"],
            _algorithm=algorithm,
            _tolerance=tolerance,
            _lookahead=lookahead,
            _keep_points_on=keep_points_on,
        )
        for line_id in range(len(topo.output["arcs"])):
            topo.output["arcs"][line_id] = topolines_simplified[line_id]

    topo_simpl_geoseries = topo.to_gdf(crs=geoseries.crs).geometry
    assert isinstance(topo_simpl_geoseries, gpd.GeoSeries)
    topo_simpl_geoseries.array.data = pygeos.make_valid(  # type: ignore
        topo_simpl_geoseries.array.data  # type: ignore
    )
    geometry_types_orig = geoseries.type.unique()
    geometry_types_orig = geometry_types_orig[geometry_types_orig != None]
    geometry_types_simpl = topo_simpl_geoseries.type.unique()
    if len(geometry_types_orig) == 1 and (
        len(geometry_types_simpl) > 1
        or geometry_types_orig[0] != geometry_types_simpl[0]
    ):
        topo_simpl_geoseries = geoseries_util.geometry_collection_extract(
            topo_simpl_geoseries,
            gfo.GeometryType(geometry_types_orig[0]).to_primitivetype,
        )
    return topo_simpl_geoseries

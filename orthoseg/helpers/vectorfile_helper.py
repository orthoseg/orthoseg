# -*- coding: utf-8 -*-
"""
Modile with generic Utility functions for vectorfile manipulations.
"""

import logging
from pathlib import Path

import geofileops as gfo

from orthoseg.util import vector_util

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def reclassify_neighbours(
    input_path: Path,
    reclassify_column: str,
    query: str,
    output_path: Path,
    class_background: str = "background",
):
    """
    For features that comply to the query, if they have a neighbour (touch/overlap),
    change their classname to that of the neighbour with the longest intersection and
    dissolve to merge them.

    The query should follow the syntax of pandas.query queries and can us the following
    fields:
        - area (float): the area of the geometry.
        - perimeter (float): the perimeter of the geometry.

    Args:
        gdf (gpd.GeoDataFrame): input
        reclassify_column (str): column to reclassify.
        query (str): th query to find the features to reclassify.
        class_background (str, optional): the classname to treat as background.
            Defaults to "background".

    Raises:
        ValueError: raised if incompatible parameters are passed.
    """
    logger.info("start reclassify_neighbours")
    input_gdf = gfo.read_file(input_path, columns=[reclassify_column])
    output_gdf = vector_util.reclassify_neighbours(
        input_gdf,
        reclassify_column=reclassify_column,
        query=query,
        border_bounds=None,
        class_background=class_background,
    )
    gfo.to_file(output_gdf, output_path)

    # Add/recalculate columns with area and nbcoords
    gfo.add_column(
        path=output_path,
        name="area",
        type=gfo.DataType.REAL,
        expression="ST_Area(geom)",
        force_update=True,
    )
    gfo.add_column(
        path=output_path,
        name="nbcoords",
        type=gfo.DataType.INTEGER,
        expression="ST_NPoints(geom)",
        force_update=True,
    )

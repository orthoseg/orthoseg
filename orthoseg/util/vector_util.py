# -*- coding: utf-8 -*-
"""
Modile with generic Utility functions for vector manipulations.
"""

import logging
import math
from typing import Tuple, Union

import geopandas as gpd
import pyproj
import shapely.ops as sh_ops

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------    
def calc_onborder(
        geoms_gdf: gpd.GeoDataFrame,
        border_bounds: Tuple[float, float, float, float],
        onborder_column_name: str = "onborder") -> gpd.GeoDataFrame:
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
    if geoms_gdf is not None and len(geoms_gdf.index) > 0:
        
        # Check 
        for i, geom_row in geoms_gdf.iterrows():
            # Check if the geom is on the border of the tile
            geom_bounds = geom_row['geometry'].bounds
            onborder = 0
            if(geom_bounds[0] <= border_bounds[0] 
               or geom_bounds[1] <= border_bounds[1] 
               or geom_bounds[2] >= border_bounds[2]
               or geom_bounds[3] >= border_bounds[3]):
                onborder = 1
            
            geoms_gdf.loc[i, onborder_column_name] = onborder

    geoms_gdf.reset_index(drop=True, inplace=True)
    return geoms_gdf

if __name__ == '__main__':
    raise Exception("Not supported")

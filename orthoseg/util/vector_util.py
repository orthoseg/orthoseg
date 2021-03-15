# -*- coding: utf-8 -*-
"""
Modile with generic Utility functions for vector manipulations.
"""

import logging
import math
from pathlib import Path
from typing import Tuple, Union

from geofileops import geofile
import geopandas as gpd
import numpy as np
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

def create_grid(xmin: float,
                ymin: float,
                xmax: float,
                ymax: float,
                cell_width: float,
                cell_height: float) -> gpd.GeoDataFrame:
    
    rows = math.ceil((ymax-ymin) / cell_height)
    cols = math.ceil((xmax-xmin) / cell_width)
     
    polygons = []
    cell_left = xmin
    cell_right = xmin + cell_width
    for _ in range(cols+1):
        if cell_left > xmax:
            break
        cell_top = ymin+cell_height
        cell_bottom = ymin
        for _ in range(rows+1):
            if cell_bottom > ymax:
                break
            polygons.append(sh_ops.Polygon([(cell_left, cell_top), (cell_right, cell_top), (cell_right, cell_bottom), (cell_left, cell_bottom)])) 
            cell_top += cell_height
            cell_bottom += cell_height
            
        cell_left += cell_width
        cell_right += cell_width
        
    return gpd.GeoDataFrame({'geometry': polygons})

'''
# TODO: using geojson is more convenient, so this code can be deleted

def read_wkt(in_wkt_filepath: str):
    # Read the geoms in wkt file
    geoms = []
    with open(in_wkt_filepath, 'r') as in_file:
        lines = in_file.readlines()
        
        for line in lines:
            geom = sh_wkt.loads(line)
            if not geom.is_empty:
                geoms.append(geom)
    
    return geoms

def write_wkt(in_geoms,
              out_wkt_filepath: str):

    # If the in_geoms array is empty, return
    if not in_geoms or len(in_geoms) == 0:
        return
    
    # Write geoms to wkt
    with open(out_wkt_filepath, 'w') as dst:
        for geom in in_geoms:
            dst.write(f"{geom}\n")
'''

if __name__ == '__main__':
    raise Exception("Not supported")

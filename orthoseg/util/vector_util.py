# -*- coding: utf-8 -*-
"""
Modile with generic Utility functions for vector manipulations.
"""

import logging
import os
import glob
import time

import concurrent.futures as futures
import fiona
import geopandas as gpd
import numpy as np
import shapely.ops as sh_ops
import shapely.geometry as sh_geom

from orthoseg.util.vector import simplify_visval as simpl_vis
from orthoseg.util.vector import simplify_rdp_plus as simpl_rdp_plus
from orthoseg.util import geofile_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------
    
def read_file(filepath) -> []: 
    with fiona.open(filepath, 'r') as in_file:
        return list(in_file)

def unary_union_with_onborder(input_gdf: gpd.GeoDataFrame,
                              input_filepath: str,
                              output_filepath: str,
                              evaluate_mode: bool = False,
                              force: bool = False):
    
    # If output file already exists and force is False, return
    if not force and os.path.exists(output_filepath):
        logger.info(f"Force is false and output file exists already, skip: {output_filepath}")
        return None
        
    # If the input geoms are not yet in memory, read them from file
    if input_gdf is None:
        input_gdf = gpd.read_file(input_filepath)
        
        if evaluate_mode:
            logger.info("Evaluate mode, so limit input to 1000 geoms")
            input_gdf = input_gdf.head(1000)

    # Check if the onborder column is available!
    if 'onborder' not in input_gdf.columns:
        message = "STOP: onborder column is not available!"
        logger.critical(message)
        raise Exception(message)
    
    # Check if the input has a crs
    if input_gdf.crs is None:
        message = "STOP: input does not have a crs!"
        logger.critical(message)
        raise Exception(message)       
    
    # Get geoms on border, and union them
    onborder_gdf = input_gdf[input_gdf['onborder'] == 1]
        
    logger.info(f"Start unary_union on {len(onborder_gdf)} geoms on border")
    onborder_union_multi = sh_ops.unary_union(onborder_gdf.geometry.tolist())
    onborder_union = extract_polygons(onborder_union_multi)

    onborder_union_gdf = gpd.GeoDataFrame(geometry=onborder_union)
    logger.info("Unary_union ready")
    
    # Get geoms not on border
    notonborder_gdf = input_gdf.loc[(input_gdf['onborder'] == 0)]
    
    # Merge onborder and notonborder geoms
    union_gdf = onborder_union_gdf.append(
            notonborder_gdf, sort=False)

    # Write to file
    # Rem: Reset index because append retains the original index values
    union_gdf.reset_index(inplace=True, drop=True)
    union_gdf['id'] = union_gdf.index
    geofile_util.to_file(union_gdf, output_filepath)
    return union_gdf
            
def extract_polygons(in_geom) -> []:
    """
    Extracts all polygons from the input geom and returns them as a list.
    """
    
    # Extract the polygons from the multipolygon
    geoms = []
    if in_geom.geom_type == 'MultiPolygon':
        geoms = list(in_geom)
    elif in_geom.geom_type == 'Polygon':
        geoms.append(in_geom)
    elif in_geom.geom_type == 'GeometryCollection':
        for geom in in_geom:
            if geom.geom_type in ('MultiPolygon', 'Polygon'):
                geoms.append(geom)
            else:
                logger.debug(f"Found {geom.geom_type}, ignore!")
    else:
        raise IOError(f"in_geom is of an unsupported type: {in_geom.geom_type}")
    
    return geoms

def fix(geometry):
    
    # First check if the geom is None...
    if geometry is None:
        return None
    # If the geometry is valid, just return it
    if geometry.is_valid:
        return geometry

    # Else... try fixing it...
    geom_buf = geometry.buffer(0)
    if geom_buf.is_valid:
        return geom_buf
    else:
        logger.error(f"Error fixing geometry {geometry}")
        return geometry

def remove_inner_rings(geometry,
                       min_area_to_keep: float = None):
    
    # First check if the geom is None...
    if geometry is None:
        return None

    # If all inner rings need to be removed...
    if min_area_to_keep is None or min_area_to_keep == 0.0:
        # If there are no interior rings anyway, just return input
        if len(geometry.interiors) == 0:
            return geometry
        else:
            # Els create new polygon with only the exterior ring
            return sh_ops.Polygon(geometry.exterior)
    
    # If only small rings need to be removed... loop over them
    ring_coords_to_keep = []
    small_ring_found = False
    for ring in geometry.interiors:
        if abs(ring.area) <= min_area_to_keep:
            small_ring_found = True
        else:
            ring_coords_to_keep.append(ring.coords)
    
    # If no small rings were found, just return input
    if small_ring_found == False:
        return geometry
    else:
        return sh_ops.Polygon(geometry.exterior.coords, 
                              ring_coords_to_keep)        

def get_nb_coords(geometry) -> int:
    # First check if the geom is None...
    if geometry is None:
        return 0
    
    # Get the number of points for all rings
    nb_coords = len(geometry.exterior.coords)
    for ring in geometry.interiors:
        nb_coords += len(ring.coords)
    
    return nb_coords
    
def simplify_visval(geometry, 
                    threshold: int,
                    preserve_topology: bool = False):
    
    # Apply the simplification
    geom_simpl_np = simpl_vis.VWSimplifier(geometry.exterior.coords).from_threshold(threshold)

    # If simplified version has at least 3 points...
    if geom_simpl_np.shape[0] >= 3:
        #logger.info(f"geom_simpl_np.size: {geom_simpl_np.size}, geom_simpl_np.shape: {geom_simpl_np.shape}, geom_simpl_np: {geom_simpl_np}")
        return sh_ops.Polygon(geom_simpl_np)
    else:
        if preserve_topology:
            return geometry
        else:
            return None

def simplify_rdp_plus(geometry, 
                      epsilon: int,
                      preserve_topology: bool = False):
    
    # Apply the simplification
    geom_simpl_coords = simpl_rdp_plus.rdp(geometry.exterior.coords, 
                                           epsilon=epsilon)

    # If simplified version has at least 3 points...
    if len(geom_simpl_coords) >= 3:
        return sh_ops.Polygon(geom_simpl_coords)
    else:
        if preserve_topology:
            return geometry
        else:
            return None
        
def calc_onborder(geoms_gdf: gpd.GeoDataFrame,
                  border_bounds,
                  onborder_column_name: str = "onborder"):
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

    #logger.info(geoms_gdf)
    return geoms_gdf

def create_grid(xmin: float,
                ymin: float,
                xmax: float,
                ymax: float,
                cell_width: float,
                cell_height: float) -> gpd.GeoDataFrame:
    
    rows = int(np.ceil((ymax-ymin) / cell_height))
    cols = int(np.ceil((xmax-xmin) / cell_width))
     
    polygons = []
    cell_left = xmin
    cell_right = xmin + cell_width
    for _ in range(cols):
        if cell_right > xmax:
            break
        cell_top = ymax
        cell_bottom = ymax-cell_height
        for _ in range(rows):
            if cell_bottom < ymin:
                break
            polygons.append(sh_ops.Polygon([(cell_left, cell_top), (cell_right, cell_top), (cell_right, cell_bottom), (cell_left, cell_bottom)])) 
            cell_top -= cell_height
            cell_bottom -= cell_height
            
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

def main():
    '''
    grid_gdf = create_grid(xmin=0, ymin=0, xmax=300000, ymax=300000,
                           cell_width=2000, cell_height=2000)
    geofile_util.to_file(grid_gdf, "X:\\Monitoring\\OrthoSeg\\grid_2000.gpkg")
    '''
    
if __name__ == '__main__':
    main()

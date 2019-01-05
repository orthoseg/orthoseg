# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:01:07 2018

@author: pierog
"""

import logging
import os
import glob

import numpy as np
import shapely.ops as sh_ops
import geopandas as gpd

import log_helper
import vector.simplify_visval as simpl_vis

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def union_vectors(base_dir: str,
                  evaluate_mode: bool = False,
                  force: bool = False):
    
    eval_suffix = ""
    if evaluate_mode:
        eval_suffix = "_eval"

    # Prepare output dir
    union_dir = os.path.join(base_dir, f"_union_out{eval_suffix}")
    geoms_unioned_simpl_filepath = os.path.join(union_dir, f"geoms_unioned_gt5m2_simpl.geojson")
    geoms_unioned_simpl_vis_filepath = os.path.join(union_dir, f"geoms_unioned_gt5m2_simpl_vis.geojson")
    
    # If unioned output file already exists... skip        
    if not force and os.path.exists(geoms_unioned_simpl_vis_filepath):
        logger.info(f"Force is false and result exists already, stop: {geoms_unioned_simpl_vis_filepath}")
        return

    if not os.path.exists(union_dir):
        os.mkdir(union_dir)
        
    # Read input
    # -------------------------------------------------------------------------  
    # If temp file with all polygons to me processed already exists
    geoms_orig_filepath = os.path.join(union_dir, f"tmp_geoms_orig.geojson")
    if not force and os.path.exists(geoms_orig_filepath):
        logger.info(f"Force is false and temp file exists already, read it: {geoms_orig_filepath}")
        geoms_gdf = gpd.read_file(geoms_orig_filepath)
        
        if evaluate_mode:
            logger.info("Evaluate mode, so limit input to 1000 geoms")
            geoms_gdf = geoms_gdf.head(1000)

    else:
        # Get list of all files to process...
        in_filepaths = glob.glob(f"{base_dir}{os.sep}**{os.sep}*_pred_cleaned.geojson", recursive=True)
        logger.info(f"Found {len(in_filepaths)} files to process in {base_dir}")

        # Loop through all files to be processed...
        sorted_filepaths = sorted(in_filepaths)
        for i, in_filepath in enumerate(sorted_filepaths):
            
            if evaluate_mode and i >= 100:
                logger.info("Evaluate mode and 100 files processed, so stop")
                break
                
            # Read the geoms in the file and add to general list of geoms
            logger.debug(f"Read input geom wkt file: {in_filepath}")
            geoms_file_gdf = gpd.read_file(in_filepath)
            
            for j, row in geoms_file_gdf.iterrows():
                basename = os.path.basename(in_filepath)
                splitted = basename.split('_')
                xmin = float(splitted[0]) + ((64*0.25) + 1)
                ymin = float(splitted[1]) + ((64*0.25) + 1)
                xmax = float(splitted[2]) - ((64*0.25) + 1)
                ymax = float(splitted[3]) - ((64*0.25) + 1)
                
                if(row.geometry.bounds[0] <= xmin 
                   or row.geometry.bounds[1] <= ymin
                   or row.geometry.bounds[2] >= xmax
                   or row.geometry.bounds[3] >= ymax):
                    geoms_file_gdf.loc[j, 'onborder'] = 1
                else:
                    geoms_file_gdf.loc[j, 'onborder'] = 0
            
            if i == 0:
                geoms_gdf = geoms_file_gdf
            else:
                geoms_gdf = geoms_gdf.append(geoms_file_gdf, sort=False)
            
            if evaluate_mode:
                logger.debug(f"File {i} contains {len(geoms_file_gdf)} geoms: {os.path.basename(geoms_orig_filepath)}")
            
        logger.info(f"Write the {len(geoms_gdf)} original geoms to {geoms_orig_filepath}")
        
        # Add id column and write to file
        # Remark: reset index because append retains the original index values
        geoms_gdf.reset_index()
        geoms_gdf['id'] = geoms_gdf.index
        # TODO: setting the CRS here hardcoded should be removed
        geoms_gdf.crs = "epsg:31370"
        geoms_gdf.to_file(geoms_orig_filepath, driver="GeoJSON", )

    # Union all data
    # -------------------------------------------------------------------------  
    # If temp file with all polygons to me processed already exists
    geoms_unioned_filepath = os.path.join(union_dir, f"geoms_unioned_gt5m2.geojson")
    
    # If unioned output file already exists... read it
    if not force and os.path.exists(geoms_unioned_filepath):
        logger.info(f"Force is false and result exists already, stop: {geoms_unioned_filepath}")

        geoms_gt5m2_gdf = gpd.read_file(geoms_unioned_filepath)

    else:        
        # Get geoms on border, those need to be unioned
        geoms_onborder_gdf = geoms_gdf[geoms_gdf['onborder'] == 1]
            
        # Calculate union of all geoms on the border
        logger.info(f"Start unary_union on {len(geoms_onborder_gdf)} geoms on border")
        geoms_onborder_unioned_gdf = unary_union(in_geoms=geoms_onborder_gdf.geometry.tolist(), 
                                                 create_multipolygon=False)
        logger.info(f"Unary_union ready")
            
        # Get only border unioned geoms > 5m² 
        geoms_gt5m2_gdf = geoms_onborder_unioned_gdf[geoms_onborder_unioned_gdf.geometry.area > 5]
    
        # Get geoms not on border >5 m² 
        geoms_notonborder_gt5m2_gdf = geoms_gdf.loc[(geoms_gdf['onborder'] == 0) &
                                                    (geoms_gdf.geometry.area > 5)]
        
        # Write all polygons to geojson file
        geoms_gt5m2_gdf = geoms_gt5m2_gdf.append(geoms_notonborder_gt5m2_gdf, sort=False)
    
        # Add some columns to the output and save.
        # Rem: Reset index because append retains the original index values
        geoms_gt5m2_gdf.reset_index()
        geoms_gt5m2_gdf['id'] = geoms_gt5m2_gdf.index
        geoms_gt5m2_gdf['area'] = geoms_gt5m2_gdf.geometry.area
        
        # TODO: setting the CRS here hardcoded should be removed
        geoms_gt5m2_gdf.crs = "epsg:31370"
        geoms_gt5m2_gdf.to_file(geoms_unioned_filepath, driver="GeoJSON")
        logger.info(f"Result written to {geoms_unioned_simpl_filepath}")

    # Simplify all data
    # -------------------------------------------------------------------------  
    logger.info("Simplify")
    # Simplify with standard shapely algo (= Deuter-Pecker)
    geoms_gt5m2_simpl_gdf = geoms_gt5m2_gdf.simplify(1)
    geoms_gt5m2_simpl_gdf.to_file(geoms_unioned_simpl_filepath, driver="GeoJSON")
    logger.info(f"Result written to {geoms_unioned_simpl_filepath}")

    # Also do the simplify with Visvalingam algo
    geoms_gt5m2_simpl_vis_gdf = geoms_gt5m2_gdf
    #geoms_gt5m2_simpl_vis_gdf['geometry'] = geoms_gt5m2_simpl_vis_gdf.geometry.apply(lambda x: sh_ops.Polygon(simpl_vis.VWSimplifier(x.exterior.coords).from_threshold(5)))
    geoms_gt5m2_simpl_vis_gdf['geometry'] = geoms_gt5m2_simpl_vis_gdf.geometry.apply(lambda geom: simplify_visval(geom, 10))
    
    geoms_gt5m2_simpl_vis_gdf.to_file(geoms_unioned_simpl_vis_filepath, driver="GeoJSON")
    logger.info(f"Result written to {geoms_unioned_simpl_vis_filepath}")
        
    '''
    # Write a simplified version of the geoms to wkt file
    simplify_tol = 1.0
    geoms_union_simpl_wkt_filepath = f"{output_base_filepath}_gt5m2_simpl.wkt"   
    logger.info(f"Write simplified version of merged geoms > 4m² to wkt file, with simplify_tol: {simplify_tol}")
    geoms_simpl = []
    with open(geoms_union_simpl_wkt_filepath, 'w') as dst:
        for geom in geoms_union:
            geom_simpl = geom.simplify(simplify_tol, preserve_topology=True)            
            if not geom_simpl.is_empty:
                if geom_simpl.area > 5:
                    geoms_simpl.append(geom_simpl)
                    dst.write(f"{geom_simpl}\n")
    '''
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
            
def unary_union(in_geoms,
                create_multipolygon: bool = True) -> gpd.GeoDataFrame:
    # Union all geoms...    
    unioned_geom = sh_ops.unary_union(in_geoms)
    
    # If we don't want one multipolygon to be returned
    if create_multipolygon:
        return unioned_geom
    else:
        # Extract the polygons from the multipolygon
        geoms = []
        if unioned_geom.geom_type == 'MultiPolygon':
            geoms = list(unioned_geom)
        elif unioned_geom.geom_type == 'Polygon':
            geoms.append(unioned_geom)
        elif unioned_geom.geom_type == 'GeometryCollection':
            for geom in unioned_geom:
                if geom.geom_type in ('MultiPolygon', 'Polygon'):
                    geoms.append(geom)
        else:
            raise IOError(f"unioned_geoms is of an unsupported type: {unioned_geom.geom_type}")
        
        return gpd.GeoDataFrame(geometry=geoms)

def write_segmented_geoms(geoms,
                          out_filepath: str,
                          src_image_crs,
                          xmin: float,
                          ymin: float,
                          xmax: float,
                          ymax: float):
    # Split geoms that need unioning versus geoms that don't
    # -> They are on the edge of a tile 
    geom_records = []
    if geoms and len(geoms) > 0:
        
        # Check 
        for geom in geoms:
            # Check if the geom is on the border of the tile
            geom_bounds = geom.bounds
            onborder = 0
            if(geom_bounds[0] <= xmin or geom_bounds[1] <= ymin or
               geom_bounds[2] >= xmax or geom_bounds[3] >= ymax):
                onborder = 1
               
            geom_records.append({'geometry': geom, 
                                 'onborder': onborder})
    
    # Convert to geodataframe and write to geojson 
    geoms_gdf = gpd.GeoDataFrame(geom_records, columns=['geometry', 'onborder'])
    geoms_gdf.crs = src_image_crs
    #logger.debug(f"Write the {len(geoms_gdf)} geoms, of types {geoms_gdf.geometry.type.unique()} in geoms_gdf to file")
    geoms_gdf.to_file(out_filepath, driver="GeoJSON")

# TODO: code isn't tested!!!
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
    for i in range(cols):
        cell_top = xmin
        cell_bottom = ymax-cell_height
        for j in range(rows):
            polygons.append(sh_ops.Polygon([(cell_left, cell_top), (cell_right, cell_top), (cell_right, cell_bottom), (cell_left, cell_bottom)])) 
            cell_top -= cell_height
            cell_bottom -= cell_height
        cell_left += cell_width
        cell_right += cell_width
        
    return gpd.GeoDataFrame({'geometry':polygons})

'''
TODO: GeoDataFrame.explode seems to do the job... so this can be removed.

def multipoly2singlepoly(input_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    singlepoly_gdf = input_gdf[input_gdf.geometry.type == 'Polygon']
    multipoly_gdf = input_gdf[input_gdf.geometry.type == 'MultiPolygon']

    for i, row in multipoly_gdf.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=multipoly_gdf.crs).T]*len(Series_geometries), ignore_index=True)
        df['geometry'] = Series_geometries
        singlepoly_gdf = pd.concat([singlepoly_gdf, df])

    singlepoly_gdf.reset_index(inplace=True, drop=True)
    return singlepoly_gdf
'''

if __name__ == '__main__':

    # Main project dir
    project_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\greenhouses"

    # Main initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)

    # Input images information
    image_pixel_width = 1024
    image_pixel_height = image_pixel_width
    pixels_overlap = 64
    to_predict_input_dir = f"X:\\GIS\\GIS DATA\\_Tmp\\Ortho_2018_autosegment_cache\\{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"
    
    # Model information
    segmentation_model = 'linknet'
    backbone_name = 'inceptionresnetv2'
    model_basename = f"greenhouse_12_{segmentation_model}_{backbone_name}_01"
    model_weights = f"{model_basename}_089_0.95702_0.97469"
    #    model_weights = f"{model_basename}_142_0.96387_0.97575"
    #model_weights = f"{model_basename}_144_0.94681_0.97451"
    
    prediction_eval_subdir = f"prediction_{model_weights}_eval"
    
    union_vectors(base_dir=f"{to_predict_input_dir}_{prediction_eval_subdir}",
                  evaluate_mode=True,
                  force=True)

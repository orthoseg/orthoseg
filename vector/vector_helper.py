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
import vector.simplify_rdp_plus as simpl_rdp_plus

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

# TODO: this function isn't a general-purpose vector helper, maybe should be 
# moved somewhere else?
def postprocess_vectors(base_dir: str,
                        evaluate_mode: bool = False,
                        force: bool = False):
    
    # TODO: this function should be split to several ones, because now it does
    # several things that aren't covered with the name
    eval_suffix = ""
    if evaluate_mode:
        eval_suffix = "_eval"

    # Prepare output dir
    union_dir = os.path.join(base_dir, f"_union_out{eval_suffix}")
    if not os.path.exists(union_dir):
        os.mkdir(union_dir)

    # Read and merge input files
    geoms_orig_filepath = os.path.join(union_dir, f"geoms_orig.geojson")
    geoms_gdf = merge_vector_files(input_dir=base_dir,
                                   output_filepath=geoms_orig_filepath,
                                   evaluate_mode=evaluate_mode,
                                   force=force)               

    # Union the data, optimized using the available onborder column
    geoms_union_filepath = os.path.join(union_dir, f"geoms_union.geojson")
    geoms_union_gdf = unary_union_with_onborder(input_gdf=geoms_gdf,
                              input_filepath=geoms_orig_filepath,
                              output_filepath=geoms_union_filepath,
                              evaluate_mode=evaluate_mode,
                              force=force)

    # Retain only geoms > 5m²
    geoms_gt5m2_filepath = os.path.join(union_dir, f"geoms_union_gt5m2.geojson")
    geoms_gt5m2_gdf = None
    if force or not os.path.exists(geoms_gt5m2_filepath):
        if geoms_union_gdf is None:
            geoms_union_gdf = gpd.read_file(geoms_union_filepath)
        geoms_gt5m2_gdf = geoms_union_gdf.loc[(geoms_union_gdf.geometry.area > 5)]
        # TODO: setting the CRS here hardcoded should be removed
        if geoms_gt5m2_gdf.crs is None:
            message = "No crs available!!!"
            logger.error(message)
            #raise Exception(message)
        
        # Qgis wants unique id column, otherwise weird effects!
        geoms_gt5m2_gdf.reset_index(inplace=True, drop=True)
        geoms_gt5m2_gdf['id'] = geoms_gt5m2_gdf.index 
        geoms_gt5m2_gdf.to_file(geoms_gt5m2_filepath, driver="GeoJSON")

    # Simplify with standard shapely algo 
    # -> if preserve_topology False, this is Ramer-Douglas-Peucker, otherwise ?
    geoms_simpl_shap_filepath = os.path.join(
            union_dir, f"geoms_simpl_shap.geojson")
    geoms_simpl_shap_gdf = None
    if force or not os.path.exists(geoms_simpl_shap_filepath):
        logger.info("Simplify with default shapely algo")
        # If input geoms not yet in memory, read from file
        if geoms_gt5m2_gdf is None:
            geoms_gt5m2_gdf = gpd.read_file(geoms_gt5m2_filepath)

        # Simplify, fix invalid geoms, remove empty geoms, 
        # apply multipart-to-singlepart, only > 5m² + write
        geoms_simpl_shap_gdf = geoms_gt5m2_gdf.copy()
        geoms_simpl_shap_gdf['geometry'] = geoms_simpl_shap_gdf.simplify(
                tolerance=0.5, preserve_topology=True)
        geoms_simpl_shap_gdf['geometry'] = geoms_simpl_shap_gdf.geometry.apply(
                lambda geom: fix(geom))
        geoms_simpl_shap_gdf.dropna(subset=['geometry'], inplace=True)
        geoms_simpl_shap_gdf = geoms_simpl_shap_gdf.reset_index(drop=True).explode()
        geoms_simpl_shap_gdf['geometry'] = geoms_simpl_shap_gdf.geometry.apply(
                lambda geom: remove_inner_rings(geom, 2))        
                
        # Add area column, and remove rows with small area
        geoms_simpl_shap_gdf['area'] = geoms_simpl_shap_gdf.geometry.area       
        geoms_simpl_shap_gdf = geoms_simpl_shap_gdf.loc[
                (geoms_simpl_shap_gdf['area'] > 5)]

        # Qgis wants unique id column, otherwise weird effects!
        geoms_simpl_shap_gdf.reset_index(inplace=True, drop=True)
        geoms_simpl_shap_gdf['id'] = geoms_simpl_shap_gdf.index 
        geoms_simpl_shap_gdf['nbcoords'] = geoms_simpl_shap_gdf.geometry.apply(
                lambda geom: get_nb_coords(geom))
        geoms_simpl_shap_gdf.to_file(geoms_simpl_shap_filepath, 
                                     driver="GeoJSON")
        logger.info(f"Result written to {geoms_simpl_shap_filepath}")
        
    # Apply begative buffer on result
    geoms_simpl_shap_m1m_filepath = os.path.join(
            union_dir, f"geoms_simpl_shap_m1.5m_3.geojson")
    if force or not os.path.exists(geoms_simpl_shap_m1m_filepath):
        logger.info("Apply negative buffer")
        # If input geoms not yet in memory, read from file
        if geoms_simpl_shap_gdf is None:
            geoms_simpl_shap_gdf = gpd.read_file(geoms_simpl_shap_filepath)
            
        # Simplify, fix invalid geoms, remove empty geoms, 
        # apply multipart-to-singlepart, only > 5m² + write
        geoms_simpl_shap_m1m_gdf = geoms_simpl_shap_gdf.copy()
        geoms_simpl_shap_m1m_gdf['geometry'] = geoms_simpl_shap_m1m_gdf.buffer(
                distance=-1.5, resolution=3)
        
        '''
        geoms_simpl_shap_gdf['geometry'] = geoms_simpl_shap_gdf.simplify(
                tolerance=0.5, preserve_topology=True)
        geoms_simpl_shap_gdf['geometry'] = geoms_simpl_shap_gdf.geometry.apply(
                lambda geom: fix(geom))
        '''
        geoms_simpl_shap_m1m_gdf.dropna(subset=['geometry'], inplace=True)
        geoms_simpl_shap_m1m_gdf = geoms_simpl_shap_m1m_gdf.reset_index(drop=True).explode()
        '''
        geoms_simpl_shap_m1m_gdf['geometry'] = geoms_simpl_shap_m1m_gdf.geometry.apply(
                lambda geom: remove_inner_rings(geom, 2))        
        '''
                
        # Add/calculate area column, and remove rows with small area
        geoms_simpl_shap_m1m_gdf['area'] = geoms_simpl_shap_m1m_gdf.geometry.area       
        geoms_simpl_shap_m1m_gdf = geoms_simpl_shap_m1m_gdf.loc[
                (geoms_simpl_shap_m1m_gdf['area'] > 5)]

        # Qgis wants unique id column, otherwise weird effects!
        geoms_simpl_shap_m1m_gdf.reset_index(inplace=True, drop=True)
        geoms_simpl_shap_m1m_gdf['id'] = geoms_simpl_shap_m1m_gdf.index 
        geoms_simpl_shap_m1m_gdf['nbcoords'] = geoms_simpl_shap_m1m_gdf.geometry.apply(
                lambda geom: get_nb_coords(geom))        
        geoms_simpl_shap_m1m_gdf.to_file(geoms_simpl_shap_m1m_filepath, 
                                     driver="GeoJSON")
        logger.info(f"Result written to {geoms_simpl_shap_m1m_filepath}")
        
    '''
    # Also do the simplify with Visvalingam algo
    geoms_simpl_vis_filepath = os.path.join(
            union_dir, f"geoms_simpl_vis.geojson")
    if force or not os.path.exists(geoms_simpl_vis_filepath):
        # If input geoms not yet in memory, read from file
        if geoms_gt5m2_gdf is None:
            geoms_gt5m2_gdf = gpd.read_file(geoms_gt5m2_filepath)

        # Simplify, fix invalid geoms, remove empty geoms, 
        # apply multipart-to-singlepart, only > 5m² + write
        geoms_simpl_vis_gdf = geoms_gt5m2_gdf.copy()
        geoms_simpl_vis_gdf['geometry'] = geoms_simpl_vis_gdf.geometry.apply(
                lambda geom: simplify_visval(geom, 10))
        geoms_simpl_vis_gdf['geometry'] = geoms_simpl_vis_gdf.geometry.apply(
                lambda geom: fix(geom))
        geoms_simpl_vis_gdf.dropna(subset=['geometry'], inplace=True)
        geoms_simpl_vis_gdf = geoms_simpl_vis_gdf.reset_index().explode()
        geoms_simpl_vis_gdf['geometry'] = geoms_simpl_vis_gdf.geometry.apply(
                lambda geom: remove_inner_rings(geom, 2))
        geoms_simpl_vis_gdf = geoms_simpl_vis_gdf.loc[
                (geoms_simpl_vis_gdf.geometry.area > 5)]
        
        # Qgis wants unique id column, otherwise weird effects!
        geoms_simpl_vis_gdf.reset_index(inplace=True)
        geoms_simpl_vis_gdf['id'] = geoms_simpl_vis_gdf.index        
        geoms_simpl_vis_gdf.to_file(geoms_simpl_vis_filepath, 
                                          driver="GeoJSON")
        logger.info(f"Result written to {geoms_simpl_vis_filepath}")
    
    # Also do the simplify with Ramer-Douglas-Peucker "plus" algo:
    # the plus is that there is some extra checks
    geoms_simpl_rdpp_filepath = os.path.join(
            union_dir, f"geoms_simpl_rdpp.geojson")
    if force or not os.path.exists(geoms_simpl_rdpp_filepath):
        # If input geoms not yet in memory, read from file
        if geoms_gt5m2_gdf is None:
            geoms_gt5m2_gdf = gpd.read_file(geoms_gt5m2_filepath)

        # Simplify, fix invalid geoms, remove empty geoms, 
        # apply multipart-to-singlepart, only > 5m² + write
        geoms_simpl_rdpp_gdf = geoms_gt5m2_gdf.copy()
        geoms_simpl_rdpp_gdf['geometry'] = geoms_simpl_rdpp_gdf.geometry.apply(
                lambda geom: simplify_rdp_plus(geom, 1))
        geoms_simpl_rdpp_gdf['geometry'] = geoms_simpl_rdpp_gdf.geometry.apply(
                lambda geom: fix(geom))
        geoms_simpl_rdpp_gdf.dropna(subset=['geometry'], inplace=True)
        geoms_simpl_rdpp_gdf = geoms_simpl_rdpp_gdf.reset_index().explode()
        geoms_simpl_rdpp_gdf['geometry'] = geoms_simpl_rdpp_gdf.geometry.apply(
                lambda geom: remove_inner_rings(geom, 2))
        # Qgis wants unique id column, otherwise weird effects!
        geoms_simpl_rdpp_gdf.reset_index(inplace=True)
        geoms_simpl_rdpp_gdf['id'] = geoms_simpl_rdpp_gdf.index               
        geoms_simpl_rdpp_gdf.to_file(geoms_simpl_rdpp_filepath, 
                                     driver="GeoJSON")
        logger.info(f"Result written to {geoms_simpl_rdpp_filepath}")
    '''
    
def merge_vector_files(input_dir: str,
                       output_filepath: str,
                       apply_on_border_distance: float = None,
                       evaluate_mode: bool = False,
                       force: bool = False) -> gpd.GeoDataFrame:
    """
    Merges all geojson files in input dir (recursively) and writes it to file.
    
    Returns the resulting GeoDataFrame or None if the output file already 
    exists.
    
    Args
        input_dir:
        output_filepath:
        apply_on_border_distance:
        evaluate_mode:
        force:
    """
    
    # Check if we need to do anything anyway
    if not force and os.path.exists(output_filepath):
        logger.info(f"Force is false and output file exists already, skip: {output_filepath}")
        return None

    # Make sure the output dir exists
    output_dir = os.path.split(output_filepath)[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get list of all files to process...
    logger.info(f"List all files to be merged in {input_dir}")
    in_filepaths = glob.glob(f"{input_dir}{os.sep}**{os.sep}*_pred_cleaned.geojson", recursive=True)
    logger.info(f"Found {len(in_filepaths)} files to process")

    # Loop through all files to be processed...
    geoms_gdf = None
    sorted_filepaths = sorted(in_filepaths)
    for i, in_filepath in enumerate(sorted_filepaths):
        
        if evaluate_mode and i >= 100:
            logger.info("Evaluate mode and 100 files processed, so stop")
            break
            
        # Read the geoms in the file and add to general list of geoms
        logger.debug(f"Read input geom wkt file: {in_filepath}")
        geoms_file_gdf = gpd.read_file(in_filepath)
        
        # If caller wants on_border column to be added
        if apply_on_border_distance is not None:
            
            # Implementation is not finished, and don't really need it anymore
            # Just keep it here for if it should be revived.
            raise Exception("Not implemented!!!")
            '''            
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
            '''
            
        if i == 0:
            geoms_gdf = geoms_file_gdf
            
            # Check if the input has a crs
            if geoms_gdf.crs is None:
                message = "STOP: input does not have a crs!"
                logger.critical(message)
                raise Exception(message)      
        else:
            geoms_gdf = geoms_gdf.append(geoms_file_gdf, sort=False)
        
        if evaluate_mode:
            logger.debug(f"File {i} contains {len(geoms_file_gdf)} geoms: {os.path.basename(in_filepath)}")
        
    logger.info(f"Write the {len(geoms_gdf)} geoms to {output_filepath}")
    
    # Add id column and write to file
    # Remark: reset index because append retains the original index values
    geoms_gdf.reset_index(inplace=True, drop=True)
    geoms_gdf['id'] = geoms_gdf.index    
    geoms_gdf.to_file(output_filepath, driver="GeoJSON")    
    
    return geoms_gdf

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
    union_gdf.to_file(output_filepath, driver="GeoJSON")
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
    for i, ring in enumerate(geometry.interiors):
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
    # Init
    project_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\greenhouses"
    log_dir = os.path.join(project_dir, "log")
    global logger
    logger = log_helper.main_log_init(log_dir, __name__)

    # Model information
    segment_subject = "horsetracks"
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"
    project_dir = os.path.join(base_dir, segment_subject)
    training_dir = os.path.join(project_dir, "training")
    traindata_dir = os.path.join(training_dir, "train_14")
    predict_basedir = "X:\\GIS\GIS DATA\\_SegmentCache\\Ortho_2018"
    predict_subdir = "1024x1024_64pxOverlap_horsetracks_17_inceptionresnetv2+linknet_0.93241_0.95943_0"
    predict_dir = os.path.join(predict_basedir, predict_subdir)
    
    postprocess_vectors(base_dir=predict_dir,
                        evaluate_mode=False,
                        force=False)
    
if __name__ == '__main__':
    main()

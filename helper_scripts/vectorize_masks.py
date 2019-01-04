# -*- coding: utf-8 -*-
"""
This is a helper script to create a geojson file based on an existing directory 
containing mask files.

If you follow the normal procedure to build up your training, validation
and testing datasets you don't need this!

@author: Pieter Roggemans
"""

import os
import glob

import rasterio as rio
import rasterio.features as rio_features
import shapely as sh
import shapely.geometry as sh_geom
import geopandas as gpd

import log_helper
import vector_helper as vh

def vectorize_masks(input_image_dir: str,
                    train_dataset_type: str = 'train'):

    # Get list of all image files to process...
    image_filepaths = []
    input_ext = ['.tif', '.jpg']
    for input_ext_cur in input_ext:
        image_filepaths.extend(glob.glob(f"{input_image_dir}{os.sep}**{os.sep}*{input_ext_cur}", recursive=True))
    nb_files = len(image_filepaths)
    logger.info(f"Found {nb_files} {input_ext} images to predict on in {input_image_dir}")

    label_records = []
    for mask_filepath in image_filepaths:
        
        # Read mask file
        with rio.open(mask_filepath) as mask_ds:
            # Read pixels
            mask_arr = mask_ds.read(1)
        
        # Pars mask_transform info from filename
        mask_filename = os.path.basename(mask_filepath)
        mask_filename_noext = os.path.splitext(mask_filename)[0]
        mask_info = mask_filename_noext.split('_')
        try:
            xmin = float(mask_info[0])
            ymin = float(mask_info[1])
            xmax = float(mask_info[2])
            ymax = float(mask_info[3])
            pixel_width = float(mask_info[4])
            pixel_height = float(mask_info[5])
            if len(mask_info) >= 7:
                label_type = mask_info[6]
            else:
                label_type = "" 
                
        except:
            logger.error(f"SKIP FILE: error extracting transform info from filename {mask_filename}")
            continue

        mask_transform = rio.transform.from_bounds(xmin, ymin, xmax, ymax, 
                                                   pixel_width, pixel_height)
        
        # Polygonize result
        # Returns a list of tupples with (geometry, value)
        shapes = rio_features.shapes(mask_arr,
                                     mask=mask_arr,
                                     transform=mask_transform)
        
        # Convert shapes to shapely geometries...
        geoms = []
        for shape in list(shapes):
            geom, value = shape
            geom_sh = sh.geometry.shape(geom)
            geoms.append(geom_sh)
            
        # If there are polygons found, convert to positive mask records
        if len(geoms) > 0:
            for geom in geoms:
                label_records.append({'geometry': geom, 
                                     'is_positive_eg': 1,
                                     'label_type': label_type,
                                     'train_dataset_type': train_dataset_type})
        else:
            # Nothing found, so it is an all-black, "negative" mask...
            label_records.append({'geometry': sh_geom.box(xmin, ymin, xmax, ymax), 
                                 'is_positive_eg': 0,
                                 'label_type': label_type,
                                 'train_dataset_type': train_dataset_type})
            
    # Convert to geodataframe and write to geojson 
    labels_gdf = gpd.GeoDataFrame(label_records, 
                                  columns=['geometry', 'is_positive_eg', 
                                           'label_type', 'train_dataset_type'])
    labels_gdf.crs = 'epsg:31370'
    
    # Cleanup data (dissolve + simplify)
    labels_gdf = labels_gdf.dissolve(by=['is_positive_eg', 'label_type', 
                                         'train_dataset_type'])
    labels_gdf = labels_gdf.reset_index().explode()
    labels_gdf.geometry = labels_gdf.geometry.simplify(0.5)
    #labels_gdf['geometry'] = labels_gdf.geometry.apply(lambda geom: vh.simplify_visval(geom, 2))
        
    # Write result to file
    #logger.debug(f"Write the {len(geoms_gdf)} geoms, of types {geoms_gdf.geometry.type.unique()} in geoms_gdf to file")
    out_filepath = os.path.join(input_image_dir, "labels.geojson")
    if os.path.exists(out_filepath):
        os.remove(out_filepath)
    labels_gdf.to_file(out_filepath, driver="GeoJSON")
          
if __name__ == '__main__':

    # Main project dir
    project_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\horsetracks"

    # Main initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)
    
    train_dir = os.path.join(project_dir, "train")
    mask_dir = os.path.join(train_dir, "mask")
    
    vectorize_masks(mask_dir)

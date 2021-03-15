# -*- coding: utf-8 -*-
"""
This is a helper script to create a geojson file based on an existing directory 
containing mask files.

If you follow the normal procedure to build up your training, validation
and testing datasets you don't need this!
"""

import os
from pathlib import Path

import rasterio as rio
import rasterio.features as rio_features
import rasterio.transform as rio_transform
import shapely as sh
import shapely.geometry as sh_geom
import geopandas as gpd
import glob

from orthoseg.helpers import log_helper

def vectorize_masks(input_image_dir: Path,
                    output_filepath: Path,
                    projection_if_missing: str):

    # Get list of all image files to process...
    image_filepaths = []
    input_ext = ['.tif', '.jpg']
    for input_ext_cur in input_ext:
        image_filepaths.extend(glob.glob(f"{input_image_dir}/**/*{input_ext_cur}", recursive=True))
    nb_files = len(image_filepaths)
    logger.info(f"Found {nb_files} {input_ext} masks to vectorize in {input_image_dir}")

    label_records = []
    label_type = None
    xmin, ymin, xmax, ymax = (None, None, None, None)
    for mask_filepath in image_filepaths:
        
        # First try to parse mask_transform info from filename
        mask_transform = None
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
            
            mask_transform = rio_transform.from_bounds(
                    xmin, ymin, xmax, ymax, pixel_width, pixel_height)
                
        except:
            #logger.warning(f"Transform info not found in filename {mask_filename}")
            None
            
        # Extract info from the mask file
        with rio.open(mask_filepath) as mask_ds:            
            # If mask_transform couldn't be read from filename, try reading
            # it from the file metadata
            if mask_transform is None:
                if(mask_ds.transform is not None
                   and mask_ds.transform[2] > 0):
                    mask_transform = mask_ds.transform
                else:
                    logger.error(f"SKIP FILE: No valid transform info found in filename nor in file metadata ({mask_ds.transform}) for: {mask_filename}")
                    continue

            # Read pixels
            mask_arr = mask_ds.read(1)
                  
        # Polygonize result
        # Returns a list of tupples with (geometry, value)
        shapes = rio_features.shapes(mask_arr,
                                     mask=mask_arr,
                                     transform=mask_transform)
        
        # Convert shapes to shapely geometries...
        geoms = []
        for shape in list(shapes):
            geom, value = shape
            geom_sh = sh_geom.shape(geom)
            geoms.append(geom_sh)
            
        # If there are polygons found, convert to positive mask records
        if len(geoms) > 0:
            for geom in geoms:
                label_records.append({'geometry': geom, 
                                      'descr': label_type,
                                      'burninmask': 1,
                                      'usebounds': 1})
        else:
            # Nothing found, so it is an all-black, "negative" mask...
            label_records.append({'geometry': sh_geom.box(xmin, ymin, 
                                                          xmax, ymax), 
                                  'descr': label_type,
                                  'burninmask': 0,
                                  'usebounds': 1})
            
    # Convert to geodataframe and write to file 
    logger.info(f"Found {len(label_records)} labels, write to file: {output_filepath}")
    labels_gdf = gpd.GeoDataFrame(label_records, 
                                  columns=['geometry', 'descr', 
                                           'burninmask', 'usebounds'])
    if labels_gdf.crs is None:
        labels_gdf.crs = projection_if_missing
    
    # Cleanup data (dissolve + simplify)
    labels_gdf = labels_gdf.dissolve(by=['descr', 'burninmask', 'usebounds'])
    labels_gdf.reset_index(inplace=True)
    # assert to evade pyLance warning
    assert isinstance(labels_gdf, gpd.GeoDataFrame)
    labels_gdf = labels_gdf.explode()
    labels_gdf.geometry = labels_gdf.geometry.simplify(0.5)
    #labels_gdf['geometry'] = labels_gdf.geometry.apply(lambda geom: vh.simplify_visval(geom, 2))
        
    # Write result to file
    #logger.debug(f"Write the {len(geoms_gdf)} geoms, of types {geoms_gdf.geometry.type.unique()} in geoms_gdf to file")
    if os.path.exists(output_filepath):
        os.remove(output_filepath)
    labels_gdf.to_file(output_filepath, driver="ESRI Shapefile")
          
if __name__ == '__main__':

    # Main project dir
    subject = "greenhouses"
    base_dir = "X:\\Monitoring\\OrthoSeg\\"
    traindata_type = "validation"
    
    # Init relevant dirs and filenames
    project_dir = os.path.join(base_dir, subject)
    train_dir = os.path.join(project_dir, "training")
    train_type_dir = os.path.join(train_dir, traindata_type)
    mask_dir = os.path.join(train_type_dir, "mask")    
    output_filename = f"{subject}_{traindata_type}labels.shp"
    output_filepath = os.path.join(mask_dir, output_filename)

    # Main initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(Path(log_dir), __name__)
    
    # Now vectorize the masks
    vectorize_masks(input_image_dir=Path(mask_dir),
                    output_filepath=Path(output_filepath),
                    projection_if_missing='epsg:31370')

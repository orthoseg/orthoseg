# -*- coding: utf-8 -*-
"""
@author: Pieter Roggemans
"""

from __future__ import print_function
import logging
import os
import shutil
import glob
import math
import random
import datetime
import filecmp

import numpy as np
#import pandas as pd

import shapely.geometry as sh_geom
#import shapely.ops as sh_ops
import fiona
import rasterio as rio
import rasterio.features as rio_features
import owslib
import geopandas as gpd

import log_helper
import ows_helper
import geofile_helper

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def prepare_traindatasets(input_vector_label_filepath: str,
                 wms_server_url: str,
                 wms_server_layer: str,
                 output_basedir: str,
                 image_subdir: str = "image",
                 mask_subdir: str = "mask",
                 wms_server_layer_style: str = "default",
                 image_srs_pixel_x_size: int = 0.25,
                 image_srs_pixel_y_size: int = 0.25,
                 image_pixel_width: int = 512,
                 image_pixel_height: int = 512,
                 max_samples: int = 5000,
                 burn_value: int = 255,
                 output_filelist_csv: str = '',
                 output_keep_nested_dirs: bool = False,
                 force: bool = False) -> (str, int):
    """
    This function prepares training data for the vector labels provided.

    It will:
        * get orthophoto's from a WMS server
        * create the corresponding label mask for each orthophoto
    """
    # First determine the corrent data version based on existing output data 
    # dir(s)
    output_dirs = glob.glob(f"{output_basedir}_*")
    if len(output_dirs) == 0:
        dataversion_new = 1
    else:
        # Get the output dir with the highest version (=first if sorted desc)
        output_dir_mostrecent = sorted(output_dirs, reverse=True)[0]
        output_subdir_mostrecent = os.path.basename(output_dir_mostrecent)
        dataversion_mostrecent = int(output_subdir_mostrecent.split('_')[1])
        dataversion_new = dataversion_mostrecent + 1
        
        # If the input vector label file didn't change since previous run 
        # dataset can be reused
        output_vector_mostrecent_filepath = os.path.join(
                output_dir_mostrecent, 
                os.path.basename(input_vector_label_filepath))
        if(os.path.exists(output_vector_mostrecent_filepath)
           and geofile_helper.cmp(input_vector_label_filepath, 
                                  output_vector_mostrecent_filepath)):
            logger.info(f"RETURN: input vector label file isn't changed since last prepare_traindatasets, so no need to recreate")
            return output_dir_mostrecent, dataversion_mostrecent
                
    # Create the output dir's if they don't exist yet...
    output_dir = f"{output_basedir}_{dataversion_new:02d}"
    output_image_dir = os.path.join(output_dir, image_subdir)
    output_mask_dir = os.path.join(output_dir, mask_subdir)
    for dir in [output_dir, output_mask_dir, output_image_dir]:
        if dir and not os.path.exists(dir):
            os.makedirs(dir)

    # Copy the vector file(s) to the dest dir so we keep knowing which file was
    # used to create the dataset
    geofile_helper.copy(input_vector_label_filepath, output_dir)
    
    # Open vector layer
    logger.info(f"Open vector file {input_vector_label_filepath}")
    input_label_gdf = gpd.read_file(input_vector_label_filepath)

    # Get the srs to use from the input vectors...
    img_srs = input_label_gdf.crs['init']
        
    # Now loop over label polygons to create the training/validation data
    wms = owslib.wms.WebMapService(wms_server_url, version='1.3.0')
    image_srs_width = math.fabs(image_pixel_width*image_srs_pixel_x_size)   # tile width in units of crs => 500 m
    image_srs_height = math.fabs(image_pixel_height*image_srs_pixel_y_size) # tile height in units of crs => 500 m
    
    # Create list with only the input labels that are positive examples, as 
    # are the only ones that will need to be burned in the mask
    #is_positive_eg
    labels_to_burn_gdf = input_label_gdf[input_label_gdf['burninmask'] == 1]
    labels_to_use_for_bounds_gdf = input_label_gdf[input_label_gdf['usebounds'] == 1]
    
    # Loop trough all train labels to get an image for each of them
    nb_todo = len(input_label_gdf)
    nb_processed = 0
    logger.info(f"Get images for {nb_todo} labels")   
    created_images_gdf = gpd.GeoDataFrame()
    created_images_gdf['geometry'] = None
    start_time = datetime.datetime.now()
    for i, label_geom in enumerate(labels_to_use_for_bounds_gdf.geometry):

        # TODO: now just the top-left part of the labeled polygon is taken
        # as a start... ideally make sure we use it entirely for cases with
        # no abundance of data
        # Make sure the image requested is of the correct size
        geom_bounds = label_geom.bounds
        xmin = geom_bounds[0]-(geom_bounds[0]%image_srs_pixel_x_size)-10
        ymin = geom_bounds[1]-(geom_bounds[1]%image_srs_pixel_y_size)-10
        xmax = xmin + image_srs_width
        ymax = ymin + image_srs_height
        img_bbox = sh_geom.box(xmin, ymin, xmax, ymax)
        
        # Skip the bbox if it overlaps with any already created images. 
        if created_images_gdf.intersects(img_bbox).any():
            logger.debug(f"Bounds overlap with already created image, skip: {img_bbox}")
            continue
        else:
            created_images_gdf = created_images_gdf.append({'geometry': img_bbox}, 
                                                           ignore_index=True)

        # Now really get the image
        logger.debug(f"Get image for coordinates {img_bbox.bounds}")
        image_filepath = ows_helper.getmap_to_file(wms=wms,
                               layers=[wms_server_layer],
                               output_dir=output_image_dir,
                               srs=img_srs,
                               bbox=img_bbox.bounds,
                               size=(image_pixel_width, image_pixel_height),
                               image_format=ows_helper.FORMAT_TIFF,
                               transparent=False)

        # Create a mask corresponding with the image file
        # image_filepath can be None if file existed already, so check if not None...
        if image_filepath:
            mask_filepath = image_filepath.replace(output_image_dir, 
                                                   output_mask_dir)
            _create_mask(input_vector_label_list=labels_to_burn_gdf.geometry,
                         input_image_filepath=image_filepath,
                         output_mask_filepath=mask_filepath,
                         burn_value=burn_value,
                         force=force)
        
        # Log the progress and prediction speed
        time_passed = (datetime.datetime.now()-start_time).seconds
        if time_passed > 0 and nb_processed > 0:
            processed_per_hour = (nb_processed/time_passed) * 3600
            hours_to_go = (int)((nb_todo - i)/processed_per_hour)
            min_to_go = (int)((((nb_todo - i)/processed_per_hour)%1)*60)
            print(f"{hours_to_go}:{min_to_go} left for {nb_todo-i} of {nb_todo} at {processed_per_hour:0.0f}/h")
            
    return output_dir, dataversion_new

def create_masks(input_vector_label_filepath: str,
                 input_image_dir: str,
                 output_mask_dir: str,
                 output_image_dir: str = None,
                 burn_value: int = 255,
#                 output_filelist_csv: str = '',
                 output_keep_nested_dirs: bool = False,
                 force: bool = False):
    """
    Create masks for the extents of the files in the input folder, based on
    the vector data in the input vector file.

    Args

    """

    # Create output dirs
    for dir in [output_mask_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Load vector input file with the labels
    # Open vector layer
    logger.info(f"Open vector file {input_vector_label_filepath}")
    input_vector_labels = fiona.open(input_vector_label_filepath)

    # Convert to lists of shapely geometries
    input_labels = [(sh_geom.shape(input_label['geometry']), input_label['properties']['CODE_OBJ']) for input_label in input_vector_labels]
    input_labels = []
    for input_label in input_vector_labels:
        input_labels.append(sh_geom.shape(input_label['geometry']))

    # iterate through images, convert to 8-bit, and create masks
    filelist = []
    search_string = f"{input_image_dir}{os.sep}**{os.sep}*.tif"
    image_filepaths = glob.glob(search_string, recursive=True)
    for i, image_filepath in enumerate(image_filepaths):

        if i%50 == 0:
            logger.info(f"Processing file nb {i} of {len(image_filepaths)}")

        # Create output file name
        output_image_filepath = None
        if output_keep_nested_dirs:
            output_mask_filepath = image_filepath.replace(input_image_dir, output_mask_dir)
            if output_image_dir:
                output_image_filepath = image_filepath.replace(input_image_dir, output_image_dir)
        else:
            filename = os.path.split(image_filepath)[1]
            output_mask_filepath = os.path.join(output_mask_dir, filename)
            if output_image_dir:
                output_image_filepath = os.path.join(output_image_dir, filename)

        logger.debug(f"Process image_filepath: {image_filepath}, with output_mask_filepath: {output_mask_filepath}, output_image_filepath: {output_image_filepath}")

        # Check if output dir exists already
        output_mask_dir = os.path.split(output_mask_filepath)[0]
        for dir in [output_mask_dir, output_image_dir]:
            if dir and not os.path.exists(dir):
                os.mkdir(dir)

        # Create mask file
        ret_val = _create_mask(input_vector_label_layer=input_labels,
                               input_image_filepath=image_filepath,
                               output_mask_filepath=output_mask_filepath,
                               output_image_filepath=output_image_filepath,
                               burn_value=burn_value,
                               minimum_pct_labeled=1,
                               force=force)

        # Add to list of files to output later on
        if ret_val is True:
            image_filename = os.path.split(image_filepath)[1]
            filelist.append([image_filename, image_filepath, image_filepath,
                             output_mask_filepath, output_mask_filepath])

    '''
    # Put list in dataframe and save to csv
    df_filelist = pd.DataFrame(filelist, columns=['image_filename', 'image_filepath',
                                                  'image_visible_filepath',
                                                  'mask_filepath', 'mask_visible_filepath'])
    if len(df_filelist) > 0:
        df_filelist.to_csv(output_filelist_csv, index=False)
    '''
    return filelist

###############################################################################

def _create_mask(input_vector_label_list,
                 input_image_filepath: str,
                 output_mask_filepath: str,
                 output_imagecopy_filepath: str = None,
                 burn_value: int = 255,
                 minimum_pct_labeled: float = 0.0,
                 force: bool = False) -> bool:
    # TODO: only supports one class at the moment.

    # If file exists already and force is False... stop.
    if(force is False
       and os.path.exists(output_mask_filepath)):
        logger.debug(f"Output file already exist, and force is False, return: {output_mask_filepath}")
        return

    # Create a mask corresponding with the image file
    # First read the properties of the input image to copy them for the output
    logger.debug("Create mask and write to file")
    with rio.open(input_image_filepath) as image_ds:
        image_profile = image_ds.profile
        image_transform_affine = image_ds.transform

    # Use meta attributes of the source image, but set band count to 1,
    # dtype to uint8 and specify LZW compression.
    image_profile.update(dtype=rio.uint8, count=1, compress='lzw', nodata=0)

    # this is where we create a generator of geom, value pairs to use in rasterizing
#    shapes = ((geom,value) for geom, value in zip(counties.geometry, counties.LSAD_NUM))
    burned = rio_features.rasterize(shapes=input_vector_label_list, fill=0,
                default_value=burn_value, transform=image_transform_affine,
                out_shape=(image_profile['width'], image_profile['height']))

    # Check of the mask meets the requirements to be written...
    nb_pixels = np.size(burned, 0) * np.size(burned, 1)
    nb_pixels_data = nb_pixels - np.sum(burned == 0)  #np.count_nonnan(image == NoData_value)
    logger.debug(f"nb_pixels: {nb_pixels}, nb_pixels_data: {nb_pixels_data}, pct data: {nb_pixels_data / nb_pixels}")

    if (nb_pixels_data / nb_pixels >= minimum_pct_labeled):
        # Write the labeled mask
        with rio.open(output_mask_filepath, 'w', **image_profile) as mask_ds:
            mask_ds.write(burned, 1)

        # Copy the original image if wanted...
        if output_imagecopy_filepath:
            shutil.copyfile(input_image_filepath, output_imagecopy_filepath)
        return True
    else:
        return False

###############################################################################
if __name__ == "__main__":
    
    # General initialisations for the segmentation project
    segment_subject = "horsetracks"
    segment_subject = "greenhouses"
    
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"
    project_dir = os.path.join(base_dir, segment_subject)
    
    # Main initialisation of the logging
    # Log dir
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info("Start loading images")
    
    # WMS server we can use to get the image data
    WMS_SERVER_URL = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'
    
    # Input label data
    input_labels_dir = os.path.join(project_dir, 'input_labels')
    input_labels_filename = f"{segment_subject}_trainlabels.shp"
    #input_labels_filename = "labels.geojson"
    
    input_labels_filepath = os.path.join(input_labels_dir,
                                         input_labels_filename)
    
    # The subdirs where the images and masks can be found by convention for training and validation
    train_dir = os.path.join(project_dir, "train_new")
    
    # If the training data doesn't exist yet, create it
    force_create_train_data = True 
    
    if(force_create_train_data 
       or not os.path.exists(train_dir)):
        logger.info('Prepare train and validation data')
        prepare_traindatasets(
                input_vector_label_filepath=input_labels_filepath,
                wms_server_url=WMS_SERVER_URL,
                wms_server_layer='ofw',
                output_dir=train_dir,
                force=force_create_train_data)
    else:
        logger.info("Train data exists already, stop...")

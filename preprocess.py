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

import numpy as np
#import pandas as pd

import shapely.geometry as sh_geom
import fiona
import rasterio as rio
import rasterio.features as rio_features
import owslib

import ows_helper

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def prepare_training_data(input_vector_label_filepath: str,
                 wms_server_url: str,
                 wms_server_layer: str,
                 output_image_dir: str,
                 output_mask_dir: str,
                 wms_server_layer_style: str = 'default',
                 image_srs_pixel_x_size: int = 0.25,
                 image_srs_pixel_y_size: int = 0.25,
                 image_pixel_width: int = 1000,
                 image_pixel_height: int = 1000,                                 
                 max_samples: int = 500,
                 burn_value: int = 255,
                 output_filelist_csv: str = '',
                 output_keep_nested_dirs: bool = False,
                 force: bool = False):
    """
    This function prepares training data for the vector labels provided.
    
    It will:
        * get orthophoto's from a WMS server
        * create the corresponding label mask for each orthophoto
    """
        # Create the output dir's if they don't exist yet...
    for dir in [output_mask_dir]:
        if dir and not os.path.exists(dir):
            os.mkdir(dir)
    
    # Open vector layer
    logger.info(f"Open vector file {input_vector_label_filepath}")
    input_vector_label_data = fiona.open(input_vector_label_filepath)
    
    # Get the srs to use from the input vectors...
    image_srs = input_vector_label_data.crs['init']
        
    # Convert to lists of shapely geometries
#    input_labels = [(sh_geom.shape(input_label['geometry']), input_label['properties']['CODE_OBJ']) for input_label in input_vector_labels]
    input_labels = []
    for input_vector_label_row in input_vector_label_data:
        input_labels.append(sh_geom.shape(input_vector_label_row['geometry']))
    
    # Now loop over label polygons to create the training/validation data
    wms = owslib.wms.WebMapService(wms_server_url, version='1.3.0')
    image_srs_width = math.fabs(image_pixel_width*image_srs_pixel_x_size)   # tile width in units of crs => 500 m
    image_srs_height = math.fabs(image_pixel_height*image_srs_pixel_y_size) # tile height in units of crs => 500 m
    
    # Get list of max_samples random samples, and get those images
    random_labels = random.sample(input_labels, max_samples)
    for label_poly in random_labels:

        # TODO: now just the top-left part of the labeled polygon is taken
        # as a start... ideally make sure we use it entirely for cases with 
        # no abundance of data        
        # Make sure the image requested is of the correct size
        poly_bounds = label_poly.bounds
        image_xmin = poly_bounds[0]-(poly_bounds[0]%image_srs_pixel_x_size)-10
        image_ymin = poly_bounds[1]-(poly_bounds[1]%image_srs_pixel_y_size)-10
        image_xmax = image_xmin + image_srs_width
        image_ymax = image_ymin + image_srs_height
        image_bounds = (image_xmin, image_ymin, image_xmax, image_ymax)
           
        # Now really get the image
        image_filepath = ows_helper.getmap_to_file(wms=wms,
                       layers=[wms_server_layer],
                       output_dir=output_image_dir,
                       srs=image_srs,
                       bbox=image_bounds,
                       size=(image_pixel_width, image_pixel_height),
                       format=ows_helper.FORMAT_GEOTIFF,
                       transparent=False)

        # Create a mask corresponding with the image file
        mask_filepath = image_filepath.replace(output_image_dir, output_mask_dir)
        _create_mask(input_vector_label_list=input_labels,
                     input_image_filepath=image_filepath,
                     output_mask_filepath=mask_filepath,
                     burn_value=burn_value,
                     force=force)
    
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
        image_transform_affine = image_ds.affine

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
    raise Exception("Not implemented!")

# -*- coding: utf-8 -*-
"""
Module to prepare the training datasets.
"""

from __future__ import print_function
import logging
import os
import shutil
import glob
import math
import datetime

import fiona
import geopandas as gpd
import numpy as np
import owslib
import rasterio as rio
import rasterio.features as rio_features
import shapely.geometry as sh_geom

from orthoseg.helpers import log_helper
from orthoseg.util import ows_util
from orthoseg.util import geofile_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def prepare_traindatasets(
        input_vector_label_filepath: str,
        image_datasources: dict,
        default_image_datasource_code: str,
        output_basedir: str,
        image_subdir: str = "image",
        mask_subdir: str = "mask",
        image_pixel_x_size: int = 0.25,
        image_pixel_y_size: int = 0.25,
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
        
    Returns a tuple with (output_dir, dataversion):
            output_dir: the dir where the traindataset was created/found
            dataversion: a version number for the dataset created/found

    Args
        input_vector_label_filepath: filepath to the vector labels to prepare train dataset for
        wms_server_url: WMS server where the images can be fetched from
        wms_layername: layername on the WMS server to use
        output_basedir: the base dir where the train dataset needs to be written to 

    """
    # Check if the input file exists, if not, return
    if not os.path.exists(input_vector_label_filepath):
        message = f"Input file doesn't exist, so do nothing and return: {input_vector_label_filepath}"
        logger.info(message)
        raise Exception(message)
    
    # Determine the current data version based on existing output data dir(s), but ignore dirs ending on _ERROR
    output_dirs = glob.glob(f"{output_basedir}_*")
    output_dirs = [output_dir for output_dir in output_dirs if output_dir.endswith('_BUSY') is False]
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
                output_dir_mostrecent, os.path.basename(input_vector_label_filepath))
        if(os.path.exists(output_vector_mostrecent_filepath)
           and geofile_util.cmp(input_vector_label_filepath, 
                                  output_vector_mostrecent_filepath)):
            logger.info(f"RETURN: input vector label file isn't changed since last prepare_traindatasets, so no need to recreate")
            return output_dir_mostrecent, dataversion_mostrecent
                
    # Create the output dir's if they don't exist yet...
    output_dir = f"{output_basedir}_{dataversion_new:02d}"
    output_tmp_dir = f"{output_basedir}_{dataversion_new:02d}_BUSY"
    output_tmp_image_dir = os.path.join(output_tmp_dir, image_subdir)
    output_tmp_mask_dir = os.path.join(output_tmp_dir, mask_subdir)

    # Prepare the output dir...
    if os.path.exists(output_tmp_dir):
        shutil.rmtree(output_tmp_dir)
    for dir in [output_tmp_dir, output_tmp_mask_dir, output_tmp_image_dir]:
        if dir and not os.path.exists(dir):
            os.makedirs(dir)

    try:
        # Copy the vector file(s) to the dest dir so we keep knowing which file was
        # used to create the dataset
        geofile_util.copy(input_vector_label_filepath, output_tmp_dir)
        
        # Open vector layer
        logger.debug(f"Open vector file {input_vector_label_filepath}")
        input_label_gdf = gpd.read_file(input_vector_label_filepath)

        # Get the srs to use from the input vectors...
        img_srs = input_label_gdf.crs['init']
            
        # Now loop over label polygons to create the training/validation data
        image_srs_width = math.fabs(image_pixel_width*image_pixel_x_size)   # tile width in units of crs => 500 m
        image_srs_height = math.fabs(image_pixel_height*image_pixel_y_size) # tile height in units of crs => 500 m
        
        # Create list with only the input labels that are positive examples, as 
        # are the only ones that will need to be burned in the mask
        #is_positive_eg
        labels_to_burn_gdf = input_label_gdf[input_label_gdf['burninmask'] == 1]
        labels_to_use_for_bounds_gdf = input_label_gdf[input_label_gdf['usebounds'] == 1]
        
        # Loop trough all train labels to get an image for each of them
        nb_todo = len(input_label_gdf)
        nb_processed = 0
        logger.info(f"Get images for {nb_todo} labels in {os.path.basename(input_vector_label_filepath)}")
        created_images_gdf = gpd.GeoDataFrame()
        created_images_gdf['geometry'] = None
        start_time = datetime.datetime.now()
        wms_servers = {}
        for i, label_tuple in enumerate(labels_to_use_for_bounds_gdf.itertuples()):
            
            # TODO: now just the top-left part of the labeled polygon is taken
            # as a start... ideally make sure we use it entirely for cases with
            # no abundance of data
            # Make sure the image requested is of the correct size
            label_geom = label_tuple.geometry            
            geom_bounds = label_geom.bounds
            xmin = geom_bounds[0]-(geom_bounds[0]%image_pixel_x_size)-10
            ymin = geom_bounds[1]-(geom_bounds[1]%image_pixel_y_size)-10
            xmax = xmin + image_srs_width
            ymax = ymin + image_srs_height
            img_bbox = sh_geom.box(xmin, ymin, xmax, ymax)
            
            # Skip the bbox if it overlaps with any already created images. 
            if created_images_gdf.intersects(img_bbox).any():
                logger.debug(f"Bounds overlap with already created image, skip: {img_bbox}")
                continue
            else:
                created_images_gdf = created_images_gdf.append(
                        {'geometry': img_bbox}, ignore_index=True)

            # If an image layer is specified, use that layer
            image_datasource_code = default_image_datasource_code
            if 'image' in label_tuple._fields:
                if(getattr(label_tuple, 'image') is not None 
                   and getattr(label_tuple, 'image') != ''):
                    image_datasource_code = getattr(label_tuple, 'image')

            # If the wms to be used hasn't been initialised yet
            if image_datasource_code not in wms_servers:
                wms_servers[image_datasource_code] = owslib.wms.WebMapService(
                        url=image_datasources[image_datasource_code]['wms_server_url'], 
                        version=image_datasources[image_datasource_code]['wms_version'])
                                            
            # Now really get the image
            logger.debug(f"Get image for coordinates {img_bbox.bounds}")
            image_filepath = ows_util.getmap_to_file(
                    wms=wms_servers[image_datasource_code],
                    layers=image_datasources[image_datasource_code]['wms_layernames'],
                    styles=image_datasources[image_datasource_code]['wms_layerstyles'],
                    output_dir=output_tmp_image_dir,
                    srs=img_srs,
                    bbox=img_bbox.bounds,
                    size=(image_pixel_width, image_pixel_height),
                    image_format=ows_util.FORMAT_JPEG,
                    image_pixels_ignore_border=image_datasources[image_datasource_code]['image_pixels_ignore_border'],
                    transparent=False)

            # Create a mask corresponding with the image file
            # image_filepath can be None if file existed already, so check if not None...
            if image_filepath:
                mask_filepath = image_filepath.replace(output_tmp_image_dir, output_tmp_mask_dir)
                _create_mask(
                        input_vector_label_list=labels_to_burn_gdf.geometry,
                        input_image_filepath=image_filepath,
                        output_mask_filepath=mask_filepath,
                        burn_value=burn_value,
                        force=force)
            
            # Log the progress and prediction speed
            nb_processed += 1
            time_passed = (datetime.datetime.now()-start_time).total_seconds()
            if time_passed > 0 and nb_processed > 0:
                processed_per_hour = (nb_processed/time_passed) * 3600
                hours_to_go = (int)((nb_todo - i)/processed_per_hour)
                min_to_go = (int)((((nb_todo - i)/processed_per_hour)%1)*60)
                print(f"\r{hours_to_go}:{min_to_go} left for {nb_todo-i} of {nb_todo} at {processed_per_hour:0.0f}/h", 
                    end="", flush=True)
    except Exception as ex:
        message = "Error preparing dataset!"
        raise Exception(message) from ex

    # If everything went fine, rename output_tmp_dir to the final output_dir
    os.rename(output_tmp_dir, output_dir)

    return output_dir, dataversion_new

def create_masks_for_images(
        input_vector_label_filepath: str,
        input_image_dir: str,
        output_basedir: str,
        image_subdir: str = 'image',
        mask_subdir: str = 'mask',
        burn_value: int = 255,
        force: bool = False):

    # Check if the input file exists, if not, return
    if not os.path.exists(input_vector_label_filepath):
        message = f"Input file doesn't exist, so do nothing and return: {input_vector_label_filepath}"
        raise Exception(message)
    # Check if the input file exists, if not, return
    if not os.path.exists(input_image_dir):
        message = f"Input image dir doesn't exist, so do nothing and return: {input_image_dir}"
        raise Exception(message)
    
    # Determine the current data version based on existing output data dir(s), but ignore dirs ending on _ERROR
    output_dirs = glob.glob(f"{output_basedir}_*")
    output_dirs = [output_dir for output_dir in output_dirs if output_dir.endswith('_BUSY') is False]
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
                output_dir_mostrecent, os.path.basename(input_vector_label_filepath))
        if(os.path.exists(output_vector_mostrecent_filepath)
           and geofile_util.cmp(input_vector_label_filepath, 
                                  output_vector_mostrecent_filepath)):
            logger.info(f"RETURN: input vector label file isn't changed since last prepare_traindatasets, so no need to recreate")
            return output_dir_mostrecent, dataversion_mostrecent
                
    # Create the output dir's if they don't exist yet...
    output_dir = f"{output_basedir}_{dataversion_new:02d}"
    output_tmp_dir = f"{output_basedir}_{dataversion_new:02d}_BUSY"
    output_tmp_image_dir = os.path.join(output_tmp_dir, image_subdir)
    output_tmp_mask_dir = os.path.join(output_tmp_dir, mask_subdir)

    # Prepare the output dir...
    if os.path.exists(output_tmp_dir):
        shutil.rmtree(output_tmp_dir)
    for dir in [output_tmp_dir, output_tmp_mask_dir, output_tmp_image_dir]:
        if dir and not os.path.exists(dir):
            os.makedirs(dir)

    # Copy the vector file(s) to the dest dir so we keep knowing which file was
    # used to create the dataset
    geofile_util.copy(input_vector_label_filepath, output_tmp_dir)
    
    # Open vector layer
    logger.debug(f"Open vector file {input_vector_label_filepath}")
    input_label_gdf = gpd.read_file(input_vector_label_filepath)

    # Create list with only the input labels that are positive examples, as 
    # are the only ones that will need to be burned in the mask
    labels_to_burn_gdf = input_label_gdf[input_label_gdf['burninmask'] == 1]
    
    # Loop trough input images
    input_image_filepaths = glob.glob(f"{input_image_dir}{os.sep}*.tif")
    logger.info(f"process {len(input_image_filepaths)} input images")
    for input_image_filepath in input_image_filepaths:
        _, input_image_filename = os.path.split(input_image_filepath)
        
        image_filepath = os.path.join(output_tmp_image_dir, input_image_filename)
        shutil.copyfile(input_image_filepath, image_filepath)

        mask_filepath = os.path.join(output_tmp_mask_dir, input_image_filename)
        _create_mask(
                input_vector_label_list=labels_to_burn_gdf.geometry,
                input_image_filepath=image_filepath,
                output_mask_filepath=mask_filepath,
                burn_value=burn_value,
                force=force)

def _create_mask(
        input_vector_label_list,
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
        image_output_profile = image_ds.profile
        image_transform_affine = image_ds.transform

    # Use meta attributes of the source image, but set band count to 1,
    # dtype to uint8 and specify LZW compression.
    image_output_profile.update(dtype=rio.uint8, count=1, compress='lzw', nodata=0)

    # Depending on the output type, set some extra field in the profile
    output_ext_lower = os.path.splitext(output_mask_filepath)[1].lower()
    if output_ext_lower == '.tif':
        image_output_profile.update(compress='lzw')
    elif output_ext_lower in ('.jpg', '.jpeg'):
        # Remark: I cannot get rid of the warning about sharing, no matter what
        image_output_profile["sharing"] = None
        if "tiled" in image_output_profile:
            del image_output_profile["tiled"]
        if "compress" in image_output_profile:
            del image_output_profile["compress"]
        if "interleave" in image_output_profile:
            del image_output_profile["interleave"]
        if "photometric" in image_output_profile:
            del image_output_profile["photometric"]

    # this is where we create a generator of geom, value pairs to use in rasterizing
    burned = rio_features.rasterize(shapes=input_vector_label_list, fill=0,
            default_value=burn_value, transform=image_transform_affine,
            out_shape=(image_output_profile['width'], image_output_profile['height']))

    # Check of the mask meets the requirements to be written...
    nb_pixels = np.size(burned, 0) * np.size(burned, 1)
    nb_pixels_data = nb_pixels - np.sum(burned == 0)  #np.count_nonnan(image == NoData_value)
    logger.debug(f"nb_pixels: {nb_pixels}, nb_pixels_data: {nb_pixels_data}, pct data: {nb_pixels_data / nb_pixels}")

    if (nb_pixels_data / nb_pixels >= minimum_pct_labeled):
        # Write the labeled mask
        with rio.open(output_mask_filepath, 'w', **image_output_profile) as mask_ds:
            mask_ds.write(burned, 1)

        # Copy the original image if wanted...
        if output_imagecopy_filepath:
            shutil.copyfile(input_image_filepath, output_imagecopy_filepath)
        return True
    else:
        return False

if __name__ == "__main__":
    raise Exception('Not implemented!')

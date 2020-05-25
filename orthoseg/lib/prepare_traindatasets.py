# -*- coding: utf-8 -*-
"""
Module to prepare the training datasets.
"""

from __future__ import print_function
import logging
import os
import shutil
import math
import datetime
from pathlib import Path
from typing import Optional, Tuple

import fiona
import pandas as pd
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
        label_files: dict,
        classes: dict,
        image_layers: dict,
        training_dir: Path,
        image_pixel_x_size: float = 0.25,
        image_pixel_y_size: float = 0.25,
        image_pixel_width: int = 512,
        image_pixel_height: int = 512,
        max_samples: int = 5000,
        output_filelist_csv: str = '',
        output_keep_nested_dirs: bool = False,
        force: bool = False) -> Tuple[Path, int]:
    """
    This function prepares training data for the vector labels provided.

    It will:
        * get orthophoto's from a WMS server
        * create the corresponding label mask for each orthophoto
        
    Returns a tuple with (output_dir, dataversion):
            output_dir: the dir where the traindataset was created/found
            dataversion: a version number for the dataset created/found

    Args
        label_files: paths to the files with label data and locations to generate images for
        wms_server_url: WMS server where the images can be fetched from
        wms_layername: layername on the WMS server to use
        output_basedir: the base dir where the train dataset needs to be written to 

    """
    image_srs_width = math.fabs(image_pixel_width*image_pixel_x_size)   # tile width in units of crs => 500 m
    image_srs_height = math.fabs(image_pixel_height*image_pixel_y_size) # tile height in units of crs => 500 m

    # Determine the current data version based on existing output data dir(s),
    # but ignore dirs ending on _ERROR
    output_dirs = training_dir.glob(f"[0-9]*/")
    output_dirs = [output_dir for output_dir in output_dirs if not '_BUSY' in output_dir.name]
    logger.info(f"output_dirs: {output_dirs}")
    if len(output_dirs) == 0:
        output_legacy_train_dirs = training_dir.glob(f"train_[0-9]*/")
        output_legacy_train_dirs = [output_dir for output_dir in output_legacy_train_dirs if output_dir.name.endswith('_BUSY') is False]
        if len(output_legacy_train_dirs) == 0:
            dataversion_new = 1
        else:
            # Get the output dir with the highest version (=first if sorted desc)
            output_legacy_dir_mostrecent = sorted(output_legacy_train_dirs, reverse=True)[0]
            dataversion_mostrecent = int(output_legacy_dir_mostrecent.name.split('_')[1])
            dataversion_new = dataversion_mostrecent + 1
    else:
        # Get the output dir with the highest version (=first if sorted desc)
        output_dir_mostrecent = sorted(output_dirs, reverse=True)[0]
        dataversion_mostrecent = int(output_dir_mostrecent.name)
        
        # If none of the input files changed since previous run, reuse dataset
        reuse = False
        for label_file_key in label_files:                  
            label_file = label_files[label_file_key]
            reuse = True
            labellocations_output_mostrecent_path = (
                    output_dir_mostrecent / label_file['locations_path'].name)
            labeldata_output_mostrecent_path = output_dir_mostrecent / label_file['data_path'].name
            if(not (labellocations_output_mostrecent_path.exists()
               and labeldata_output_mostrecent_path.exists()
               and geofile_util.cmp(
                        label_file['locations_path'], labellocations_output_mostrecent_path)
               and geofile_util.cmp(label_file['data_path'], labeldata_output_mostrecent_path))):
                logger.info(f"RETURN: input label file(s) changed since last prepare_traindatasets, recreate")
                reuse = False
                break
        if reuse == True:
            return (output_dir_mostrecent, dataversion_mostrecent)
        else:
            dataversion_new = dataversion_mostrecent + 1
    
    # Prepare the output basedir...
    for i in range(100):
        output_tmp_basedir = training_dir / f"{dataversion_new:02d}_BUSY_{i:02d}"
        if output_tmp_basedir.exists():
            try:
                shutil.rmtree(output_tmp_basedir)
            except:
                None
        if not output_tmp_basedir.exists():
            try:
                output_tmp_basedir.mkdir(parents=True)
                break
            except:
                None

    # Process all input files
    labellocations_gdf = None
    labeldata_gdf = None
    logger.info(label_files)
    for label_file_key in label_files:
        label_file = label_files[label_file_key]

        # Copy the vector files to the dest dir so we keep knowing which files 
        # were used to create the dataset
        geofile_util.copy(label_file['locations_path'], output_tmp_basedir)
        geofile_util.copy(label_file['data_path'], output_tmp_basedir)

        # Read label data and append to general dataframes
        logger.debug(f"Read label locations from {label_file['locations_path']}")
        file_labellocations_gdf = geofile_util.read_file(label_file['locations_path'])
        if file_labellocations_gdf is not None and len(file_labellocations_gdf) > 0:
            file_labellocations_gdf.loc[:, 'image_layer'] = label_file['image_layer']
            if labellocations_gdf is None:
                labellocations_gdf = file_labellocations_gdf
            else:
                labellocations_gdf = gpd.GeoDataFrame(
                        pd.concat([labellocations_gdf, file_labellocations_gdf], ignore_index=True),
                        crs=file_labellocations_gdf.crs)
        else:
            logger.warn(f"No label locations data found in {label_file['locations_path']}")
        logger.debug(f"Read label data from {label_file['data_path']}")
        file_labeldata_gdf = geofile_util.read_file(label_file['data_path'])
        if file_labeldata_gdf is not None and len(file_labeldata_gdf) > 0:
            file_labeldata_gdf.loc[:, 'image_layer'] = label_file['image_layer']
            if labeldata_gdf is None:
                labeldata_gdf = file_labeldata_gdf
            else:
                labeldata_gdf = gpd.GeoDataFrame(
                        pd.concat([labeldata_gdf, file_labeldata_gdf], ignore_index=True),
                        crs=file_labeldata_gdf.crs)
        else:
            logger.warn(f"No label data found in {label_file['data_path']}")

    # Get the srs to use from the input vectors...
    try:
        img_srs = labellocations_gdf.crs['init']
    except Exception as ex:
        logger.exception(f"Error getting crs from labellocations, labellocations_gdf.crs: {labellocations_gdf.crs}")
        raise ex
    
    # Create list with only the input labels that need to be burned in the mask
    if labeldata_gdf is not None and 'label_name' in labeldata_gdf.columns:
        # If there is a column 'label_name', filter on the labels provided
        labels_to_burn_gdf = (
                labeldata_gdf.loc[labeldata_gdf['label_name'].isin(classes)]).copy()
        labels_to_burn_gdf['burn_value'] = 0
        for label_name in classes:
            labels_to_burn_gdf.loc[(labels_to_burn_gdf['label_name'] == label_name),
                                   'burn_value'] = classes[label_name]['burn_value']
        if len(labeldata_gdf) != len(labels_to_burn_gdf):
            logger.warn(f"Number of labels to burn changed from {len(labeldata_gdf)} to {len(labels_to_burn_gdf)} with filter on classes: {classes}")
    elif len(classes) == 2:
        labels_to_burn_gdf = labeldata_gdf
        labels_to_burn_gdf.loc[:, 'burn_value'] = classes[list(classes)[1]]['burn_value']
    else:
        raise Exception(f"Column 'label_name' is mandatory in labeldata if multiple classes specified: {classes}")
                    
    # Prepare the different traindata types
    for traindata_type in ['train', 'validation', 'test']:
                   
        # Create the output dir's if they don't exist yet...
        output_tmp_dir = output_tmp_basedir / traindata_type
        output_tmp_image_dir = output_tmp_dir / 'image'
        output_tmp_mask_dir = output_tmp_dir / 'mask'

        # Create output dirs...
        for dir in [output_tmp_dir, output_tmp_mask_dir, output_tmp_image_dir]:
            if dir and not dir.exists():
                dir.mkdir(parents=True)

        try:
            # Get the label locations for this traindata type
            labels_to_use_for_bounds_gdf = (
                    labellocations_gdf[labellocations_gdf['traindata_type'] == traindata_type])
            
            # Loop trough all locations labels to get an image for each of them
            nb_todo = len(labels_to_use_for_bounds_gdf)
            nb_processed = 0
            logger.info(f"Get images for {nb_todo} {traindata_type} labels")
            created_images_gdf = gpd.GeoDataFrame()
            created_images_gdf['geometry'] = None
            start_time = datetime.datetime.now()
            wms_servers = {}
            for i, label_tuple in enumerate(labels_to_use_for_bounds_gdf.itertuples()):
                
                # TODO: update the polygon if it doesn't match the size of the image...
                # as a start... ideally make sure we use it entirely for cases with
                # no abundance of data
                # Make sure the image requested is of the correct size
                label_geom = label_tuple.geometry            
                geom_bounds = label_geom.bounds
                xmin = geom_bounds[0]-(geom_bounds[0]%image_pixel_x_size)
                ymin = geom_bounds[1]-(geom_bounds[1]%image_pixel_y_size)
                xmax = xmin + image_srs_width
                ymax = ymin + image_srs_height
                img_bbox = sh_geom.box(xmin, ymin, xmax, ymax)
                image_layer = getattr(label_tuple, 'image_layer')

                # If the wms to be used hasn't been initialised yet
                if image_layer not in wms_servers:
                    wms_servers[image_layer] = owslib.wms.WebMapService(
                            url=image_layers[image_layer]['wms_server_url'], 
                            version=image_layers[image_layer]['wms_version'])
                                                
                # Now really get the image
                logger.debug(f"Get image for coordinates {img_bbox.bounds}")
                image_filepath = ows_util.getmap_to_file(
                        wms=wms_servers[image_layer],
                        layers=image_layers[image_layer]['wms_layernames'],
                        styles=image_layers[image_layer]['wms_layerstyles'],
                        output_dir=output_tmp_image_dir,
                        srs=img_srs,
                        bbox=img_bbox.bounds,
                        size=(image_pixel_width, image_pixel_height),
                        image_format=ows_util.FORMAT_PNG,
                        #image_format_save=ows_util.FORMAT_TIFF,
                        image_pixels_ignore_border=image_layers[image_layer]['image_pixels_ignore_border'],
                        transparent=False,
                        layername_in_filename=True)

                # Create a mask corresponding with the image file
                # image_filepath can be None if file existed already, so check if not None...
                if image_filepath is not None:
                    # Mask should not be in a lossy format!
                    mask_filepath = Path(str(image_filepath)
                            .replace(str(output_tmp_image_dir), str(output_tmp_mask_dir))
                            .replace('.jpg', '.png'))
                    nb_classes = len(classes)
                    # Only keep the labels that are meant for this image layer
                    labels_gdf = (labels_to_burn_gdf.loc[
                                        labels_to_burn_gdf['image_layer'] == image_layer]).copy()
                    if len(labels_gdf) == 0:
                        logger.info("No labels to be burned for this layer, this is weird!")
                    _create_mask(
                            input_image_filepath=image_filepath,
                            output_mask_filepath=mask_filepath,
                            labels_to_burn_gdf=labels_gdf,
                            nb_classes=nb_classes,
                            force=force)
                
                # Log the progress and prediction speed
                nb_processed += 1
                time_passed = (datetime.datetime.now()-start_time).total_seconds()
                if time_passed > 0 and nb_processed > 0:
                    processed_per_hour = (nb_processed/time_passed) * 3600
                    hours_to_go = (int)((nb_todo - i)/processed_per_hour)
                    min_to_go = (int)((((nb_todo - i)/processed_per_hour)%1)*60)
                    print(f"\r{hours_to_go:3d}:{min_to_go:2d} left for {nb_todo-i} of {nb_todo} at {processed_per_hour:0.0f}/h", 
                          end="", flush=True)
        except Exception as ex:
            message = "Error preparing dataset!"
            raise Exception(message) from ex

    # If everything went fine, rename output_tmp_dir to the final output_dir
    output_basedir = training_dir / f"{dataversion_new:02d}"
    os.rename(output_tmp_basedir, output_basedir)

    return (output_basedir, dataversion_new)

''' Not maintained!
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
    input_image_filepaths = glob.glob(f"{input_image_dir}/*.tif")
    logger.info(f"process {len(input_image_filepaths)} input images")
    for input_image_filepath in input_image_filepaths:
        _, input_image_filename = os.path.split(input_image_filepath)
        
        image_filepath = os.path.join(output_tmp_image_dir, input_image_filename)
        shutil.copyfile(input_image_filepath, image_filepath)

        mask_filepath = os.path.join(output_tmp_mask_dir, input_image_filename)
        _create_mask(
                input_image_filepath=image_filepath,
                output_mask_filepath=mask_filepath,
                labels_to_burn_gdf=labels_to_burn_gdf,
                force=force)
'''

def _create_mask(
        input_image_filepath: Path,
        output_mask_filepath: Path,
        labels_to_burn_gdf: gpd.geodataframe,
        nb_classes: int = 1,
        output_imagecopy_filepath: Optional[Path] = None,
        minimum_pct_labeled: float = 0.0,
        force: bool = False) -> Optional[bool]:

    # If file exists already and force is False... stop.
    if(force is False
       and output_mask_filepath.exists()):
        logger.debug(f"Output file already exist, and force is False, return: {output_mask_filepath}")
        return

    # Create a mask corresponding with the image file
    # First read the properties of the input image to copy them for the output
    logger.debug(f"Create mask to {output_mask_filepath}")
    with rio.open(input_image_filepath) as image_ds:
        image_input_profile = image_ds.profile
        image_transform_affine = image_ds.transform

    # Prepare the file profile for the mask depending on output type
    output_ext_lower = output_mask_filepath.suffix.lower()
    if output_ext_lower == '.tif':
        image_output_profile = rio.profiles.DefaultGTiffProfile(
                count=1, transform=image_transform_affine, crs=image_input_profile['crs'])
    if output_ext_lower == '.png':
        image_output_profile = rio.profiles.Profile(driver='PNG', count=1)
    else:
        raise Exception(f"Unsupported mask extension (should be a lossless format!): {output_ext_lower}")
    image_output_profile.update(
            width=image_input_profile['width'], height=image_input_profile['height'], 
            dtype=rio.uint8)

    # Burn the vectors in a mask
    burn_shapes = ((geom, value) 
            for geom, value in zip(labels_to_burn_gdf.geometry, labels_to_burn_gdf.burn_value) if geom is not None)
    try:
        mask_arr = rio_features.rasterize(
                shapes=burn_shapes, transform=image_transform_affine,
                dtype=rio.uint8, fill=0, 
                out_shape=(image_output_profile['width'], image_output_profile['height']))
    except Exception as ex:
        raise Exception(f"Error creating mask for {image_transform_affine}") from ex

    # Check if the mask meets the requirements to be written...
    if minimum_pct_labeled > 0:
        nb_pixels = np.size(mask_arr, 0) * np.size(mask_arr, 1)
        nb_pixels_data = nb_pixels - np.sum(mask_arr == 0)  #np.count_nonnan(image == NoData_value)
        logger.debug(f"nb_pixels: {nb_pixels}, nb_pixels_data: {nb_pixels_data}, pct data: {nb_pixels_data / nb_pixels}")

        if (nb_pixels_data / nb_pixels < minimum_pct_labeled):
            return False

    # Write the labeled mask
    image_output_profile = ows_util.get_cleaned_write_profile(image_output_profile)
    with rio.open(str(output_mask_filepath), 'w', **image_output_profile) as mask_ds:
        mask_ds.write(mask_arr, 1)
        
    return True

if __name__ == "__main__":
    raise Exception('Not implemented!')

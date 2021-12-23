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
import pprint
from typing import List, Optional, Tuple

from geofileops import geofile
from geofileops import geofileops
import pandas as pd
import geopandas as gpd
import numpy as np
import owslib
import owslib.wms
import owslib.util
from PIL import Image
import rasterio as rio
import rasterio.features as rio_features
import rasterio.profiles as rio_profiles
import shapely.geometry as sh_geom

from orthoseg.util import ows_util
from orthoseg.helpers.progress_helper import ProgressHelper

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

class LabelInfo:
    def __init__(self,
            locations_path: Path,
            polygons_path: Path,
            image_layer: str):
        self.locations_path = locations_path
        self.polygons_path = polygons_path
        self.image_layer = image_layer
    def __repr__(self):
       return f"LabelInfo with image_layer: {self.image_layer}, locations_path: {self.locations_path}, polygons_path: {self.polygons_path}"

def prepare_traindatasets(
        label_infos: List[LabelInfo],
        classes: dict,
        image_layers: dict,
        training_dir: Path,
        labelname_column: str,
        image_pixel_x_size: float = 0.25,
        image_pixel_y_size: float = 0.25,
        image_pixel_width: int = 512,
        image_pixel_height: int = 512,
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
        label_infos (List[LabelInfo]): paths to the files with label polygons 
            and locations to generate images for
        classes (dict): dict with the classes to detect as keys. The values 
            are the following:
                - labelnames: list of labels to use for this class
                - weight:
                - burn_value:
        image_layers (dict):
        training_dir (Path):
        labelname_column (str): the column where the label names are stored in 
            the polygon files
    """
    ### Init stuff ###
    auth = None
    ssl_verify = True
    if ssl_verify is False: 
        auth = owslib.util.Authentication(verify=ssl_verify)
        import urllib3
        urllib3.disable_warnings()
        logger.warn("SSL VERIFICATION IS TURNED OFF!!!")

    # Check if the first class is named "background"
    if len(classes) == 0:
        raise Exception("No classes specified")
    elif list(classes)[0].lower() != 'background':
        classes_str = pprint.pformat(classes, sort_dicts=False, width=50)
        raise Exception(f"By convention, the first class (the background) should be called 'background'!\n{classes_str}")

    image_crs_width = math.fabs(image_pixel_width*image_pixel_x_size)   # tile width in units of crs => 500 m
    image_crs_height = math.fabs(image_pixel_height*image_pixel_y_size) # tile height in units of crs => 500 m

    ### Check if the latest version of training data is already ok ### 
    # Determine the current data version based on existing output data dir(s),
    # If dir ends on _TMP_* ignore it, as it (probably) ended with an error.
    output_dirs = training_dir.glob(f"[0-9]*/")
    output_dirs = [output_dir for output_dir in output_dirs if not '_TMP_' in output_dir.name]
    
    reuse_traindata = False
    if len(output_dirs) == 0:
        dataversion_new = 1
    else:
        # Get the output dir with the highest version (=first if sorted desc)
        output_dir_mostrecent = sorted(output_dirs, reverse=True)[0]
        dataversion_mostrecent = int(output_dir_mostrecent.name)
        
        # If none of the input files changed since previous run, reuse dataset
        for label_file in label_infos:
            reuse_traindata = True
            labellocations_output_mostrecent_path = (
                    output_dir_mostrecent / label_file.locations_path.name)
            labeldata_output_mostrecent_path = output_dir_mostrecent / label_file.polygons_path.name
            if(not (labellocations_output_mostrecent_path.exists()
                    and labeldata_output_mostrecent_path.exists()
                    and geofile.cmp(label_file.locations_path, labellocations_output_mostrecent_path)
                    and geofile.cmp(label_file.polygons_path, labeldata_output_mostrecent_path))):
                reuse_traindata = False
                break
        if reuse_traindata == True:
            dataversion_new = dataversion_mostrecent
            logger.info(f"Input label file(s) haven't changed since last prepare_traindatasets, so reuse version {dataversion_new}")
        else:
            dataversion_new = dataversion_mostrecent + 1
            logger.info(f"Input label file(s) changed since last prepare_traindatasets, so create new training data version {dataversion_new}")

            # In this case, first check all input files if they are valid...
            invalid_geom_paths = []
            for label_file in label_infos:
                is_valid = geofileops.isvalid(label_file.locations_path, force=True)
                if is_valid is False:
                    invalid_geom_paths.append(str(label_file.locations_path))
                is_valid = geofileops.isvalid(label_file.polygons_path, force=True)
                if is_valid is False:
                    invalid_geom_paths.append(str(label_file.polygons_path))
            if len(invalid_geom_paths) > 0:
                raise Exception(f"Invalid geometries found in: {', '.join(invalid_geom_paths)}")
    
    # Determine the output dir 
    training_dataversion_dir = training_dir / f"{dataversion_new:02d}"

    # If the train data is already ok, just return 
    if reuse_traindata is True:
        return (training_dataversion_dir, dataversion_new)

    ### The input labels have changed: copy to new train version ###
    # Create the training dataset first to a temp dir, so it can be 
    # removed/ignored if an error occurs while creating it.
    output_tmp_dir = create_tmp_dir(training_dir, f"{dataversion_new:02d}", remove_existing=True)
    
    labellocations_gdf = None
    labelpolygons_gdf = None
    logger.info(f"Label info: \n{pprint.pformat(label_infos, indent=4)}")
    for label_file in label_infos:

        # Copy the vector files to the dest dir so we keep knowing which files 
        # were used to create the dataset
        geofile.copy(label_file.locations_path, output_tmp_dir)
        geofile.copy(label_file.polygons_path, output_tmp_dir)

        # Read label data and append to general dataframes
        logger.debug(f"Read label locations from {label_file.locations_path}")
        file_labellocations_gdf = geofile.read_file(label_file.locations_path)
        if file_labellocations_gdf is not None and len(file_labellocations_gdf) > 0:
            file_labellocations_gdf.loc[:, 'filepath'] = str(label_file.locations_path)   # type: ignore
            file_labellocations_gdf.loc[:, 'image_layer'] = label_file.image_layer        # type: ignore
            # Remark: geopandas 0.7.0 drops the fid column internaly, so cannot be retrieved
            file_labellocations_gdf.loc[:, 'row_nb_orig'] = file_labellocations_gdf.index # type: ignore
            if labellocations_gdf is None:
                labellocations_gdf = file_labellocations_gdf
            else:
                labellocations_gdf = gpd.GeoDataFrame(
                        pd.concat([labellocations_gdf, file_labellocations_gdf], ignore_index=True),
                        crs=file_labellocations_gdf.crs)
        else:
            logger.warn(f"No label locations data found in {label_file.locations_path}")
        logger.debug(f"Read label data from {label_file.polygons_path}")
        file_labelpolygons_gdf = geofile.read_file(label_file.polygons_path)
        if file_labelpolygons_gdf is not None and len(file_labelpolygons_gdf) > 0:
            file_labelpolygons_gdf.loc[:, 'image_layer'] = label_file.image_layer # type: ignore
            if labelpolygons_gdf is None:
                labelpolygons_gdf = file_labelpolygons_gdf
            else:
                labelpolygons_gdf = gpd.GeoDataFrame(
                        pd.concat([labelpolygons_gdf, file_labelpolygons_gdf], ignore_index=True),
                        crs=file_labelpolygons_gdf.crs)
        else:
            logger.warn(f"No label data found in {label_file.polygons_path}")

    # Check if we ended up with labellocations...
    if labellocations_gdf is None:
        raise Exception("Not any labellocation found in the training data, so stop")

    # Get the crs to use from the input vectors...
    img_crs = None
    try:
        img_crs = labellocations_gdf.crs
    except Exception as ex:
        logger.exception(f"Error getting crs from labellocations, labellocation_gdf.crs: {labellocations_gdf.crs}")
        raise ex
    if img_crs is None:
        raise Exception(f"Error getting crs from labellocations, labellocation_gdf.crs: {labellocations_gdf.crs}")

    # Create list with only the input labels that need to be burned in the mask
    if labelpolygons_gdf is not None and labelname_column in labelpolygons_gdf.columns:
        # If there is a column labelname_column (default='label_name'), use the 
        # burn values specified in the configuration
        labels_to_burn_gdf = labelpolygons_gdf
        labels_to_burn_gdf['burn_value'] = None
        for classname in classes:
            labels_to_burn_gdf.loc[(labels_to_burn_gdf[labelname_column].isin(classes[classname]['labelnames'])),
                                   'burn_value'] = classes[classname]['burn_value']
        
        # If there are burn_values that are not filled out, log + stop!
        invalid_labelnames_gdf = labels_to_burn_gdf.loc[labels_to_burn_gdf['burn_value'].isnull()]
        if len(invalid_labelnames_gdf) > 0:
            raise Exception(f"Unknown labelnames (not in config) were found in {len(invalid_labelnames_gdf)} rows, so stop: {invalid_labelnames_gdf[labelname_column].unique()}")
        
        # Filter away rows that are going to burn 0, as this is useless...
        labels_to_burn_gdf = labels_to_burn_gdf.loc[labels_to_burn_gdf['burn_value'] != 0]

    elif len(classes) == 2:
        # There is no column with label names, but there are only 2 classes (background + subsject), so no problem...
        logger.info(f'Column with label names ({labelname_column}) not found, so use all polygons')
        labels_to_burn_gdf = labelpolygons_gdf
        labels_to_burn_gdf.loc[:, 'burn_value'] = 1 # type: ignore

    else:
        # There is no column with label names, but more than two classes, so stop.
        raise Exception(f"Column {labelname_column} is mandatory in labeldata if multiple classes specified: {classes}")

    # Check if we ended up with label data to burn.
    if labels_to_burn_gdf is None:
        raise Exception("Not any labelpolygon retained to burn in the training data, so stop")

    ### Now create the images/masks for the new train version ###
    # Prepare the image data for the different traindata types.
    for traindata_type in ['train', 'validation', 'test']:
        
        # If traindata exists already... continue
        output_imagedatatype_dir = output_tmp_dir / traindata_type
        output_imagedata_image_dir = output_imagedatatype_dir / 'image'
        output_imagedata_mask_dir = output_imagedatatype_dir / 'mask'

        # Create output dirs...
        for dir in [output_imagedatatype_dir, output_imagedata_mask_dir, output_imagedata_image_dir]:
            if dir and not dir.exists():
                dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get the label locations for this traindata type
            labels_to_use_for_bounds_gdf = (
                    labellocations_gdf[labellocations_gdf['traindata_type'] == traindata_type])
            
            # Loop trough all locations labels to get an image for each of them
            nb_todo = len(labels_to_use_for_bounds_gdf)
            logger.info(f"Get images for {nb_todo} {traindata_type} labels")
            created_images_gdf = gpd.GeoDataFrame()
            created_images_gdf['geometry'] = None
            wms_imagelayer_layersources = {}
            progress = ProgressHelper(message="prepare training images", nb_steps_total=nb_todo)
            for i, label_tuple in enumerate(labels_to_use_for_bounds_gdf.itertuples()):
                
                # TODO: update the polygon if it doesn't match the size of the image...
                # as a start... ideally make sure we use it entirely for cases with
                # no abundance of data
                # Make sure the image requested is of the correct size
                label_geom = label_tuple.geometry
                if label_geom is None:
                    logger.warn(f"No geometry found in file {label_tuple.filepath}, (zero based) row_nb_orig: {label_tuple.row_nb_orig}")
                    continue
                geom_bounds = label_geom.bounds
                xmin = geom_bounds[0]-(geom_bounds[0]%image_pixel_x_size)
                ymin = geom_bounds[1]-(geom_bounds[1]%image_pixel_y_size)
                xmax = xmin + image_crs_width
                ymax = ymin + image_crs_height
                img_bbox = sh_geom.box(xmin, ymin, xmax, ymax)
                image_layer = getattr(label_tuple, 'image_layer')

                # If the wms to be used hasn't been initialised yet
                if image_layer not in wms_imagelayer_layersources:
                    wms_imagelayer_layersources[image_layer] = []
                    for layersource in image_layers[image_layer]['layersources']:
                        wms_service = owslib.wms.WebMapService(
                                url=layersource['wms_server_url'], 
                                version=layersource['wms_version'],
                                auth=auth)
                        wms_imagelayer_layersources[image_layer].append(
                                ows_util.LayerSource(
                                        wms_service=wms_service,
                                        layernames=layersource['layernames'],
                                        layerstyles=layersource['layerstyles'],
                                        bands=layersource['bands'],
                                        random_sleep=layersource['random_sleep']))
                                                
                # Now really get the image
                logger.debug(f"Get image for coordinates {img_bbox.bounds}")
                image_filepath = ows_util.getmap_to_file(
                        layersources=wms_imagelayer_layersources[image_layer],
                        output_dir=output_imagedata_image_dir,
                        crs=img_crs,
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
                            .replace(str(output_imagedata_image_dir), str(output_imagedata_mask_dir))
                            .replace('.jpg', '.png'))
                    nb_classes = len(classes)
                    # Only keep the labels that are meant for this image layer
                    labels_gdf = (labels_to_burn_gdf.loc[
                                        labels_to_burn_gdf['image_layer'] == image_layer]).copy()
                    # assert to evade pyLance warning
                    assert isinstance(labels_gdf, gpd.GeoDataFrame)
                    if len(labels_gdf) == 0:
                        logger.info("No labels to be burned for this layer, this is weird!")
                    _create_mask(
                            input_image_filepath=image_filepath,
                            output_mask_filepath=mask_filepath,
                            labels_to_burn_gdf=labels_gdf,
                            nb_classes=nb_classes,
                            force=force)
                
                # Log the progress and prediction speed
                progress.step()

        except Exception as ex:
            raise ex

    # If everything went fine, rename output_tmp_dir to the final output_dir
    output_tmp_dir.rename(training_dataversion_dir)

    return (training_dataversion_dir, dataversion_new)

def create_tmp_dir(
        parent_dir: Path,
        dir_name: str,
        remove_existing: bool = False) -> Path:
    """
    Helper function to create a 'TMP' dir based on a directory name:
        parent_dir / <dir_name>_TMP_<sequence>
    
    Use: if you want to write data to a directory in a "transactional way", 
    it is the safest to write to a temp dir first, and then rename it to the 
    final name. This way, if a hard crash occurs while writing the data, it 
    is clear that the directory wasn't ready. Additionally, in case of a hard 
    crash, file locks can remain which makes it impossible to remove a 
    directory for a while.

    Args:
        parent_dir (Path): The dir to create the temp dir in.
        dir_name (str): The name of the dir to base the temp dir on.
        remove_existing (bool, optional): If True, existing TMP directories 
            will be removed if possible. Defaults to False.

    Raises:
        Exception: [description]

    Returns:
        Path: [description]
    """
    # The final dir should not exists yet!
    final_dir = parent_dir / dir_name
    if final_dir.exists():
        raise Exception(f"It is not supported to create a TMP dir for an existing dir: {final_dir}")
    
    # Try to delete all existing TMP dir's
    if remove_existing is True:
        existing_tmp_dirs = parent_dir.glob(f"{dir_name}_TMP_*")
        for existing_tmp_dir in existing_tmp_dirs:
            try:
                shutil.rmtree(existing_tmp_dir)
            except:
                tmp_dir = None

    # Create the TMP dir
    tmp_dir = None
    for i in range(100):
        tmp_dir = parent_dir / f"{dir_name}_TMP_{i:02d}"

        if tmp_dir.exists():
            # If it (still) exists try next sequence
            tmp_dir = None
            continue
        else:
            # If it doesn't exist, try to create it
            try:
                tmp_dir.mkdir(parents=True)
                break
            except:
                # If it fails to create, try next sequence
                tmp_dir = None
                continue
    
    # If no output tmp dir could be found/created... stop...
    if tmp_dir is None:
        raise Exception(f"Error creating/replacing TMP dir for {dir_name} in {parent_dir}")

    return tmp_dir

def _create_mask(
        input_image_filepath: Path,
        output_mask_filepath: Path,
        labels_to_burn_gdf: gpd.GeoDataFrame,
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
        image_output_profile = rio_profiles.DefaultGTiffProfile(
                count=1, transform=image_transform_affine, crs=image_input_profile['crs'])
    if output_ext_lower == '.png':
        image_output_profile = rio_profiles.Profile(driver='PNG', count=1)
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

    # Write the labeled mask as .png (so without transform/crs info)
    im = Image.fromarray(mask_arr)
    im.save(output_mask_filepath)
    
    return True

if __name__ == "__main__":
    raise Exception('Not implemented!')

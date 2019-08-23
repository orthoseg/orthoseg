# -*- coding: utf-8 -*-
"""
Module with functions for post-processing prediction masks towards polygons.
"""

import logging
import os
import glob
import shutil

import numpy as np
import rasterio as rio
import rasterio.features as rio_features
import shapely as sh
import geopandas as gpd

import orthoseg.vector.vector_helper as vh
import orthoseg.helpers.geofile as geofile_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------
"""
Not used anymore!

import skimage
import skimage.morphology       # Needs to be imported explicitly as it is a submodule
from scipy import ndimage

def region_segmentation(predicted_mask,
                        thresshold_ok: float = 0.5):
    
    if predicted_mask.dtype != np.float32:
        raise Exception("Input should be dtype = float32, not: {in_arr.dtype}")

    # ???
    elevation_map = skimage.filters.sobel(predicted_mask)
    
    # First apply some basic thressholds...
    markers = np.zeros_like(predicted_mask)
    markers[predicted_mask < thresshold_ok] = 1
    markers[predicted_mask >= thresshold_ok] = 2

    # Clean    
    segmentation = skimage.morphology.watershed(elevation_map, markers)   
    segmentation = segmentation.astype(np.uint8)
    
    # Remove holes in the mask
    segmentation = ndimage.binary_fill_holes(segmentation - 1)
    
    return segmentation
"""
def to_binary_uint8(in_arr, thresshold_ok):
    if in_arr.dtype != np.uint8:
        raise Exception("Input should be dtype = uint8, not: {in_arr.dtype}")
        
    # First copy to new numpy array, otherwise input array is changed
    out_arr = np.copy(in_arr)
    out_arr[out_arr >= thresshold_ok] = 255
    out_arr[out_arr < thresshold_ok] = 0
    
    return out_arr

'''
def clean_and_save_prediction(
        image_filepath: str,
        image_crs: str,
        image_transform,
        output_dir: str,
        image_pred_arr = None,
        image_pred_filepath = None,
        input_mask_dir: str = None,
        border_pixels_to_ignore: int = 0,
        evaluate_mode: bool = False,
        force: bool = False):

    logger.debug(f"Start postprocess for {image_pred_filepath}")
    
    # Either the filepath to a temp file with the prediction or the data 
    # itself need to be provided
    if(image_pred_arr is None
       and image_pred_filepath is None):
        message = f"postprocess_prediction needs either image_pred_arr or image_pred_filepath, now both are None"
        logger.error(message)
        raise Exception(message)

    # If no decent transform metadata, write warning
    if image_transform is None or image_transform[0] == 0:
        # If in evaluate mode... warning is enough, otherwise error!
        message = f"No transform found for {image_filepath}: {image_transform}"        
        if evaluate_mode:
            logger.warn(message)
        else:
            logger.error(message)
            raise Exception(message)          

    try:      

        # Prepare the filepath for the output
        input_image_dir, image_filename = os.path.split(image_filepath)
        image_filename_noext, image_ext = os.path.splitext(image_filename)
        
        if image_pred_arr is None:
            image_pred_arr = np.load(image_pred_filepath)
            os.remove(image_pred_filepath)
        
        # Check the number of channels of the output prediction
        # TODO: maybe param name should be clearer !!!
        n_channels = image_pred_arr.shape[2]
        if n_channels > 1:
            raise Exception(f"Not implemented: processing prediction output with multiple channels: {n_channels}")
        
        # Input should be in float32
        if image_pred_arr.dtype != np.float32:
            raise Exception(f"Image prediction should be in numpy.float32, but is in: {image_pred_arr.dtype}")

        # Reshape array from 4 dims (image_nb, width, height, nb_channels) to 2.
        image_pred_arr = image_pred_arr.reshape((image_pred_arr.shape[0], image_pred_arr.shape[1]))
                
        # Make the pixels at the borders of the prediction black so they are ignored
        if border_pixels_to_ignore and border_pixels_to_ignore > 0:
            image_pred_arr[0:border_pixels_to_ignore,:] = 0    # Left border
            image_pred_arr[-border_pixels_to_ignore:,:] = 0    # Right border
            image_pred_arr[:,0:border_pixels_to_ignore] = 0    # Top border
            image_pred_arr[:,-border_pixels_to_ignore:] = 0    # Bottom border
    
        # Check if the result is entirely black... if so no cleanup needed
        all_black = False
        thresshold_ok_float32 = 0.5
        thresshold_ok_uint8 = 128
        image_pred_uint8_cleaned_bin = None
        if not np.any(image_pred_arr >= thresshold_ok_float32):
            logger.debug('Prediction is entirely black!')
            all_black = True
        """
        else:
            # Cleanup the image so it becomes a clean 2 color one instead of grayscale
            logger.debug("Clean prediction")
            image_pred_uint8_cleaned_bin = region_segmentation(
                    image_pred_arr, thresshold_ok=thresshold_ok_float32)
            image_pred_uint8_cleaned_bin = image_pred_uint8_cleaned_bin * 255
            if not np.any(image_pred_uint8_cleaned_bin > thresshold_ok_uint8):
                logger.info('Prediction became entirely black!')
                all_black = True
        """
        
        # If not in evaluate mode and the prediction is all black, return
        if not evaluate_mode and all_black:
            logger.debug("All black prediction, no use saving this")
            return 

        # Convert to uint8 + create binary version
        image_pred_uint8 = (image_pred_arr * 255).astype(np.uint8)
        image_pred_uint8_bin = to_binary_uint8(image_pred_uint8, 
                                               thresshold_ok_uint8)
        image_pred_uint8_base10 = (image_pred_arr * 10).astype(np.uint8)
        image_pred_uint8_base10 = image_pred_uint8_base10 * 25

        if image_pred_uint8_cleaned_bin is None:
            image_pred_uint8_cleaned_bin = image_pred_uint8_bin

        # Make sure the output dir exists...
        os.makedirs(output_dir, exist_ok=True)
                
        # If in evaluate mode, put a prefix in the file name
        pred_prefix_str = ''
        if evaluate_mode:
            
            def jaccard_similarity(im1, im2):
                if im1.shape != im2.shape:
                    message = f"Shape mismatch: input have different shape: im1: {im1.shape}, im2: {im2.shape}"
                    logger.critical(message)
                    raise ValueError(message)
    
                intersection = np.logical_and(im1, im2)
                union = np.logical_or(im1, im2)
    
                sum_union = float(union.sum())
                if sum_union == 0.0:
                    # If 0 positive pixels in union: perfect prediction, so 1
                    return 1
                else:
                    sum_intersect = intersection.sum()
                    return sum_intersect/sum_union
    
            # If there is a mask dir specified... use the groundtruth mask
            if input_mask_dir and os.path.exists(input_mask_dir):
                # Read mask file and get all needed info from it...
                mask_filepath = image_filepath.replace(input_image_dir,
                                                       input_mask_dir)
                # Check if this file exists, if not, look for similar files
                if not os.path.exists(mask_filepath):
                    mask_filepath_noext = os.path.splitext(mask_filepath)[0]
                    files = glob.glob(mask_filepath_noext + '*')
                    if len(files) == 1:
                        mask_filepath = files[0]
                    else:
                        message = f"Error finding mask file with {mask_filepath_noext + '*'}: {len(files)} mask(s) found"
                        logger.error(message)
                        raise Exception(message)
    
                with rio.open(mask_filepath) as mask_ds:
                    # Read pixels
                    mask_arr = mask_ds.read(1)
    
                # Make the pixels at the borders of the mask black so they are 
                # ignored in the comparison
                if border_pixels_to_ignore and border_pixels_to_ignore > 0:
                    mask_arr[0:border_pixels_to_ignore,:] = 0    # Left border
                    mask_arr[-border_pixels_to_ignore:,:] = 0    # Right border
                    mask_arr[:,0:border_pixels_to_ignore] = 0    # Top border
                    mask_arr[:,-border_pixels_to_ignore:] = 0    # Bottom border
                    
                #similarity = jaccard_similarity(mask_arr, image_pred)
                # Use accuracy as similarity... is more practical than jaccard
                similarity = np.equal(mask_arr, image_pred_uint8_cleaned_bin
                                     ).sum()/image_pred_uint8_cleaned_bin.size
                pred_prefix_str = f"{similarity:0.3f}_"
                
                # Copy mask file if the file doesn't exist yet
                mask_copy_dest_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_mask.tif"
                if not os.path.exists(mask_copy_dest_filepath):
                    shutil.copyfile(mask_filepath, mask_copy_dest_filepath)
    
            else:
                # If all_black, no need to calculate again
                if all_black: 
                    pct_black = 1
                else:
                    # Calculate percentage black pixels
                    pct_black = 1 - ((image_pred_uint8_bin.sum()/255)
                                     /image_pred_uint8_bin.size)
                
                # If the result after segmentation is all black, set all_black
                if pct_black == 1:
                    # Force the prefix to be really high so it is clear they are entirely black
                    pred_prefix_str = "1.001_"
                    all_black = True
                else:
                    pred_prefix_str = f"{pct_black:0.3f}_"
    
                # If there are few white pixels, don't save it,
                # because we are in evaluetion mode anyway...
                #if similarity >= 0.95:
                    #continue
          
            # Copy the input image if it doesn't exist yet in output path
            image_copy_dest_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}{image_ext}"
            if not os.path.exists(image_copy_dest_filepath):
                shutil.copyfile(image_filepath, image_copy_dest_filepath)

            # If all_black, we are ready now
            if all_black:
                logger.debug("All black prediction, no use proceding")
                return 
        
        # Get some info about images
        image_width = image_pred_arr.shape[0]
        image_height = image_pred_arr.shape[1]

        # Now write +- original prediction (as uint8) to file
        logger.debug("Save semi-detailed (uint8, 10 different values) prediction")
        
        image_pred_orig_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_pred.tif"
        with rio.open(image_pred_orig_filepath, 'w', driver='GTiff', 
                      compress='lzw', predictor=2, num_threads=4, tiled='no',
                      height=image_height, width=image_width, 
                      count=1, dtype=rio.uint8, crs=image_crs, transform=image_transform) as dst:
            dst.write(image_pred_uint8_base10, 1)
'''
def postprocess_prediction(
        image_filepath: str,
        image_crs: str,
        image_transform,
        output_dir: str,
        image_pred_arr = None,
        image_pred_filepath = None,
        input_mask_dir: str = None,
        border_pixels_to_ignore: int = 0,
        evaluate_mode: bool = False,
        force: bool = False):
    """
    TODO
    
    Args
        
    """
    
    logger.debug(f"Start postprocess for {image_pred_filepath}")
    
    # Either the filepath to a temp file with the prediction or the data 
    # itself need to be provided
    if(image_pred_arr is None
       and image_pred_filepath is None):
        message = f"postprocess_prediction needs either image_pred_arr or image_pred_filepath, now both are None"
        logger.error(message)
        raise Exception(message)

    # If no decent transform metadata, write warning
    if image_transform is None or image_transform[0] == 0:
        # If in evaluate mode... warning is enough, otherwise error!
        message = f"No transform found for {image_filepath}: {image_transform}"        
        if evaluate_mode:
            logger.warn(message)
        else:
            logger.error(message)
            raise Exception(message)          

    try:      

        # Prepare the filepath for the output
        input_image_dir, image_filename = os.path.split(image_filepath)
        image_filename_noext, image_ext = os.path.splitext(image_filename)
        
        if image_pred_arr is None:
            image_pred_arr = np.load(image_pred_filepath)
            os.remove(image_pred_filepath)
        
        # Check the number of channels of the output prediction
        # TODO: maybe param name should be clearer !!!
        n_channels = image_pred_arr.shape[2]
        if n_channels > 1:
            raise Exception(f"Not implemented: processing prediction output with multiple channels: {n_channels}")
        
        # Input should be in float32
        if image_pred_arr.dtype != np.float32:
            raise Exception(f"Image prediction should be in numpy.float32, but is in: {image_pred_arr.dtype}")

        # Reshape array from 4 dims (image_nb, width, height, nb_channels) to 2.
        image_pred_arr = image_pred_arr.reshape((image_pred_arr.shape[0], image_pred_arr.shape[1]))
                
        # Make the pixels at the borders of the prediction black so they are ignored
        if border_pixels_to_ignore and border_pixels_to_ignore > 0:
            image_pred_arr[0:border_pixels_to_ignore,:] = 0    # Left border
            image_pred_arr[-border_pixels_to_ignore:,:] = 0    # Right border
            image_pred_arr[:,0:border_pixels_to_ignore] = 0    # Top border
            image_pred_arr[:,-border_pixels_to_ignore:] = 0    # Bottom border
    
        # Check if the result is entirely black... if so no cleanup needed
        all_black = False
        thresshold_ok_float32 = 0.5
        thresshold_ok_uint8 = 128
        image_pred_uint8_cleaned_bin = None
        if not np.any(image_pred_arr >= thresshold_ok_float32):
            logger.debug('Prediction is entirely black!')
            all_black = True
        '''
        else:
            # Cleanup the image so it becomes a clean 2 color one instead of grayscale
            logger.debug("Clean prediction")
            image_pred_uint8_cleaned_bin = region_segmentation(
                    image_pred_arr, thresshold_ok=thresshold_ok_float32)
            image_pred_uint8_cleaned_bin = image_pred_uint8_cleaned_bin * 255
            if not np.any(image_pred_uint8_cleaned_bin > thresshold_ok_uint8):
                logger.info('Prediction became entirely black!')
                all_black = True
        '''
        
        # If not in evaluate mode and the prediction is all black, return
        if not evaluate_mode and all_black:
            logger.debug("All black prediction, no use saving this")
            return 

        # Convert to uint8 + create binary version
        image_pred_uint8 = (image_pred_arr * 255).astype(np.uint8)
        image_pred_uint8_bin = to_binary_uint8(image_pred_uint8, 
                                               thresshold_ok_uint8)
        image_pred_uint8_base10 = (image_pred_arr * 10).astype(np.uint8)
        image_pred_uint8_base10 = image_pred_uint8_base10 * 25

        if image_pred_uint8_cleaned_bin is None:
            image_pred_uint8_cleaned_bin = image_pred_uint8_bin

        # Make sure the output dir exists...
        os.makedirs(output_dir, exist_ok=True)
                
        # If in evaluate mode, put a prefix in the file name
        pred_prefix_str = ''
        if evaluate_mode:
            
            def jaccard_similarity(im1, im2):
                if im1.shape != im2.shape:
                    message = f"Shape mismatch: input have different shape: im1: {im1.shape}, im2: {im2.shape}"
                    logger.critical(message)
                    raise ValueError(message)
    
                intersection = np.logical_and(im1, im2)
                union = np.logical_or(im1, im2)
    
                sum_union = float(union.sum())
                if sum_union == 0.0:
                    # If 0 positive pixels in union: perfect prediction, so 1
                    return 1
                else:
                    sum_intersect = intersection.sum()
                    return sum_intersect/sum_union
    
            # If there is a mask dir specified... use the groundtruth mask
            if input_mask_dir and os.path.exists(input_mask_dir):
                # Read mask file and get all needed info from it...
                mask_filepath = image_filepath.replace(input_image_dir,
                                                       input_mask_dir)
                # Check if this file exists, if not, look for similar files
                if not os.path.exists(mask_filepath):
                    mask_filepath_noext = os.path.splitext(mask_filepath)[0]
                    files = glob.glob(mask_filepath_noext + '*')
                    if len(files) == 1:
                        mask_filepath = files[0]
                    else:
                        message = f"Error finding mask file with {mask_filepath_noext + '*'}: {len(files)} mask(s) found"
                        logger.error(message)
                        raise Exception(message)
    
                with rio.open(mask_filepath) as mask_ds:
                    # Read pixels
                    mask_arr = mask_ds.read(1)
    
                # Make the pixels at the borders of the mask black so they are 
                # ignored in the comparison
                if border_pixels_to_ignore and border_pixels_to_ignore > 0:
                    mask_arr[0:border_pixels_to_ignore,:] = 0    # Left border
                    mask_arr[-border_pixels_to_ignore:,:] = 0    # Right border
                    mask_arr[:,0:border_pixels_to_ignore] = 0    # Top border
                    mask_arr[:,-border_pixels_to_ignore:] = 0    # Bottom border
                    
                #similarity = jaccard_similarity(mask_arr, image_pred)
                # Use accuracy as similarity... is more practical than jaccard
                similarity = np.equal(mask_arr, image_pred_uint8_cleaned_bin
                                     ).sum()/image_pred_uint8_cleaned_bin.size
                pred_prefix_str = f"{similarity:0.3f}_"
                
                # Copy mask file if the file doesn't exist yet
                mask_copy_dest_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_mask.tif"
                if not os.path.exists(mask_copy_dest_filepath):
                    shutil.copyfile(mask_filepath, mask_copy_dest_filepath)
    
            else:
                # If all_black, no need to calculate again
                if all_black: 
                    pct_black = 1
                else:
                    # Calculate percentage black pixels
                    pct_black = 1 - ((image_pred_uint8_bin.sum()/255)
                                     /image_pred_uint8_bin.size)
                
                # If the result after segmentation is all black, set all_black
                if pct_black == 1:
                    # Force the prefix to be really high so it is clear they are entirely black
                    pred_prefix_str = "1.001_"
                    all_black = True
                else:
                    pred_prefix_str = f"{pct_black:0.3f}_"
    
                # If there are few white pixels, don't save it,
                # because we are in evaluetion mode anyway...
                #if similarity >= 0.95:
                    #continue
          
            # Copy the input image if it doesn't exist yet in output path
            image_copy_dest_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}{image_ext}"
            if not os.path.exists(image_copy_dest_filepath):
                shutil.copyfile(image_filepath, image_copy_dest_filepath)

            # If all_black, we are ready now
            if all_black:
                logger.debug("All black prediction, no use proceding")
                return 
        
        # Get some info about images
        image_width = image_pred_arr.shape[0]
        image_height = image_pred_arr.shape[1]

        # Now write +- original prediction (as uint8) to file
        logger.debug("Save semi-detailed (uint8, 10 different values) prediction")
        
        image_pred_orig_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_pred.tif"
        with rio.open(image_pred_orig_filepath, 'w', driver='GTiff', 
                      compress='lzw', predictor=2, num_threads=4, tiled='no',
                      height=image_height, width=image_width, 
                      count=1, dtype=rio.uint8, crs=image_crs, transform=image_transform) as dst:
            dst.write(image_pred_uint8_base10, 1)
    
        # TODO:temp hack to test performance without polygonize...
        #return

        # Polygonize result
        # Returns a list of tupples with (geometry, value)
        polygonized_records = list(rio_features.shapes(
                image_pred_uint8_bin, mask=image_pred_uint8_bin, transform=image_transform))
        
        # If nothing found, we can return
        if len(polygonized_records) == 0:
            logger.warning(f"Despite that all-black images are normally not written, this prediction didn't result in any polygons: {image_pred_orig_filepath}")
            return 

        # Convert shapes to shapely geoms 
        geoms = []
        for geom, _ in polygonized_records:
            geoms.append(sh.geometry.shape(geom))   
        geoms_gdf = gpd.GeoDataFrame(geoms, columns=['geometry'])
        geoms_gdf.crs = image_crs
        
        # If not in evaluate mode, write the geom to wkt file
        if evaluate_mode is False:
            logger.debug('Before writing orig geom file')
            image_bounds = rio.transform.array_bounds(
                    image_height, image_width, image_transform)
            x_pixsize = get_pixelsize_x(image_transform)
            y_pixsize = get_pixelsize_y(image_transform)
            border_bounds = (image_bounds[0]+border_pixels_to_ignore*x_pixsize,
                             image_bounds[1]+border_pixels_to_ignore*y_pixsize,
                             image_bounds[2]-border_pixels_to_ignore*x_pixsize,
                             image_bounds[3]-border_pixels_to_ignore*y_pixsize)
            geoms_gdf = vh.calc_onborder(geoms_gdf, border_bounds)
            geom_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_pred_cleaned.geojson"
            geofile_util.to_file(geoms_gdf, geom_filepath)
            
        else:
            # For easier evaluation, write the cleaned version as raster
            # Write the standard cleaned output to file
            logger.debug("Save binary prediction")
            image_pred_cleaned_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_pred_bin.tif"
            with rio.open(image_pred_cleaned_filepath, 'w', driver='GTiff', 
                          compress='lzw',
                          height=image_height, width=image_width, 
                          count=1, dtype=rio.uint8, 
                          crs=image_crs, transform=image_transform) as dst:
                dst.write(image_pred_uint8_bin, 1)
            
            """
            # If different from normal bin, also write cleaned bin
            nb_not_equal_pix = np.not_equal(image_pred_uint8_bin, 
                                            image_pred_uint8_cleaned_bin
                                           ).sum()
            if nb_not_equal_pix > 0:
                logger.info("Cleaned binary different from normal, save")
                image_pred_orig_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_pred_cleaned_bin.tif"
                with rio.open(image_pred_orig_filepath, 'w', driver='GTiff', 
                              compress='lzw',
                              height=image_height, width=image_width, 
                              count=1, dtype=rio.uint8, 
                              crs=image_crs, transform=image_transform) as dst:
                    dst.write(image_pred_uint8_cleaned_bin, 1)
            """
            
            # If the input image contained a tranform, also create an image 
            # based on the simplified vectors
            if(image_transform[0] != 0 
               and len(geoms) > 0):
                # Simplify geoms
                geoms_simpl = []
                geoms_simpl_vis = []
                for geom in geoms:
                    # The simplify of shapely uses the deuter-pecker algo
                    # preserve_topology is slower bu makes sure no polygons are removed
                    geom_simpl = geom.simplify(0.5, preserve_topology=True)
                    if not geom_simpl.is_empty:
                        geoms_simpl.append(geom_simpl)
                    
                    '''
                    # Also do the simplify with Visvalingam algo
                    # -> seems to give better results for this application
                    # Throws away too much info!!!
                    geom_simpl_vis = vh.simplify_visval(geom, 5)
                    if geom_simpl_vis is not None:
                        geoms_simpl_vis.append(geom_simpl_vis)
                    '''
                
                # Write simplified wkt result to raster for comparing. 
                if len(geoms_simpl) > 0:
                    # TODO: doesn't support multiple classes
                    logger.debug('Before writing simpl rasterized file')
                    image_pred_simpl_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_pred_cleaned_simpl.tif"
                    with rio.open(image_pred_simpl_filepath, 'w', driver='GTiff', compress='lzw',
                                  height=image_height, width=image_width, 
                                  count=1, dtype=rio.uint8, crs=image_crs, transform=image_transform) as dst:
                        # this is where we create a generator of geom, value pairs to use in rasterizing
                        logger.debug('Before rasterize')
                        burned = rio_features.rasterize(
                                shapes=geoms_simpl, 
                                out_shape=(image_height, image_width),
                                fill=0, default_value=255, dtype=rio.uint8,
                                transform=image_transform)
                        dst.write(burned, 1)
                
                # Write simplified wkt result to raster for comparing. Use the same
                if len(geoms_simpl_vis) > 0:
                    # file profile as created before for writing the raw prediction result
                    # TODO: doesn't support multiple classes
                    logger.debug('Before writing simpl with visvangali algo rasterized file')
                    image_pred_simpl_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_pred_cleaned_simpl_vis.tif"
                    with rio.open(image_pred_simpl_filepath, 'w', driver='GTiff', compress='lzw',
                                  height=image_height, width=image_width, 
                                  count=1, dtype=rio.uint8, crs=image_crs, transform=image_transform) as dst:
                        # this is where we create a generator of geom, value pairs to use in rasterizing
                        logger.debug('Before rasterize')
                        burned = rio_features.rasterize(
                                shapes=geoms_simpl_vis, 
                                out_shape=(image_height, image_width),
                                fill=0, default_value=255, dtype=rio.uint8,
                                transform=image_transform)
                        dst.write(burned, 1)
    
    except:
        logger.error(f"Exception postprocessing prediction for {image_filepath}\n: file {image_pred_filepath}!!!")
        raise
        
#-------------------------------------------------------------
# Helpers for working with Affine objects...                    
#-------------------------------------------------------------

def get_pixelsize_x(transform):
    return transform[0]
    
def get_pixelsize_y(transform):
    return -transform[4]

#-------------------------------------------------------------
# Postprocess to use on all vector outputs
#-------------------------------------------------------------

def postprocess_vectors(input_dir: str,
                        output_filepath: str,
                        evaluate_mode: bool = False,
                        force: bool = False):
    """
    Merges all geojson files in input dir (recursively), unions them, and 
    does general cleanup on them.
    
    Outputs actually several files. The output files will have a suffix 
    to the general output_filepath provided depending on the type of the 
    output file.
    
    Args
        input_dir: the dir where all geojson files can be found. All geojson 
                files will be searched for recursively.
        output_filepath: the filepath where the output file(s) will be written.
        force: False to just keep existing output files instead of processing
                them again. 
        evaluate_mode: True to apply the logic to a subset of the files.            
    """
    
    # TODO: this function should be split to several ones, because now it does
    # several things that aren't covered with the name
    
    eval_suffix = ""
    if evaluate_mode:
        eval_suffix = "_eval"
    
    # Prepare output dir
    output_dir, output_filename = os.path.split(output_filepath)
    output_dir = f"{output_dir}{eval_suffix}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare output driver
    output_basefilename_noext, output_ext = os.path.splitext(output_filename)
    output_ext = output_ext.lower()
    
    # Read and merge input files
    geoms_orig_filepath = os.path.join(
            output_dir, f"{output_basefilename_noext}_orig{output_ext}")    
    try:
        vh.merge_vector_files(input_dir=input_dir,
                              output_filepath=geoms_orig_filepath,
                              evaluate_mode=evaluate_mode,
                              force=force)               
        geoms_gdf = geofile_util.read_file(geoms_orig_filepath)
    except RuntimeWarning as ex:
        logger.warn(f"No vector files found to merge, ex: {ex}")
        if str(ex) == "NOFILESFOUND":
            logger.warn("No vector files found to merge")
            return

    # Union the data, optimized using the available onborder column
    geoms_union_filepath = os.path.join(
            output_dir, f"{output_basefilename_noext}_union{output_ext}")
    geoms_union_gdf = vh.unary_union_with_onborder(input_gdf=geoms_gdf,
                              input_filepath=geoms_orig_filepath,
                              output_filepath=geoms_union_filepath,
                              evaluate_mode=evaluate_mode,
                              force=force)

    # Retain only geoms > 5m²
    geoms_gt5m2_filepath = os.path.join(
            output_dir, f"{output_basefilename_noext}_union_gt5m2{output_ext}")
    geoms_gt5m2_gdf = None
    if force or not os.path.exists(geoms_gt5m2_filepath):
        if geoms_union_gdf is None:
            geoms_union_gdf = geofile_util.read_file(geoms_union_filepath)
        geoms_gt5m2_gdf = geoms_union_gdf.loc[(geoms_union_gdf.geometry.area > 5)]
        # TODO: setting the CRS here hardcoded should be removed
        if geoms_gt5m2_gdf.crs is None:
            message = "No crs available!!!"
            logger.error(message)
            #raise Exception(message)
        
        # Qgis wants unique id column, otherwise weird effects!
        geoms_gt5m2_gdf.reset_index(inplace=True, drop=True)
        #geoms_gt5m2_gdf['id'] = geoms_gt5m2_gdf.index 
        geoms_gt5m2_gdf.loc[:, 'id'] = geoms_gt5m2_gdf.index 
        geofile_util.to_file(geoms_gt5m2_gdf, geoms_gt5m2_filepath)

    # Simplify with standard shapely algo 
    # -> if preserve_topology False, this is Ramer-Douglas-Peucker, otherwise ?
    geoms_simpl_shap_filepath = os.path.join(
            output_dir, f"{output_basefilename_noext}_simpl_shap{output_ext}")
    geoms_simpl_shap_gdf = None
    if force or not os.path.exists(geoms_simpl_shap_filepath):
        logger.info("Simplify with default shapely algo")
        # If input geoms not yet in memory, read from file
        if geoms_gt5m2_gdf is None:
            geoms_gt5m2_gdf = geofile_util.read_file(geoms_gt5m2_filepath)

        # Simplify, fix invalid geoms, remove empty geoms, 
        # apply multipart-to-singlepart, only > 5m² + write
        geoms_simpl_shap_gdf = geoms_gt5m2_gdf.copy()
        geoms_simpl_shap_gdf['geometry'] = geoms_simpl_shap_gdf.simplify(
                tolerance=0.5, preserve_topology=True)
        geoms_simpl_shap_gdf['geometry'] = geoms_simpl_shap_gdf.geometry.apply(
                lambda geom: vh.fix(geom))
        geoms_simpl_shap_gdf.dropna(subset=['geometry'], inplace=True)
        geoms_simpl_shap_gdf = geoms_simpl_shap_gdf.reset_index(drop=True).explode()
        geoms_simpl_shap_gdf['geometry'] = geoms_simpl_shap_gdf.geometry.apply(
                lambda geom: vh.remove_inner_rings(geom, 2))        
                
        # Add area column, and remove rows with small area
        geoms_simpl_shap_gdf['area'] = geoms_simpl_shap_gdf.geometry.area       
        geoms_simpl_shap_gdf = geoms_simpl_shap_gdf.loc[
                (geoms_simpl_shap_gdf['area'] > 5)]

        # Qgis wants unique id column, otherwise weird effects!
        geoms_simpl_shap_gdf.reset_index(inplace=True, drop=True)
        geoms_simpl_shap_gdf['id'] = geoms_simpl_shap_gdf.index 
        geoms_simpl_shap_gdf['nbcoords'] = geoms_simpl_shap_gdf.geometry.apply(
                lambda geom: vh.get_nb_coords(geom))
        geofile_util.to_file(geoms_simpl_shap_gdf, geoms_simpl_shap_filepath)
        logger.info(f"Result written to {geoms_simpl_shap_filepath}")
        
    # Apply begative buffer on result
    geoms_simpl_shap_m1m_filepath = os.path.join(
            output_dir, f"{output_basefilename_noext}_simpl_shap_m1.5m_3{output_ext}")    
    if force or not os.path.exists(geoms_simpl_shap_m1m_filepath):
        logger.info("Apply negative buffer")
        # If input geoms not yet in memory, read from file
        if geoms_simpl_shap_gdf is None:
            geoms_simpl_shap_gdf = geofile_util.read_file(geoms_simpl_shap_filepath)
            
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
                lambda geom: vh.get_nb_coords(geom))        
        geofile_util.to_file(geoms_simpl_shap_m1m_gdf, geoms_simpl_shap_m1m_filepath)
        logger.info(f"Result written to {geoms_simpl_shap_m1m_filepath}")
        
    '''
    # Also do the simplify with Visvalingam algo
    geoms_simpl_vis_filepath = os.path.join(
            union_dir, f"geoms_simpl_vis.geojson")
    if force or not os.path.exists(geoms_simpl_vis_filepath):
        # If input geoms not yet in memory, read from file
        if geoms_gt5m2_gdf is None:
            geoms_gt5m2_gdf = geofile_util.read_file(geoms_gt5m2_filepath)

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
        geofile_util.to_file(geoms_simpl_vis_gdf, geoms_simpl_vis_filepath)
        logger.info(f"Result written to {geoms_simpl_vis_filepath}")
    
    # Also do the simplify with Ramer-Douglas-Peucker "plus" algo:
    # the plus is that there is some extra checks
    geoms_simpl_rdpp_filepath = os.path.join(
            union_dir, f"geoms_simpl_rdpp.geojson")
    if force or not os.path.exists(geoms_simpl_rdpp_filepath):
        # If input geoms not yet in memory, read from file
        if geoms_gt5m2_gdf is None:
            geoms_gt5m2_gdf = geofile_util.read_file(geoms_gt5m2_filepath)

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
        geofile_util.to_file(geoms_simpl_rdpp_gdf, geoms_simpl_rdpp_filepath)
        logger.info(f"Result written to {geoms_simpl_rdpp_filepath}")
    '''
    
#-------------------------------------------------------------
# If the script is ran directly...
#-------------------------------------------------------------

if __name__ == '__main__':
    message = "Not implemented"
    logger.error(message)
    raise Exception(message)

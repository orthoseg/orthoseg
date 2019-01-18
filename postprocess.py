# -*- coding: utf-8 -*-
"""
Module with functions for post-processing prediction masks towards polygons.

@author: Pieter Roggemans
"""

import logging
import os
import glob
import shutil

import numpy as np
import skimage
import skimage.morphology       # Needs to be imported explicitly as it is a submodule
from scipy import ndimage
import rasterio as rio
import rasterio.features as rio_features
import shapely as sh
import geopandas as gpd

import vector.vector_helper as vh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

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

def to_binary_uint8(in_arr, thresshold_ok):
    if in_arr.dtype != np.uint8:
        raise Exception("Input should be dtype = uint8, not: {in_arr.dtype}")
        
    # First copy to new numpy array, otherwise input array is changed
    out_arr = np.copy(in_arr)
    out_arr[out_arr >= thresshold_ok] = 255
    out_arr[out_arr < thresshold_ok] = 0
    
    return out_arr

def postprocess_prediction(image_filepath: str,
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
        if image_pred_uint8_cleaned_bin is None:
            image_pred_uint8_cleaned_bin = image_pred_uint8_bin

        # Make sure the output dir exists...
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
                
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

        # Now write original prediction (as uint8) to file
        logger.debug("Save detailed (uint8) prediction")
        image_pred_orig_filepath = f"{output_dir}{os.sep}{pred_prefix_str}{image_filename_noext}_pred.tif"
        with rio.open(image_pred_orig_filepath, 'w', driver='GTiff', compress='lzw',
                      height=image_height, width=image_width, 
                      count=1, dtype=rio.uint8, crs=image_crs, transform=image_transform) as dst:
            dst.write(image_pred_uint8, 1)
            
        # Polygonize result
        # Returns a list of tupples with (geometry, value)
        shapes = rio_features.shapes(image_pred_uint8_bin,
                                     mask=image_pred_uint8_bin,
                                     transform=image_transform)
    
        # Convert shapes to shapely geoms 
        geoms = []
        for shape in list(shapes):
            geom, value = shape
            geom_sh = sh.geometry.shape(geom)
            geoms.append(geom_sh)   
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
            geoms_gdf.to_file(geom_filepath, driver="GeoJSON")
            
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
                    
                    # Also do the simplify with Visvalingam algo
                    # -> seems to give better results for this application
                    geom_simpl_vis = vh.simplify_visval(geom, 5)
                    if geom_simpl_vis is not None:
                        geoms_simpl_vis.append(geom_simpl_vis)
                
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
# If the script is ran directly...
#-------------------------------------------------------------
    
if __name__ == '__main__':
    message = "Not implemented"
    logger.error(message)
    raise Exception(message)
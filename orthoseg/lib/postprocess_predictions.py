# -*- coding: utf-8 -*-
"""
Module with functions for post-processing prediction masks towards polygons.
"""

from concurrent import futures
import logging
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Optional

# Evade having many info warnings about self intersections from shapely
logging.getLogger('shapely.geos').setLevel(logging.WARNING)
import fiona
import geofile_ops.geofile_ops as geofile_ops
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features as rio_features
import shapely as sh

from orthoseg.util import geofile_util
from orthoseg.util import vector_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

#-------------------------------------------------------------
# Postprocess to use on all vector outputs
#-------------------------------------------------------------

def postprocess_predictions(
        input_dir: Path,
        output_filepath: Path,
        input_ext: str,
        border_pixels_to_ignore: int = 0,
        evaluate_mode: bool = False,
        cancel_filepath: Optional[Path] = None,
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
        cancel_filepath: If the file in this path exists, processing stops asap
        force: False to skip results that already exist, true to
               ignore existing results and overwrite them           
    """
    
    # TODO: this function should be split to several ones, because now it does
    # several things that aren't covered with the name
    ##### Init #####
    # Prepare output dir
    eval_suffix = ""
    output_dir = output_filepath.parent
    if evaluate_mode:
        eval_suffix = "_eval"
        output_dir = output_dir.parent / (output_dir.name + eval_suffix)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Prepare output driver
    output_basefilename_noext = output_filepath.stem
    output_ext = output_filepath.suffix.lower()
    
    # Polygonize all prediction files if orig file doesn't exist yet
    geoms_orig_filepath = output_dir / f"{output_basefilename_noext}{output_ext}"
    if not geoms_orig_filepath.exists():
        try:
            polygonize_prediction_files(
                    input_dir=input_dir,
                    output_filepath=geoms_orig_filepath,
                    input_ext=input_ext,
                    border_pixels_to_ignore=border_pixels_to_ignore,
                    evaluate_mode=evaluate_mode,
                    cancel_filepath=cancel_filepath,
                    force=force)               
        except RuntimeWarning as ex:
            logger.warn(f"No prediction files found to merge, ex: {ex}")
            if str(ex) == "NOFILESFOUND":
                logger.warn("No prediction files found to merge")
                return
    else:
        logger.info(f"Output file exists already, so continue postprocess: {geoms_orig_filepath}")

    # If the cancel file exists, stop processing...
    if cancel_filepath is not None and cancel_filepath.exists():
        logger.info(f"Cancel file found, so stop: {cancel_filepath}")
        return

    # Union the data
    input_cardsheets_path = r"X:\GIS\GIS DATA\Versnijdingen\Kaartbladversnijdingen_NGI_numerieke_reeks_Shapefile\Shapefile\Kbl8.shp"
    geoms_union_filepath = output_dir / f"{output_basefilename_noext}_union{output_ext}"
    geofile_ops.dissolve_cardsheets(
            input_path=geoms_orig_filepath,
            input_cardsheets_path=input_cardsheets_path,
            output_path=geoms_union_filepath,
            explodecollections=True)

    # Retain only geoms > 5m²
    geoms_gt5m2_filepath = output_dir / f"{output_basefilename_noext}_union_gt5m2{output_ext}"
    geoms_gt5m2_gdf = None
    if force or not geoms_gt5m2_filepath.exists():
        geoms_union_gdf = geofile_util.read_file(geoms_union_filepath)
        geoms_union_gdf.reset_index(inplace=True)
        geoms_gt5m2_gdf = geoms_union_gdf.loc[(geoms_union_gdf.geometry.area > 5)].copy()
        # TODO: setting the CRS here hardcoded should be removed
        if geoms_gt5m2_gdf.crs is None:
            message = "No crs available!!!"
            logger.error(message)
            #raise Exception(message)
        
        # Qgis wants unique id column, otherwise weird effects!
        geoms_gt5m2_gdf.reset_index(inplace=True, drop=True)
        geoms_gt5m2_gdf.loc[:, 'id'] = geoms_gt5m2_gdf.index 
        geofile_util.to_file(geoms_gt5m2_gdf, geoms_gt5m2_filepath)

    # Simplify with standard shapely algo 
    # -> if preserve_topology False, this is Ramer-Douglas-Peucker, otherwise ?
    geoms_simpl_filepath = output_dir / f"{output_basefilename_noext}_simpl{output_ext}"
    geoms_simpl_gdf = None
    if force or not geoms_simpl_filepath.exists():
        logger.info("Simplify with default shapely algo")
        # If input geoms not yet in memory, read from file
        if geoms_gt5m2_gdf is None:
            geoms_gt5m2_gdf = geofile_util.read_file(geoms_gt5m2_filepath)

        # Simplify, fix invalid geoms, remove empty geoms, 
        # apply multipart-to-singlepart, only > 5m² + write
        geoms_simpl_gdf = geoms_gt5m2_gdf.copy()
        geoms_simpl_gdf['geometry'] = geoms_simpl_gdf.simplify(
                tolerance=0.5, preserve_topology=True)
        geoms_simpl_gdf['geometry'] = geoms_simpl_gdf.geometry.apply(
                lambda geom: vector_util.fix(geom))
        geoms_simpl_gdf.dropna(subset=['geometry'], inplace=True)

        # Explode only works with one column, but cast to geodataframe, otherwise geoseries 
        geoms_simpl_gdf = gpd.GeoDataFrame(geoms_simpl_gdf['geometry'])
        geoms_simpl_gdf = geoms_simpl_gdf.explode().reset_index()
        geoms_simpl_gdf['geometry'] = geoms_simpl_gdf.geometry.apply(
                lambda geom: vector_util.remove_inner_rings(geom, 2))        
                
        # Add area column, and remove rows with small area
        geoms_simpl_gdf['area'] = geoms_simpl_gdf.geometry.area       
        geoms_simpl_gdf = geoms_simpl_gdf.loc[
                (geoms_simpl_gdf['area'] > 5)]

        # Qgis wants unique id column, otherwise weird effects!
        geoms_simpl_gdf.reset_index(inplace=True, drop=True)
        geoms_simpl_gdf['id'] = geoms_simpl_gdf.index 
        geoms_simpl_gdf['nbcoords'] = geoms_simpl_gdf.geometry.apply(
                lambda geom: vector_util.get_nb_coords(geom))
        geofile_util.to_file(geoms_simpl_gdf, geoms_simpl_filepath)
        logger.info(f"Result written to {geoms_simpl_filepath}")
        
    # Apply negative buffer on result
    geoms_simpl_m1m_filepath = (
            output_dir / f"{output_basefilename_noext}_simpl_m1.5m{output_ext}")
    if force or not geoms_simpl_m1m_filepath.exists():
        logger.info("Apply negative buffer")
        # If input geoms not yet in memory, read from file
        if geoms_simpl_gdf is None:
            geoms_simpl_gdf = geofile_util.read_file(geoms_simpl_filepath)
            
        # Simplify, fix invalid geoms, remove empty geoms, 
        # apply multipart-to-singlepart, only > 5m² + write
        geoms_simpl_m1m_gdf = geoms_simpl_gdf.copy()
        geoms_simpl_m1m_gdf['geometry'] = geoms_simpl_m1m_gdf.buffer(
                distance=-1.5, resolution=3)       
        geoms_simpl_m1m_gdf.dropna(subset=['geometry'], inplace=True)

        # Explode only possible if one column, but should stay a geodataframe to add columns later
        geoms_simpl_m1m_gdf = gpd.GeoDataFrame(geoms_simpl_m1m_gdf['geometry'])
        geoms_simpl_m1m_gdf = geoms_simpl_m1m_gdf.explode()
        geoms_simpl_m1m_gdf.reset_index(inplace=True, drop=True)
                
        # Add/calculate area column, and remove rows with small area
        geoms_simpl_m1m_gdf['area'] = geoms_simpl_m1m_gdf.geometry.area       
        geoms_simpl_m1m_gdf = geoms_simpl_m1m_gdf.loc[
                geoms_simpl_m1m_gdf['area'] > 5]

        # Qgis wants unique id column, otherwise weird effects!
        geoms_simpl_m1m_gdf.reset_index(inplace=True, drop=True)
        geoms_simpl_m1m_gdf['id'] = geoms_simpl_m1m_gdf.index 
        geoms_simpl_m1m_gdf['nbcoords'] = geoms_simpl_m1m_gdf.geometry.apply(
                lambda geom: vector_util.get_nb_coords(geom))        
        geofile_util.to_file(geoms_simpl_m1m_gdf, geoms_simpl_m1m_filepath)
        logger.info(f"Result written to {geoms_simpl_m1m_filepath}")

def polygonize_prediction_files(
            input_dir: Path,
            output_filepath: Path,
            input_ext: str,
            border_pixels_to_ignore: int = 0,
            apply_on_border_distance: float = None,
            evaluate_mode: bool = False,
            cancel_filepath: Optional[Path] = None,
            force: bool = False):
    """
    Polygonizes all prediction files in input dir (recursively) and writes it to ones file.
    
    Returns the resulting GeoDataFrame or None if the output file already 
    exists.
    
    Args
        input_dir:
        output_filepath:
        apply_on_border_distance:
        evaluate_mode: True to run in evaluate mode
        cancel_filepath: If the file in this path exists, processing stops asap
        force: False to skip images that already have a prediction, true to
               ignore existing predictions and overwrite them
    """
    ##### Init #####
    # Check if we need to do anything anyway
    if not force and output_filepath.exists():
        logger.info(f"Force is false and output file exists already, skip: {output_filepath}")
        return None

    # Check if we are in interactive mode, because otherwise the ProcessExecutor 
    # hangs
    if sys.__stdin__.isatty():
        logger.warn(f"Running in interactive mode???")
        #raise Exception("You cannot run this in interactive mode, because it doens't support multiprocessing.")

    # Make sure the output dir exists
    output_dir = output_filepath.parent
    if not output_dir.exists():
        output_dir.mkdir()

    # Get list of all files to process...
    logger.info(f"List all files to be merged in {input_dir}")
    filepaths = sorted(list(input_dir.rglob(f"*_pred*{input_ext}")))

    # Check if files were found...
    nb_files = len(filepaths)
    if nb_files == 0:
        logger.warn("No files found to process... so return")
        raise RuntimeWarning("NOFILESFOUND")
    logger.info(f"Found {nb_files} files to process")

    # First write to tmp output file so it is clear if the file was ready or not
    layer = output_filepath.stem
    output_tmp_filepath = output_filepath.parent / f"{output_filepath.stem}_BUSY{output_filepath.suffix}"
    output_tmp_file = None
    if output_tmp_filepath.exists():
        output_tmp_filepath.unlink()

    # Loop through all files to be processed...
    try:       
        max_parallel = multiprocessing.cpu_count()
        with futures.ProcessPoolExecutor(max_parallel) as read_pool:

            future_to_filepath_queue = {}
            geoms_gdf = None
            nb_files_done = 0
            nb_files_in_queue = 0
            next_file_to_queue = 0
            max_files_in_queue = max_parallel * 2

            # Loop till all files are done
            while nb_files_done <= (nb_files-1):

                # If the cancel file exists, stop processing...
                if cancel_filepath is not None and cancel_filepath.exists():
                    logger.info(f"Cancel file found, so stop: {cancel_filepath}")
                    return

                # If the queue isn't complely filled or all files are being treated, 
                # add files to processing queue
                while(nb_files_in_queue < max_files_in_queue
                      and next_file_to_queue < nb_files):
                    
                    # Read prediction file
                    future = read_pool.submit(
                            read_prediction_file, 
                            filepaths[next_file_to_queue],
                            border_pixels_to_ignore)
                    future_to_filepath_queue[future] = filepaths[next_file_to_queue]
                    nb_files_in_queue += 1
                    next_file_to_queue += 1

                # Get the futures that are ready from the queue, and process the result
                futures_done = futures.wait(future_to_filepath_queue, return_when='FIRST_COMPLETED').done
                for future in futures_done:
                    # Get result
                    nb_files_done += 1

                    #logger.info(f"Ready processing {future_to_filepath[future]}")
                    try:
                        geoms_file_gdf = future.result()

                        if geoms_file_gdf is None:
                            continue
                        if geoms_gdf is None:
                            # Check if the input has a crs
                            if geoms_file_gdf.crs is None:
                                raise Exception("STOP: input does not have a crs!") 
                            geoms_gdf = geoms_file_gdf
                        else:
                            geoms_gdf = gpd.GeoDataFrame(
                                    pd.concat([geoms_gdf, geoms_file_gdf], ignore_index=True), 
                                    crs=geoms_file_gdf.crs)
                        
                    except Exception as ex:
                        logger.exception(f"Error reading {future_to_filepath_queue[future]}")
                    finally:
                        # Remove processed future from queue
                        nb_files_in_queue -= 1
                        del future_to_filepath_queue[future]
                    
                    # If all files are treated or enough geoms are read, write to file
                    if geoms_gdf is not None:
                        nb_geoms_ready_to_write = len(geoms_gdf)
                    else:
                        nb_geoms_ready_to_write = 0
                    if nb_files_done%100 == 0:
                        logger.debug(f"{nb_files_done} of {nb_files} processed ({(nb_files_done*100/nb_files):0.0f}%), {nb_geoms_ready_to_write} ready to write")
                    if(nb_geoms_ready_to_write > 0
                       and (nb_files_done == (nb_files-1)
                            or nb_geoms_ready_to_write > 10000)):
                        try:
                            # If output file isn't created yet, do so...
                            if output_tmp_file is None:
                                # Open the destination file
                                output_tmp_file = fiona.open(
                                        output_tmp_filepath, 'w', 
                                        driver=geofile_util.get_driver(output_tmp_filepath), 
                                        layer=layer, crs=geoms_gdf.crs.to_wkt(), 
                                        schema=gpd.io.file.infer_schema(geoms_gdf))
                            
                            # Now write to file
                            output_tmp_file.writerecords(geoms_gdf.iterfeatures())
                            geoms_gdf = None
                            logger.info(f"{nb_files_done} of {nb_files} processed + saved ({(nb_files_done*100/nb_files):0.0f}%)")
                        except Exception as ex:
                            raise Exception(f"Error saving gdf to {output_tmp_filepath}") from ex

            # If we get here, file is normally created successfully, so rename to real output
            if output_tmp_file is not None:
                output_tmp_file.close()
                os.rename(output_tmp_filepath, output_filepath)
    except Exception as ex:
        if output_tmp_file is not None:
            output_tmp_file.close()      
        raise Exception(f"Error creating file {output_tmp_filepath}") from ex
    
def read_prediction_file(
        filepath: Path,
        border_pixels_to_ignore: int = 0) -> gpd.geodataframe:
    ext_lower = filepath.suffix.lower()
    if ext_lower == '.geojson':
        return geofile_util.read_file(filepath)
    elif ext_lower == '.tif':
        return polygonize_pred_from_file(filepath, border_pixels_to_ignore)
    else:
        raise Exception(f"Unsupported extension: {ext_lower}")

def to_binary_uint8(
        in_arr: np.array, 
        thresshold_ok: int = 128) -> np.array:

    # Check input parameters
    if in_arr.dtype != np.uint8:
        raise Exception("Input should be dtype = uint8, not: {in_arr.dtype}")
        
    # First copy to new numpy array, otherwise input array is changed
    out_arr = np.copy(in_arr)
    out_arr[out_arr >= thresshold_ok] = 255
    out_arr[out_arr < thresshold_ok] = 0
    
    return out_arr

def postprocess_for_evaluation(
        image_filepath: Path,
        image_crs: str,
        image_transform,
        image_pred_filepath: Path,
        image_pred_uint8_cleaned_bin: np.array,
        output_dir: Path,
        output_suffix: str = None,
        input_image_dir: Optional[Path] = None,
        input_mask_dir: Optional[Path] = None,
        border_pixels_to_ignore: int = 0,
        force: bool = False):
    """
    This function postprocesses a prediction to make it easy to evaluate 
    visually if the result is OK by creating images of the different stages of 
    the prediction logic by creating the following output:
        - the input image
        - the mask image as digitized in the train files (if available)
        - the "raw" prediction image
        - the "raw" polygonized prediction, as an image
        - the simplified polygonized prediction, as an image

    The filenames start with a prefix:
        - if a mask is available, the % overlap between the result and the mask
        - if no mask is available, the % of pixels that is white
    
    Args
        
    """
    
    logger.debug(f"Start postprocess for {image_pred_filepath}")
    all_black = False
    try:      
        
        # If the image wasn't saved, it must have been all black
        if image_pred_filepath is None:
            all_black = True

        # Make sure the output dir exists...
        output_dir.mkdir(parents=True, exist_ok=True)
                
        # Determine the prefix to use for the output filenames
        pred_prefix_str = ''            
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
        if input_mask_dir is not None and input_mask_dir.exists():
            # Read mask file and get all needed info from it...
            mask_filepath = Path(str(image_filepath).
                    replace(str(input_image_dir), str(input_mask_dir)))
                
            # Check if this file exists, if not, look for similar files
            if not mask_filepath.exists():
                files = list(mask_filepath.parent.glob(mask_filepath.stem + '*'))
                if len(files) == 1:
                    mask_filepath = files[0]
                else:
                    message = f"Error finding mask file with {mask_filepath.stem + '*'}: {len(files)} mask(s) found"
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
            mask_copy_dest_filepath = (output_dir / 
                    f"{pred_prefix_str}{image_filepath.stem}_mask{mask_filepath.suffix}")
            if not mask_copy_dest_filepath.exists():
                shutil.copyfile(mask_filepath, mask_copy_dest_filepath)

        else:
            # If all_black, no need to calculate again
            if all_black: 
                pct_black = 1
            else:
                # Calculate percentage black pixels
                pct_black = 1 - ((image_pred_uint8_cleaned_bin.sum()/250)
                                    /image_pred_uint8_cleaned_bin.size)
            
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
        output_basefilepath = output_dir / f"{pred_prefix_str}{image_filepath.stem}{output_suffix}"
        image_dest_filepath = Path(str(output_basefilepath) + image_filepath.suffix)
        if not image_dest_filepath.exists():
            shutil.copyfile(image_filepath, image_dest_filepath)

        # Rename the prediction file so it also contains the prefix,... 
        if image_pred_filepath is not None:
            image_dest_filepath = Path(f"{str(output_basefilepath)}_pred{image_filepath.suffix}")
            if not image_dest_filepath.exists():
                shutil.move(str(image_pred_filepath), image_dest_filepath)

        # If all_black, we are ready now
        if all_black:
            logger.debug("All black prediction, no use proceding")
            return 
        
        polygonize_pred_for_evaluation(
                image_pred_uint8_bin=image_pred_uint8_cleaned_bin,
                image_crs=image_crs,
                image_transform=image_transform,
                output_basefilepath=output_basefilepath)
    
    except Exception as ex:
        message = f"Exception postprocessing prediction for {image_filepath}\n: file {image_pred_filepath}!!!"
        raise Exception(message) from ex

def polygonize_pred_for_evaluation(
        image_pred_uint8_bin,
        image_crs: str,
        image_transform,
        output_basefilepath: Path):

    # Polygonize result
    try:
        # Returns a list of tupples with (geometry, value)
        polygonized_records = list(rio_features.shapes(
                image_pred_uint8_bin, mask=image_pred_uint8_bin, transform=image_transform))
        
        # If nothing found, we can return
        if len(polygonized_records) == 0:
            logger.debug("This prediction didn't result in any polygons")
            return

        # Convert shapes to geopandas geodataframe 
        geoms = []
        for geom, _ in polygonized_records:
            geoms.append(sh.geometry.shape(geom))   
        geoms_gdf = gpd.GeoDataFrame(geoms, columns=['geometry'])
        geoms_gdf.crs = image_crs

        image_shape = image_pred_uint8_bin.shape
        image_width = image_shape[0]
        image_height = image_shape[1]

        # For easier evaluation, write the cleaned version as raster
        # Write the standard cleaned output to file
        logger.debug("Save binary prediction")
        image_pred_cleaned_filepath = Path(f"{str(output_basefilepath)}_pred_bin.tif")
        with rio.open(image_pred_cleaned_filepath, 'w', driver='GTiff', 
                        compress='lzw',
                        height=image_height, width=image_width, 
                        count=1, dtype=rio.uint8, 
                        crs=image_crs, transform=image_transform) as dst:
            dst.write(image_pred_uint8_bin, 1)
        
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
            
            # Write simplified wkt result to raster for comparing. 
            if len(geoms_simpl) > 0:
                # TODO: doesn't support multiple classes
                logger.debug('Before writing simpl rasterized file')
                image_pred_simpl_filepath = f"{str(output_basefilepath)}_pred_cleaned_simpl.tif"
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
                image_pred_simpl_filepath = f"{str(output_basefilepath)}_pred_cleaned_simpl_vis.tif"
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

    except Exception as ex:
        message = f"Exception while polygonizing to file {output_basefilepath}!"
        raise Exception(message) from ex

def polygonize_pred_from_file(
        image_pred_filepath: Path,
        border_pixels_to_ignore: int = 0,
        save_to_file: bool = False) -> gpd.geodataframe:

    try:
        with rio.open(image_pred_filepath) as image_ds:
            # Read geo info
            image_crs = image_ds.profile['crs']
            image_transform = image_ds.transform
            
            # Read pixels and change from (channels, width, height) to 
            # (width, height, channels) and normalize to values between 0 and 1
            image_data = image_ds.read()
    
         # Create binary version
        #image_data = rio_plot.reshape_as_image(image_data)
        image_pred_uint8_bin = to_binary_uint8(image_data, 125)

        output_basefilepath = None
        if save_to_file is True:
            output_basefilepath = image_pred_filepath.parent / image_pred_filepath.stem
        return polygonize_pred(
                image_pred_uint8_bin=image_pred_uint8_bin,
                image_crs=image_crs,
                image_transform=image_transform,
                image_pred_filepath=image_pred_filepath,
                output_basefilepath=output_basefilepath,
                border_pixels_to_ignore=border_pixels_to_ignore)

    except Exception as ex:
        raise Exception(f"Error in polygonize_pred_from_file on {image_pred_filepath}") from ex

def polygonize_pred(
        image_pred_uint8_bin,
        image_crs: str,
        image_transform,
        image_pred_filepath: Optional[Path] = None,
        output_basefilepath: Optional[Path] = None,
        border_pixels_to_ignore: int = 0) -> gpd.geodataframe:

    # Polygonize result
    try:
        # Returns a list of tupples with (geometry, value)
        polygonized_records = list(rio_features.shapes(
                image_pred_uint8_bin, mask=image_pred_uint8_bin, transform=image_transform))

        # If nothing found, we can return
        if len(polygonized_records) == 0:
            logger.warn(f"Prediction didn't result in any polygons: {image_pred_filepath}")
            return None

        # Convert shapes to geopandas geodataframe 
        geoms = []
        for geom, _ in polygonized_records:
            geoms.append(sh.geometry.shape(geom))   
        geoms_gdf = gpd.GeoDataFrame(geoms, columns=['geometry'])
        geoms_gdf.crs = image_crs

        # Apply a minimal simplify to the geometries based on the pixel size
        pixel_sizex = image_transform[0]
        pixel_sizey = -image_transform[4]
        min_pixel_size = min(pixel_sizex, pixel_sizey)
        # Calculate the tolerance as half the diagonal of the square formed 
        # by the min pixel size, rounded up in centimeter 
        simplify_tolerance = math.ceil(math.sqrt(pow(min_pixel_size, 2)/2)*100)/100
        geoms_gdf.geometry = geoms_gdf.geometry.simplify(simplify_tolerance)
        
        # Fix + remove empty geom rows
        geoms_gdf['geometry'] = geoms_gdf.geometry.apply(lambda geom: vector_util.fix(geom))
        geoms_gdf.dropna(subset=['geometry'], inplace=True)
        geoms_gdf = geoms_gdf.reset_index(drop=True).explode()

        # Calculate the bounds of the image in projected coordinates
        image_shape = image_pred_uint8_bin.shape
        image_width = image_shape[0]
        image_height = image_shape[1]
        image_bounds = rio.transform.array_bounds(
                image_height, image_width, image_transform)
        x_pixsize = get_pixelsize_x(image_transform)
        y_pixsize = get_pixelsize_y(image_transform)
        border_bounds = (image_bounds[0]+border_pixels_to_ignore*x_pixsize,
                            image_bounds[1]+border_pixels_to_ignore*y_pixsize,
                            image_bounds[2]-border_pixels_to_ignore*x_pixsize,
                            image_bounds[3]-border_pixels_to_ignore*y_pixsize)
        
        # Now we can calculate the "onborder" property
        geoms_gdf = vector_util.calc_onborder(geoms_gdf, border_bounds)

        # Write the geoms to file
        if output_basefilepath is not None:
            geom_filepath = Path(f"{str(output_basefilepath)}_pred_cleaned_2.geojson")
            geofile_util.to_file(geoms_gdf, geom_filepath)
        
        return geoms_gdf
            
    except Exception as ex:
        message = f"Exception while polygonizing to file {output_basefilepath}"
        raise Exception(message) from ex

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
    raise Exception("Not implemented")

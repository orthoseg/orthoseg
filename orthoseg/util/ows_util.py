# -*- coding: utf-8 -*-
"""
Module with generic usable utility functions to make some tasks using OWS services easier.
"""

import logging
import os
import time
import datetime
import random
import math
import itertools as it
import concurrent.futures as futures

import owslib
import owslib.wms
import rasterio as rio
import geopandas as gpd
import shapely.geometry as sh_geom

from orthoseg.util import vector_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
FORMAT_GEOTIFF = 'image/geotiff'
FORMAT_GEOTIFF_EXT = '.tif'
FORMAT_GEOTIFF_EXT_WORLD = '.tfw'

FORMAT_TIFF = 'image/tiff'
FORMAT_TIFF_EXT = '.tif'
FORMAT_TIFF_EXT_WORLD = '.tfw'

FORMAT_JPEG = 'image/jpeg'
FORMAT_JPEG_EXT = '.jpg'
FORMAT_JPEG_EXT_WORLD = '.jgw'

FORMAT_PNG = 'image/png'
FORMAT_PNG_EXT = '.png'
FORMAT_PNG_EXT_WORLD = '.pgw'

# Get a logger...
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def get_images_for_grid(
        wms_server_url: str,
        wms_version: str,
        wms_layernames: [str],
        srs: str,
        output_image_dir: str,
        image_gen_bounds: (float, float, float, float) = None,
        image_gen_roi_filepath: str = None,
        grid_xmin: float = 0.0,
        grid_ymin: float = 0.0,
        image_srs_pixel_x_size: int = 0.25,
        image_srs_pixel_y_size: int = 0.25,
        image_pixel_width: int = 1024,
        image_pixel_height: int = 1024,
        image_pixels_ignore_border: int = 0,
        nb_concurrent_calls: int = 1,
        random_sleep: float = 0.0,        
        image_format: str = FORMAT_GEOTIFF,
        image_format_save: str = None,
        tiff_compress: str = 'lzw',
        transparent: str = False,
        wms_layerstyles: [str] = ['default'],
        pixels_overlap: int = 0,
        column_start: int = 0,
        nb_images_to_skip: int = None,
        max_nb_images: int = -1,
        force: bool = False):
    
    ##### Init #####
    if image_format_save is None:
        image_format_save = image_format

    srs_width = math.fabs(image_pixel_width*image_srs_pixel_x_size)   # tile width in units of crs => 500 m
    srs_height = math.fabs(image_pixel_height*image_srs_pixel_y_size) # tile height in units of crs => 500 m
    is_srs_projected = rio.crs.CRS.from_string(srs).is_projected
    
    # Read the region of interest file if provided
    roi_gdf = None
    if image_gen_roi_filepath:
        # Open vector layer
        logger.info(f"Open vector file {image_gen_roi_filepath}")
        roi_gdf = gpd.read_file(image_gen_roi_filepath)
                
        # If the generate_window wasn't specified, calculate the bounds
        # based on the roi (but make sure they fit the grid!)
        if not image_gen_bounds:
            roi_bounds = roi_gdf.geometry.total_bounds
            image_gen_bounds = (roi_bounds[0]-((roi_bounds[0]-grid_xmin)%srs_width),
                                roi_bounds[1]-((roi_bounds[1]-grid_ymin)%srs_height),
                                roi_bounds[2]+(grid_xmin-((roi_bounds[2]-grid_xmin)%srs_width)),
                                roi_bounds[3]+(grid_ymin-((roi_bounds[3]-grid_ymin)%srs_height)))
            logger.info(f"roi_bounds: {roi_bounds}, image_gen_bounds: {image_gen_bounds}")
        
        # If there are large objects in the roi, segment them to speed up
        # TODO: implement usefull check to see if segmenting is usefull...
        # TODO: support creating a grid in latlon????
        if is_srs_projected:
            
            # Create grid
            grid = vector_util.create_grid(
                    xmin=image_gen_bounds[0], ymin=image_gen_bounds[1], 
                    xmax=image_gen_bounds[2], ymax=image_gen_bounds[3],
                    cell_width=4000, cell_height=4000)
            # Create intersection layer between grid and roi
            #roi_gdf = roi_gdf.geometry.intersection(grid.geometry)
            roi_gdf = gpd.overlay(grid, roi_gdf, how='intersection')
            
            # Explode possible multipolygons to polygons
            roi_gdf = roi_gdf.reset_index(drop=True).explode()
                        
            #roi_gdf.to_file("X:\\Monitoring\\OrthoSeg\\roi_gridded.shp")
            
    # Check if the image_gen_bounds are compatible with the grid...
    if (image_gen_bounds[0]-grid_xmin)%srs_width != 0:
        xmin_new = image_gen_bounds[0] - ((image_gen_bounds[0]-grid_xmin)%srs_width)
        logger.warning(f"xmin {image_gen_bounds[0]} in compatible with grid, {xmin_new} will be used")
        image_gen_bounds = (xmin_new, image_gen_bounds[1], image_gen_bounds[2], image_gen_bounds[3])
    if (image_gen_bounds[1]-grid_ymin)%srs_height != 0:
        ymin_new = image_gen_bounds[1] - ((image_gen_bounds[1]-grid_ymin)%srs_height)
        logger.warning(f"ymin {image_gen_bounds[1]} incompatible with grid, {ymin_new} will be used")
        image_gen_bounds = (image_gen_bounds[0], ymin_new, image_gen_bounds[2], image_gen_bounds[3])
    if (image_gen_bounds[2]-grid_xmin)%srs_width != 0:
        xmax_new = image_gen_bounds[2] + srs_width - ((image_gen_bounds[2]-grid_xmin)%srs_width)
        logger.warning(f"xmax {image_gen_bounds[2]} incompatible with grid, {xmax_new} will be used")
        image_gen_bounds = (image_gen_bounds[0], image_gen_bounds[1], xmax_new, image_gen_bounds[3])
    if (image_gen_bounds[3]-grid_ymin)%srs_height != 0:
        ymax_new = image_gen_bounds[3] + srs_height - ((image_gen_bounds[3]-grid_ymin)%srs_height)
        logger.warning(f"ymax {image_gen_bounds[3]} incompatible with grid, {ymax_new} will be used")
        image_gen_bounds = (image_gen_bounds[0], image_gen_bounds[1], image_gen_bounds[2], ymax_new)

    # Calculate width and height...
    dx = math.fabs(image_gen_bounds[0] - image_gen_bounds[2]) # area width in units of crs
    dy = math.fabs(image_gen_bounds[1] - image_gen_bounds[3]) # area height in units of crs
    cols = int(math.ceil(dx / srs_width)) + 1
    rows = int(math.ceil(dy / srs_height)) + 1

    # Inits to start getting images 
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    auth = owslib.util.Authentication()
    wms = owslib.wms.WebMapService(wms_server_url, version=wms_version, auth=auth)

    with futures.ThreadPoolExecutor(nb_concurrent_calls) as pool:

        # Loop through all columns and get the images...
        start_time = None
        nb_todo = cols*rows
        nb_processed = 0
        nb_downloaded = 0
        nb_ignore_in_progress = 0
        bbox_list = []
        size_list = []
        output_filename_list = []
        for col in range(column_start, cols):
            
            image_xmin = col * srs_width + image_gen_bounds[0]
            image_xmax = (col + 1) * srs_width + image_gen_bounds[0]
    
            # If overlapping images are wanted... increase image bbox
            if pixels_overlap:
                image_xmin = image_xmin-(pixels_overlap*image_srs_pixel_x_size)
                image_xmax = image_xmax+(pixels_overlap*image_srs_pixel_x_size)
                
            # Put all the images of this column in a dir
            if is_srs_projected:
                output_dir = os.path.join(output_image_dir, f"{image_xmin:06.0f}")
            else:
                output_dir = os.path.join(output_image_dir, f"{image_xmin:09.4f}")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
                
            logger.info(f"Start processing column {col}")
            for row in range(0, rows):
                nb_processed += 1
                
                # To be able to quickly get images spread over the roi...
                if(nb_images_to_skip
                   and (nb_processed%nb_images_to_skip) != 0):
                    #logger.debug(f"Skip this image, because {nb_processed}%{nb_images_to_skip} is not 0")
                    continue
    
                # Calculate y bounds
                image_ymin = row * srs_height + image_gen_bounds[1]
                image_ymax = (row + 1) * srs_height + image_gen_bounds[1]
    
                # If overlapping images are wanted... increase image bbox
                if pixels_overlap:
                    image_ymin = image_ymin-(pixels_overlap*image_srs_pixel_y_size)
                    image_ymax = image_ymax+(pixels_overlap*image_srs_pixel_y_size)
    
                # Create output filename
                output_filename = create_filename(
                        srs=srs,
                        bbox=(image_xmin, image_ymin, image_xmax, image_ymax),
                        size=(image_pixel_width+2*pixels_overlap, 
                              image_pixel_height+2*pixels_overlap),
                        image_format=image_format,
                        layername='_'.join(wms_layernames))
                output_filepath = os.path.join(output_dir, output_filename)
                if not force and os.path.exists(output_filepath):
                    nb_ignore_in_progress += 1
                    logger.debug("    -> image exists already, so skip")
                    continue
                
                # If roi was provided, check first if the current image overlaps
                # TODO: using an chopped up version of the ROI is probably faster
                # TODO: possibly checking if the dest file exists already before 
                # this is faster...
                
                if roi_gdf is not None:
                    image_shape = sh_geom.box(image_xmin, image_ymin, image_xmax, image_ymax)
                    
                    #intersections = gpd.sjoin(roi_gdf, lines, how="inner", op='intersects')
                    spatial_index = roi_gdf.sindex
                    possible_matches_index = list(spatial_index.intersection(image_shape.bounds))
                    possible_matches = roi_gdf.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(image_shape)]
                    
                    if len(precise_matches) == 0:
                        nb_ignore_in_progress += 1
                        logger.debug("    -> image doesn't overlap with roi, so skip")
                        continue                                       
                    
                    # TODO: cleanup old code
                    '''
                    if not roi_gdf.intersects(image_shape).any():
                        nb_ignore_in_progress += 1
                        logger.debug("    -> image doesn't overlap with roi, so skip")
                        continue                                       
                    '''
                    '''
                    intersects = False
                    # Loop through the roi rows. If one of them overlaps -> OK.
                    for roi_row in roi_rows:
                        # Check first on bounds to improve performance
                        if(roi_row['geom_bounds'].intersection(image_shape)
                           and roi_row['geom'].intersection(image_shape)):
                            intersects = True
                            break
                    if not intersects:
                        nb_ignore_in_progress += 1
                        logger.debug("    -> image doesn't overlap with roi, so skip")
                        continue
                    '''
                
                # Now we are getting to start fetching images... init start_time 
                if start_time is None:
                    start_time = datetime.datetime.now()
                
                # Append the image info to the batch arrays so they can be treated in 
                # bulk if the batch size is reached
                bbox_list.append((image_xmin, image_ymin, image_xmax, image_ymax))
                size_list.append((image_pixel_width+2*pixels_overlap, 
                                  image_pixel_height+2*pixels_overlap))
                output_filename_list.append(output_filename)
                
                # If the batch size is reached or we are at the last images
                nb_images_in_batch = len(bbox_list)
                if(nb_images_in_batch == nb_concurrent_calls or nb_processed == (nb_todo-1)):
    
                    start_time_batch_read = datetime.datetime.now()
                    
                    # Exec in parallel 
                    read_results = pool.map(
                            getmap_to_file,        # Function 
                            it.repeat(wms),
                            it.repeat(wms_layernames),
                            it.repeat(output_dir),
                            it.repeat(srs),
                            bbox_list,
                            size_list,
                            it.repeat(image_format),
                            it.repeat(image_format_save),
                            output_filename_list,
                            it.repeat(transparent),
                            it.repeat(tiff_compress),
                            it.repeat(wms_layerstyles),
                            it.repeat(random_sleep),
                            it.repeat(image_pixels_ignore_border),
                            it.repeat(force))
                            
                    for _ in read_results:
                        nb_downloaded += 1
                    
                    # Progress
                    logger.debug(f"Process image {nb_processed} out of {cols*rows}: {nb_processed/(cols*rows):.2f} %")

                    # Log the progress and prediction speed
                    time_passed_s = (datetime.datetime.now()-start_time).total_seconds()
                    time_passed_lastbatch_s = (datetime.datetime.now()-start_time_batch_read).total_seconds()
                    if time_passed_s > 0 and time_passed_lastbatch_s > 0:
                        nb_per_hour = ((nb_processed-nb_ignore_in_progress)/time_passed_s) * 3600
                        nb_per_hour_lastbatch = (nb_images_in_batch/time_passed_lastbatch_s) * 3600
                        hours_to_go = (int)((nb_todo - nb_processed)/nb_per_hour)
                        min_to_go = (int)((((nb_todo - nb_processed)/nb_per_hour)%1)*60)
                        print(f"\r{hours_to_go}:{min_to_go} left for {nb_todo-nb_processed} images at {nb_per_hour:0.0f}/h ({nb_per_hour_lastbatch:0.0f}/h last batch), with {nb_ignore_in_progress} skipped",
                              end='', flush=True)
                    
                    # Reset variable for next batch
                    bbox_list = []
                    size_list = []
                    output_filename_list = []

                    if max_nb_images > -1 and nb_downloaded >= max_nb_images:
                        return            
                    '''
                    # TODO: cleanup
                    #logger.info(f"Nb processed: {nb_processed} of {cols*rows} ({nb_processed/(cols*rows):.2f}%), Nb downloaded: {counter_downloaded}")
                    # Log the progress and prediction speed once 10 seconds have passed
                    secs_since_last_log = (datetime.datetime.now()-last_progress_log_time).total_seconds()
                    if secs_since_last_log > 10:
                        last_progress_log_time = datetime.datetime.now()
                        secs_passed_total = (datetime.datetime.now()-start_time).total_seconds()
                        nb_per_hour = ((nb_processed-nb_ignore_in_progress)/secs_passed_total) * 3600
                        hours_to_go = (int)((nb_todo-nb_processed)/nb_per_hour)
                        min_to_go = (int)((((nb_todo-nb_processed)/nb_per_hour)%1)*60)
                        print(f"\r{hours_to_go}:{min_to_go} left for {nb_todo-nb_processed} todo at {nb_per_hour:0.0f}/h",
                              end="", flush=True)
                    '''
                    
def getmap_to_file(
        wms: owslib.wms.WebMapService,
        layers: [str],
        output_dir: str,
        srs: str,
        bbox,
        size,
        image_format: str = FORMAT_GEOTIFF,
        image_format_save: str = None,
        output_filename: str = None,
        transparent: bool = False,
        tiff_compress: str = 'lzw',
        styles: [str] = ['default'],
        random_sleep: float = 0.0,
        image_pixels_ignore_border: int = 0,
        force: bool = False) -> str:
    """

    Args
        random_sleep: sleep a random time between 0 and this amount of seconds
                      between requests tot the WMS server
    """
    ##### Init #####
    # If no seperate save format is specified, use the standard image_format
    if image_format_save is None:
        image_format_save = image_format

    # If there isn't a filename supplied, create one...
    if output_filename is None:
        output_filename = create_filename(
                srs=srs, 
                bbox=bbox, size=size, 
                image_format=image_format_save,
                layername='_'.join(layers))

    # Create full output filepath
    output_filepath = os.path.join(output_dir, output_filename)

    # If force is false and file exists already, stop...
    if force == False and os.path.exists(output_filepath):
        logger.debug(f"File already exists, skip: {output_filepath}")
        return None

    logger.debug(f"Get image to {output_filepath}")

    ##### Get image #####
    # Retry 10 times...
    nb_retries = 0
    time_sleep = 0
    while True:
        try:
            logger.debug(f"Start call GetMap for bbox {bbox}")

            # Some hacks for special cases...
            bbox_for_getmap = bbox
            size_for_getmap = size
            # Dirty hack to ask a bigger picture, and then remove the border again!
            if image_pixels_ignore_border > 0:
                x_pixsize = (bbox[2]-bbox[0])/size[0]
                y_pixsize = (bbox[3]-bbox[1])/size[1]
                bbox_for_getmap = (bbox[0] - x_pixsize*image_pixels_ignore_border,
                                   bbox[1] - y_pixsize*image_pixels_ignore_border,
                                   bbox[2] + x_pixsize*image_pixels_ignore_border,
                                   bbox[3] + y_pixsize*image_pixels_ignore_border)
                size_for_getmap = (size[0] + 2*image_pixels_ignore_border,
                                   size[1] + 2*image_pixels_ignore_border)
            # Dirty hack to support y,x cordinate system
            if srs.lower() == 'epsg:3059':
                bbox_for_getmap = (bbox_for_getmap[1], bbox_for_getmap[0], 
                                   bbox_for_getmap[3], bbox_for_getmap[2])

            response = wms.getmap(
                    layers=layers,
                    styles=styles,
                    srs=srs,
                    bbox=bbox_for_getmap,
                    size=size_for_getmap,
                    format=image_format,
                    transparent=transparent)
            logger.debug(f"Finished doing request {response.geturl()}")
            
            # If a random sleep was specified... apply it
            if random_sleep:
                time.sleep(random.uniform(0, random_sleep))
            
            # Image was retrieved... so stop loop
            break

        except Exception as ex:
            # Retry 10 times... and increase sleep time every time
            if nb_retries < 10:
                nb_retries += 1
                time_sleep += 5                
                time.sleep(time_sleep)
                continue
            else:
                message = f"Retried 10 times and didn't work, with layers: {layers}, styles: {styles}"
                logger.error(message)
                raise Exception(message) from ex

    ##### Save image to file #####
    # Write image to file...
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_filepath, 'wb') as image_file:
        image_file.write(response.read())

    ##### Make the output image compliant with image_format_save #####

    # If geotiff is asked, check if the the coordinates are embedded...
    if image_format_save == FORMAT_GEOTIFF:
        # Read output image to check if coördinates are there
        with rio.open(output_filepath) as image_ds:
            image_profile_orig = image_ds.profile
            image_transform_affine = image_ds.transform

            if image_pixels_ignore_border == 0:
                image_data = image_ds.read()
            else:
                image_data = image_ds.read(window=rio.windows.Window(
                        image_pixels_ignore_border, image_pixels_ignore_border, 
                        size[0], size[1]))

        logger.debug(f"original image_profile: {image_profile_orig}")

        # If coordinates are not embedded add them, if image_pixels_ignore_border
        # change them
        if((image_transform_affine[2] == 0 and image_transform_affine[5] == 0)
            or image_pixels_ignore_border > 0):
            logger.debug(f"Coordinates not present in image, driver: {image_profile_orig['driver']}")

            # If profile format is not gtiff, create new profile
            if image_profile_orig['driver'] != 'GTiff':
                image_profile_gtiff = rio.profiles.DefaultGTiffProfile.defaults

                # Copy appropriate info from source file
                image_profile_gtiff.update(
                        count=image_profile_orig['count'], width=size[0], height=size[1],
                        nodata=image_profile_orig['nodata'], dtype=image_profile_orig['dtype'])
                image_profile = image_profile_gtiff
            else:
                image_profile = image_profile_orig

            # Set the asked compression
            image_profile_gtiff.update(compress=tiff_compress)

            logger.debug(f"Map request bbox: {bbox_for_getmap}")
            logger.debug(f"Map request size: {size_for_getmap}")

            # For some coordinate systems apparently the axis ordered is configured wrong in LibOWS :-(
            srs_pixel_x_size = (bbox[2]-bbox[0])/size[0]
            srs_pixel_y_size = (bbox[1]-bbox[3])/size[1]

            logger.debug(f"Coordinates to put in geotiff:\n" +
                         f"    - x-component of the pixel width, W-E: {srs_pixel_x_size}\n" +
                         f"    - y-component of the pixel width, W-E (0 if image is exactly N up): 0\n" +
                         f"    - top-left x: {bbox[0]}\n" +
                         f"    - x-component of the pixel height, N-S (0 if image is exactly N up): \n" +
                         f"    - y-component of the pixel height, N-S: {srs_pixel_y_size}\n" +
                         f"    - top-left y: {bbox[3]}")

            # Add transform and srs to the profile
            image_profile.update(
                    transform = rio.transform.Affine(
                                srs_pixel_x_size, 0, bbox[0],
                                0 , srs_pixel_y_size, bbox[3]),
                    crs=srs)

            # Delete output file, and write again
            os.remove(output_filepath)
            with rio.open(output_filepath, 'w', **image_profile) as image_file:
                image_file.write(image_data)

    else:
        # For file formats that doesn't support coordinates, we add a worldfile       
        srs_pixel_x_size = (bbox[2]-bbox[0])/size[0]
        srs_pixel_y_size = (bbox[1]-bbox[3])/size[1]

        path_noext, _ = os.path.splitext(output_filepath)
        ext_world = get_world_ext_for_image_format(image_format_save)
        output_worldfile_filepath = f"{path_noext}{ext_world}"
        
        with open(output_worldfile_filepath, 'w') as wld_file:
            wld_file.write(f"{srs_pixel_x_size}")
            wld_file.write("\n0.000")
            wld_file.write("\n0.000")
            wld_file.write(f"\n{srs_pixel_y_size}")
            wld_file.write(f"\n{bbox[0]}")
            wld_file.write(f"\n{bbox[3]}")
        
        # If the image format to save is different, or if a border needs to be ignored
        if(image_format != image_format_save 
           or image_pixels_ignore_border > 0):

            # Read image 
            with rio.open(output_filepath) as image_ds:
                image_profile_orig = image_ds.profile
                image_transform_affine = image_ds.transform

                # If border needs to be ignored, only read data we are interested in
                if image_pixels_ignore_border == 0:
                    image_data = image_ds.read()
                else:
                    image_data = image_ds.read(window=rio.windows.Window(
                            image_pixels_ignore_border, image_pixels_ignore_border, 
                            size[0], size[1]))
            
            # If same save format, reuse profile
            if image_format == image_format_save:
                image_profile_output = image_profile_orig
                if image_pixels_ignore_border != 0:
                    image_profile_output.update(width=size[0], height=size[1])
            else:
                if image_format_save == FORMAT_TIFF:
                    driver = 'GTiff'
                    compress=tiff_compress
                else:
                    raise Exception(f"Unsupported image_format_save: {image_format_save}")
                image_profile_output = rio.profiles.Profile(
                        width=size[0], height=size[1], count=image_profile_orig['count'],
                        nodata=image_profile_orig['nodata'], dtype=image_profile_orig['dtype'],
                        compress=compress, driver=driver)

            # Delete output file, and write again
            os.remove(output_filepath)
            image_profile_output = get_cleaned_write_profile(image_profile_output)
            with rio.open(output_filepath, 'w', **image_profile_output) as image_file:
                image_file.write(image_data)               

            #raise Exception(f"Different save format not supported between {image_format} and {image_format_save}")

    return output_filepath

def create_filename(srs: str,
                    bbox,
                    size,
                    image_format: str,
                    layername: str):
    
    # Get image extension based on format
    image_ext = get_ext_for_image_format(image_format)

    # Use different file names for projected vs geographic SRS
    is_srs_projected = rio.crs.CRS.from_string(srs).is_projected
    if is_srs_projected:
        output_filename = f"{bbox[0]:06.0f}_{bbox[1]:06.0f}_{bbox[2]:06.0f}_{bbox[3]:06.0f}_{size[0]}_{size[1]}_{layername}{image_ext}"
    else:
        output_filename = f"{bbox[0]:09.4f}_{bbox[1]:09.4f}_{bbox[2]:09.4f}_{bbox[3]:09.4f}_{size[0]}_{size[1]}_{layername}{image_ext}"

    return output_filename

def get_ext_for_image_format(image_format: str) -> str:
    # Choose image extension based on format
    if image_format == FORMAT_GEOTIFF:
        return FORMAT_GEOTIFF_EXT
    elif image_format == FORMAT_TIFF:
        return FORMAT_TIFF_EXT
    elif image_format == FORMAT_JPEG:
        return FORMAT_JPEG_EXT
    elif image_format == FORMAT_PNG:
        return FORMAT_PNG_EXT
    else:
        raise Exception(f"get_ext_for_image_format for image format {image_format} is not implemented!")

def get_world_ext_for_image_format(image_format: str) -> str:
    # Choose image extension based on format
    if image_format == FORMAT_GEOTIFF:
        return FORMAT_GEOTIFF_EXT_WORLD
    elif image_format == FORMAT_TIFF:
        return FORMAT_TIFF_EXT_WORLD
    elif image_format == FORMAT_JPEG:
        return FORMAT_JPEG_EXT_WORLD
    elif image_format == FORMAT_PNG:
        return FORMAT_PNG_EXT_WORLD
    else:
        raise Exception(f"get_world_ext_for_image_format for image format {image_format} is not implemented!")

def get_cleaned_write_profile(profile: dict) -> dict:

    # Depending on the driver, different profile keys are supported    
    if profile['driver'] == 'JPEG':
        # Don't copy profile keys to cleaned version that are not supported for JPEG
        profile_cleaned = {}
        for profile_key in profile:
            if profile_key not in ['tiled', 'compress', 'interleave', 'photometric']:
                profile_cleaned[profile_key] = profile[profile_key]
    elif profile['driver'] == 'PNG':
        # Don't copy profile keys to cleaned version that are not supported for JPEG
        profile_cleaned = {}
        for profile_key in profile:
            if profile_key not in ['tiled', 'interleave']:
                profile_cleaned[profile_key] = profile[profile_key]
    else:
        profile_cleaned = profile.copy()

    return profile_cleaned

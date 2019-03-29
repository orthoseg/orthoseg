# -*- coding: utf-8 -*-
"""
Helper module to support some tasks regarding using OWS services.

@author: Pieter Roggemans
"""

import logging
import os
import time
import datetime
import random
import math
import itertools as it
import concurrent.futures as futures

from owslib.wms import WebMapService
import rasterio as rio
import geopandas as gpd
import shapely.geometry as sh_geom

import orthoseg.vector.vector_helper as vh

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

def get_images_for_grid(wms_server_url: str,
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
                        nb_concurrent_calls: int = 0,
                        image_format: str = FORMAT_GEOTIFF,
                        tiff_compress: str = 'lzw',
                        transparent: str = False,
                        wms_server_layers_styles: [str] = ['default'],
                        pixels_overlap: int = 0,
                        random_sleep: float = 0.0,
                        column_start: int = 0,
                        nb_images_to_skip: int = None,
                        force: bool = False):

    
    # TODO: might be interesting to parallelize a bit... it is rather slow...
    
    srs_width = math.fabs(image_pixel_width*image_srs_pixel_x_size)   # tile width in units of crs => 500 m
    srs_height = math.fabs(image_pixel_height*image_srs_pixel_y_size) # tile height in units of crs => 500 m
    is_srs_projected = rio.crs.CRS.from_string(srs).is_projected
    
    # Read the region of interest file if provided
    roi_gdf = None
    if image_gen_roi_filepath:
        # Open vector layer
        # TODO: performance can be improved by intersecting with a 2x2 km grid
        # so calculating intersections per cell afterwards is cheaper...
        # eg: grid res_intersection = geopandas.overlay(df1, df2, how='intersection')
        logger.info(f"Open vector file {image_gen_roi_filepath}")
        roi_gdf = gpd.read_file(image_gen_roi_filepath)
        
        '''
        roi_data = fiona.open(image_gen_roi_filepath)

        # Convert to lists of shapely geometries
        roi_rows = []
        for roi_row in roi_data:
            geom = sh_geom.shape(roi_row['geometry'])
            geom_bounds = geom.bounds
            # Needs to be a shapely geom, not a tupple to use for intersection
            geom_bounds_box = sh_geom.box(geom_bounds[0], geom_bounds[1],
                                          geom_bounds[2], geom_bounds[3])
            roi_rows.append({'geom_bounds': geom_bounds_box, 'geom': geom})
        '''
        
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
            grid = vh.create_grid(xmin=image_gen_bounds[0], ymin=image_gen_bounds[1], 
                               xmax=image_gen_bounds[2], ymax=image_gen_bounds[3],
                               cell_width=4000, cell_height=4000)
            # Create intersection layer between grid and roi
            #roi_gdf = roi_gdf.geometry.intersection(grid.geometry)
            roi_gdf = gpd.overlay(grid, roi_gdf, how='intersection')
            
            # Explode possible multipolygons to polygons
            roi_gdf = roi_gdf.reset_index(drop=True).explode()
                        
            #roi_gdf.to_file("X:\\Monitoring\\OrthoSeg\\roi_gridded.shp")
            
    # Check if the image_gen_bounds are compatible with the grid...
    error_message = None
    if((image_gen_bounds[0]-grid_xmin)%srs_width != 0):
        error_message += f"image_gen_bounds[0] (xmin) is not compatible with grid!\n"
    elif((image_gen_bounds[2]-grid_xmin)%srs_width != 0):
        error_message += f"image_gen_bounds[2] (xmax) is not compatible with grid!\n"
    elif((image_gen_bounds[1]-grid_ymin)%srs_height != 0):
        error_message += f"image_gen_bounds[1] (ymin) is not compatible with grid!\n"
    elif((image_gen_bounds[3]-grid_ymin)%srs_height != 0):
        error_message += f"image_gen_bounds[3] (ymax) is not compatible with grid!\n"

    # If there was an error, stop!
    if error_message:
        logger.critical(error_message)
        raise Exception(error_message)

    dx = math.fabs(image_gen_bounds[0] - image_gen_bounds[2]) # area width in units of crs
    dy = math.fabs(image_gen_bounds[1] - image_gen_bounds[3]) # area height in units of crs

    cols = int(math.ceil(dx / srs_width)) + 1
    rows = int(math.ceil(dy / srs_height)) + 1

    print(f"Number rows: {rows}, number columns: {cols}")

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    wms = WebMapService(wms_server_url, version='1.3.0')

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
                        image_format=image_format)
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
                
                # Now we are getting to start fetcing images... init start_time 
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
                            it.repeat([wms_layernames]),
                            it.repeat(output_dir),
                            it.repeat(srs),
                            bbox_list,
                            size_list,
                            it.repeat(image_format),
                            output_filename_list,
                            it.repeat(transparent),
                            it.repeat(tiff_compress),
                            it.repeat(wms_server_layers_styles),
                            it.repeat(random_sleep),
                            it.repeat(force))
                            
                    for j, read_result in enumerate(read_results):
                        nb_downloaded += 1
    
                    '''
                    # Now really get the image
                    res = getmap_to_file(
                            wms=wms,
                            layers=wms_layernames,
                            output_dir=output_dir,
                            srs=srs,
                            bbox=(image_xmin, image_ymin, image_xmax, image_ymax),
                            size=(image_pixel_width+2*pixels_overlap, 
                                  image_pixel_height+2*pixels_overlap),
                            image_format=image_format,
                            output_filename=output_filename,
                            transparent=transparent,
                            tiff_compress=tiff_compress,
                            layers_styles=wms_server_layers_styles,
                            random_sleep=random_sleep,
                            force=force)
                    if(res is not None):
                        nb_downloaded += 1
                    '''
                    
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
                        print(f"\r{hours_to_go}:{min_to_go} left for {nb_todo-nb_processed} images at {nb_per_hour:0.0f}/h ({nb_per_hour_lastbatch:0.0f}/h last batch)",
                              end='', flush=True)
                    
                    # Reset variable for next batch
                    bbox_list = []
                    size_list = []
                    output_filename_list = []
                    
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
                    
def getmap_to_file(wms: WebMapService,
                   layers: [str],
                   output_dir: str,
                   srs: str,
                   bbox,
                   size,
                   image_format: str = FORMAT_GEOTIFF,
                   output_filename: str = None,
                   transparent: bool = False,
                   tiff_compress: str = 'lzw',
                   layers_styles: [str] = ['default'],
                   random_sleep: float = 0.0,
                   force: bool = False) -> str:
    """

    Args
        random_sleep: sleep a random time between 0 and this amount of seconds
                      between requests tot the WMS server
    """
    # If there isn't a filename supplied, create one...
    if output_filename is None:
        # Choose image extension based on format
        if image_format == FORMAT_GEOTIFF:
            image_ext = FORMAT_GEOTIFF_EXT
        elif image_format == FORMAT_TIFF:
            image_ext = FORMAT_TIFF_EXT
        elif image_format == FORMAT_JPEG:
            image_ext = FORMAT_JPEG_EXT
        elif image_format == FORMAT_PNG:
            image_ext = FORMAT_PNG_EXT
        else:
            raise Exception(f"image format {image_format} is not implemented!")
            image_ext = None

        # Use different file names for projected vs geographic SRS
        is_srs_projected = rio.crs.CRS.from_string(srs).is_projected
        if is_srs_projected:
            output_filename = f"{bbox[0]:06.0f}_{bbox[1]:06.0f}_{bbox[2]:06.0f}_{bbox[3]:06.0f}_{size[0]}_{size[1]}{image_ext}"
        else:
            output_filename = f"{bbox[0]:09.4f}_{bbox[1]:09.4f}_{bbox[2]:09.4f}_{bbox[3]:09.4f}_{size[0]}_{size[1]}{image_ext}"

    # Create full output filepath
    output_filepath = os.path.join(output_dir, output_filename)

    # If force is false and file exists already, stop...
    if force == False and os.path.exists(output_filepath):
        logger.debug(f"File already exists, skip: {output_filepath}")
        return None #output_filepath

    # Retry 10 times...
    nb_retries = 0
    while True:
        try:
            logger.debug(f"Call GetMap for bbox {bbox}")
            img = wms.getmap(layers=layers,
                             styles=layers_styles,
                             srs=srs,
                             bbox=bbox,
                             size=size,
                             format=image_format,
                             transparent=transparent)
            
            # If a random sleep was specified... apply it
            if random_sleep:
                time.sleep(random.uniform(0, random_sleep))
            
            # Image was retrieved... so stop loop
            break

        except:
            # Retry 10 times...
            if nb_retries < 10:
                nb_retries += 1
                time.sleep(10)
                continue
            else:
                logger.error(f"Retried 10 times and didn't work, with layers: {layers}, styles: {layers_styles}")
                raise

    # Write image to file...
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_filepath, 'wb') as image_file:
        image_file.write(img.read())

    # If geotiff is asked, check if the the coordinates are embedded...
    if image_format == FORMAT_GEOTIFF:
        # Read output image to check if coördinates are there
        with rio.open(output_filepath) as image_ds:
            image_profile_orig = image_ds.profile
            image_transform_affine = image_ds.transform
            image_data = image_ds.read()

        logger.debug(f"original image_profile: {image_profile_orig}")

        # If coordinates are not embedded add them!
        if image_transform_affine[2] == 0 and image_transform_affine[5] == 0:
            logger.debug(f"Coordinates not present in image, driver: {image_profile_orig['driver']}")

            # If profile format is not gtiff, create new profile
            if image_profile_orig['driver'] != 'GTiff':
                image_profile_gtiff = rio.profiles.DefaultGTiffProfile.defaults

                # Copy appropriate info from source file
                image_profile_gtiff.update(
#                        driver=rio.profiles.DefaultGTiffProfile.driver,
                        count=image_profile_orig['count'],
                        width=image_profile_orig['width'],
                        height=image_profile_orig['height'],
                        nodata=image_profile_orig['nodata'],
                        dtype=image_profile_orig['dtype'])
                image_profile = image_profile_gtiff
            else:
                image_profile = image_profile_orig

            # Set the asked comression
            image_profile_gtiff.update(compress=tiff_compress)

            logger.debug(f"Map request bbox: {bbox}")
            logger.debug(f"Map request size: {size}")

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

            logger.debug(f"image_profile that will be used: {image_profile}")
            with rio.open(output_filepath, 'w', **image_profile) as image_file:
                image_file.write(image_data)

    else:
        # If a file format is asked that doesn't support coordinates,
        # write world file
        srs_pixel_x_size = (bbox[2]-bbox[0])/size[0]
        srs_pixel_y_size = (bbox[1]-bbox[3])/size[1]

        path_noext, ext = os.path.splitext(output_filepath)
        if ext.lower() == FORMAT_TIFF_EXT:
            output_worldfile_filepath = f"{path_noext}{FORMAT_TIFF_EXT_WORLD}"
        elif ext.lower() == FORMAT_JPEG_EXT:
            output_worldfile_filepath = f"{path_noext}{FORMAT_JPEG_EXT_WORLD}"
        elif ext.lower() == FORMAT_PNG_EXT:
            output_worldfile_filepath = f"{path_noext}{FORMAT_PNG_EXT_WORLD}"
        else:
            error_message = "Error: extension not supported to create world file: {ext.lower()}"
            logger.critical(error_message)
            raise Exception(error_message)

        with open(output_worldfile_filepath, 'w') as wld_file:
            wld_file.write(f"{srs_pixel_x_size}")
            wld_file.write("\n0.000")
            wld_file.write("\n0.000")
            wld_file.write(f"\n{srs_pixel_y_size}")
            wld_file.write(f"\n{bbox[0]}")
            wld_file.write(f"\n{bbox[3]}")

    return output_filepath

def create_filename(srs: str,
                    bbox,
                    size,
                    image_format: str):
    
    # Choose image extension based on format
    if image_format == FORMAT_GEOTIFF:
        image_ext = FORMAT_GEOTIFF_EXT
    elif image_format == FORMAT_TIFF:
        image_ext = FORMAT_TIFF_EXT
    elif image_format == FORMAT_JPEG:
        image_ext = FORMAT_JPEG_EXT
    elif image_format == FORMAT_PNG:
        image_ext = FORMAT_PNG_EXT
    else:
        raise Exception(f"image format {image_format} is not implemented!")
        image_ext = None

    # Use different file names for projected vs geographic SRS
    is_srs_projected = rio.crs.CRS.from_string(srs).is_projected
    if is_srs_projected:
        output_filename = f"{bbox[0]:06.0f}_{bbox[1]:06.0f}_{bbox[2]:06.0f}_{bbox[3]:06.0f}_{size[0]}_{size[1]}{image_ext}"
    else:
        output_filename = f"{bbox[0]:09.4f}_{bbox[1]:09.4f}_{bbox[2]:09.4f}_{bbox[3]:09.4f}_{size[0]}_{size[1]}{image_ext}"

    return output_filename

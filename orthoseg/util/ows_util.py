# -*- coding: utf-8 -*-
"""
Module with generic usable utility functions to make some tasks using OWS services easier.
"""

import concurrent.futures as futures
import datetime
import itertools as it
import logging
import math
from pathlib import Path
import random
import time
from typing import List, Optional, Tuple, Union
import numpy as np

import owslib
import owslib.wms
import owslib.util
import pycron
import pyproj
import rasterio as rio
from rasterio import plot as rio_plot
from rasterio import profiles as rio_profiles
from rasterio import transform as rio_transform
from rasterio import windows as rio_windows
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
        layersources: List[dict],
        crs: pyproj.CRS,
        output_image_dir: Path,
        image_gen_bounds: Tuple[float, float, float, float] = None,
        image_gen_roi_filepath: Optional[Path] = None,
        grid_xmin: float = 0.0,
        grid_ymin: float = 0.0,
        image_crs_pixel_x_size: float = 0.25,
        image_crs_pixel_y_size: float = 0.25,
        image_pixel_width: int = 1024,
        image_pixel_height: int = 1024,
        image_pixels_ignore_border: int = 0,
        nb_concurrent_calls: int = 1,
        random_sleep: float = 0.0,
        cron_schedule: str = None,     
        image_format: str = FORMAT_GEOTIFF,
        image_format_save: str = None,
        tiff_compress: str = 'lzw',
        transparent: bool = False,
        pixels_overlap: int = 0,
        column_start: int = 0,
        nb_images_to_skip: int = None,
        max_nb_images: int = -1,
        force: bool = False):
    
    ##### Init #####
    if image_format_save is None:
        image_format_save = image_format

    crs_width = math.fabs(image_pixel_width*image_crs_pixel_x_size)   # tile width in units of crs => 500 m
    crs_height = math.fabs(image_pixel_height*image_crs_pixel_y_size) # tile height in units of crs => 500 m
    if cron_schedule is not None:
        logger.info(f"A cron_schedule was specified, so the download will only proceed in the specified time range: {cron_schedule}")

    # Read the region of interest file if provided
    roi_gdf = None
    if image_gen_roi_filepath is not None:
        # Read roi
        logger.info(f"Read region of interest from {image_gen_roi_filepath}")
        roi_gdf = gpd.read_file(str(image_gen_roi_filepath))
                
        # If the generate_window wasn't specified, calculate the bounds
        # based on the roi (but make sure they fit the grid!)
        if not image_gen_bounds:
            roi_bounds = roi_gdf.geometry.total_bounds
            image_gen_bounds = (roi_bounds[0]-((roi_bounds[0]-grid_xmin)%crs_width),
                                roi_bounds[1]-((roi_bounds[1]-grid_ymin)%crs_height),
                                roi_bounds[2]+(grid_xmin-((roi_bounds[2]-grid_xmin)%crs_width)),
                                roi_bounds[3]+(grid_ymin-((roi_bounds[3]-grid_ymin)%crs_height)))
            logger.debug(f"roi_bounds: {roi_bounds}, image_gen_bounds: {image_gen_bounds}")
        
        # If there are large objects in the roi, segment them to speed up
        # TODO: support creating a grid in latlon????
        if crs.is_projected:
            
            # Create grid
            grid = vector_util.create_grid(
                    xmin=image_gen_bounds[0], ymin=image_gen_bounds[1], 
                    xmax=image_gen_bounds[2], ymax=image_gen_bounds[3],
                    cell_width=4000, cell_height=4000)

            # Create intersection layer between grid and roi
            #roi_gdf = roi_gdf.geometry.intersection(grid.geometry)
            roi_gdf = gpd.overlay(grid, roi_gdf, how='intersection')
            
            # Explode possible multipolygons to polygons
            roi_gdf.reset_index(drop=True, inplace=True)
            # assert to evade pyLance warning
            assert isinstance(roi_gdf, gpd.GeoDataFrame)
            roi_gdf = roi_gdf.explode()
            roi_gdf.reset_index(drop=True, inplace=True)
            #roi_gdf.to_file("X:\\Monitoring\\OrthoSeg\\roi_gridded.gpkg")
            
    # Check if the image_gen_bounds are compatible with the grid...
    if (image_gen_bounds[0]-grid_xmin)%crs_width != 0:
        xmin_new = image_gen_bounds[0] - ((image_gen_bounds[0]-grid_xmin)%crs_width)
        logger.warning(f"xmin {image_gen_bounds[0]} incompatible with grid, {xmin_new} will be used")
        image_gen_bounds = (xmin_new, image_gen_bounds[1], image_gen_bounds[2], image_gen_bounds[3])
    if (image_gen_bounds[1]-grid_ymin)%crs_height != 0:
        ymin_new = image_gen_bounds[1] - ((image_gen_bounds[1]-grid_ymin)%crs_height)
        logger.warning(f"ymin {image_gen_bounds[1]} incompatible with grid, {ymin_new} will be used")
        image_gen_bounds = (image_gen_bounds[0], ymin_new, image_gen_bounds[2], image_gen_bounds[3])
    if (image_gen_bounds[2]-grid_xmin)%crs_width != 0:
        xmax_new = image_gen_bounds[2] + crs_width - ((image_gen_bounds[2]-grid_xmin)%crs_width)
        logger.warning(f"xmax {image_gen_bounds[2]} incompatible with grid, {xmax_new} will be used")
        image_gen_bounds = (image_gen_bounds[0], image_gen_bounds[1], xmax_new, image_gen_bounds[3])
    if (image_gen_bounds[3]-grid_ymin)%crs_height != 0:
        ymax_new = image_gen_bounds[3] + crs_height - ((image_gen_bounds[3]-grid_ymin)%crs_height)
        logger.warning(f"ymax {image_gen_bounds[3]} incompatible with grid, {ymax_new} will be used")
        image_gen_bounds = (image_gen_bounds[0], image_gen_bounds[1], image_gen_bounds[2], ymax_new)

    # Calculate width and height...
    dx = math.fabs(image_gen_bounds[0] - image_gen_bounds[2]) # area width in units of crs
    dy = math.fabs(image_gen_bounds[1] - image_gen_bounds[3]) # area height in units of crs
    cols = int(math.ceil(dx / crs_width)) + 1
    rows = int(math.ceil(dy / crs_height)) + 1

    # Inits to start getting images 
    if not output_image_dir.exists():
        output_image_dir.mkdir(parents=True)
    
    layersources_prepared = []
    for layersource in layersources:
        wms_service = owslib.wms.WebMapService(
                url=layersource['wms_server_url'], 
                version=layersource['wms_version'])
        layersources_prepared.append(
                LayerSource(
                        wms_service=wms_service,
                        layernames=layersource['layernames'],
                        layerstyles=layersource['layerstyles'],
                        bands=layersource['bands'],
                        random_sleep=layersource['random_sleep']))
                                
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
        output_dir_list = []
        for col in range(column_start, cols):
            
            image_xmin = col * crs_width + image_gen_bounds[0]
            image_xmax = (col + 1) * crs_width + image_gen_bounds[0]
    
            # If overlapping images are wanted... increase image bbox
            if pixels_overlap:
                image_xmin = image_xmin-(pixels_overlap*image_crs_pixel_x_size)
                image_xmax = image_xmax+(pixels_overlap*image_crs_pixel_x_size)
                
            # Put all the images of this column in a dir
            if crs.is_projected:
                output_dir = output_image_dir / f"{image_xmin:06.0f}"
            else:
                output_dir = output_image_dir / f"{image_xmin:09.4f}"
            if not output_dir.exists():
                output_dir.mkdir()
                
            logger.debug(f"load_images to {output_image_dir.parent.name}/{output_image_dir.name}, column {col} ({output_dir.name})")
            for row in range(0, rows):

                # If a cron_schedule is specified, check if we should be running
                if cron_schedule is not None:
                    # Sleep till the schedule becomes active
                    first_time = True
                    while not pycron.is_now(cron_schedule):
                        # The first time, log message that we are going to sleep...
                        if first_time is True:
                            # Send a newline to the output, because the progress messages don't do newlines 
                            print()
                            logger.info(f"There is a time schedule specified, and we need to sleep: {cron_schedule}")
                            first_time = False
                        time.sleep(60)

                nb_processed += 1

                # Calculate y bounds
                image_ymin = row * crs_height + image_gen_bounds[1]
                image_ymax = (row + 1) * crs_height + image_gen_bounds[1]
    
                # If overlapping images are wanted... increase image bbox
                if pixels_overlap:
                    image_ymin = image_ymin-(pixels_overlap*image_crs_pixel_y_size)
                    image_ymax = image_ymax+(pixels_overlap*image_crs_pixel_y_size)
    
                # Create output filename
                output_filename = create_filename(
                        crs=crs,
                        bbox=(image_xmin, image_ymin, image_xmax, image_ymax),
                        size=(image_pixel_width+2*pixels_overlap, 
                            image_pixel_height+2*pixels_overlap),
                        image_format=image_format,
                        layername=None)
                output_filepath = output_dir / output_filename

                # Do some checks to know if the image needs to be downloaded 
                image_to_be_skipped = False
                if(nb_images_to_skip
                   and (nb_processed%nb_images_to_skip) != 0):
                    # If we need to skip images, do so...
                    nb_ignore_in_progress += 1
                    image_to_be_skipped = True
                elif not force and output_filepath.exists() and output_filepath.stat().st_size > 0:
                    # Image exists already
                    nb_ignore_in_progress += 1
                    image_to_be_skipped = True
                    logger.debug("    -> image exists already, so skip")
                elif roi_gdf is not None:
                    # If roi was provided, check first if the current image overlaps
                    image_shape = sh_geom.box(image_xmin, image_ymin, image_xmax, image_ymax)
                    
                    spatial_index = roi_gdf.sindex
                    possible_matches_index = list(spatial_index.intersection(image_shape.bounds))
                    possible_matches = roi_gdf.loc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(image_shape)]
                    
                    if len(precise_matches) == 0:
                        nb_ignore_in_progress += 1
                        logger.debug("    -> image doesn't overlap with roi, so skip")
                        image_to_be_skipped = True                                      

                # If the image doesn't need to be skipped... append the image 
                # info to the batch arrays so they can be treated in 
                # bulk if the batch size is reached
                if image_to_be_skipped is False:
                    bbox_list.append((image_xmin, image_ymin, image_xmax, image_ymax))
                    size_list.append((image_pixel_width+2*pixels_overlap, 
                                     image_pixel_height+2*pixels_overlap))
                    output_filename_list.append(output_filename)
                    output_dir_list.append(output_dir)

                # If the batch size is reached or we are at the last images
                nb_images_in_batch = len(bbox_list)
                if(nb_images_in_batch == nb_concurrent_calls or nb_processed == (nb_todo-1)):
    
                    # Now we are getting to start fetching images... init start_time 
                    if start_time is None:
                        start_time = datetime.datetime.now()
                    start_time_batch_read = datetime.datetime.now()
                    
                    # Exec in parallel 
                    read_results = pool.map(
                            getmap_to_file,        # Function 
                            it.repeat(layersources_prepared),
                            output_dir_list,
                            it.repeat(crs),
                            bbox_list,
                            size_list,
                            it.repeat(image_format),
                            it.repeat(image_format_save),
                            output_filename_list,
                            it.repeat(transparent),
                            it.repeat(tiff_compress),
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
                        print(f"\rload_images to {output_image_dir.parent.name}/{output_image_dir.name}, {hours_to_go:3d}:{min_to_go:2d} left for {nb_todo-nb_processed} images at {nb_per_hour:0.0f}/h ({nb_per_hour_lastbatch:0.0f}/h last batch), with {nb_ignore_in_progress} skipped",
                              end='', flush=True)
                    
                    # Reset variable for next batch
                    bbox_list = []
                    size_list = []
                    output_filename_list = []
                    output_dir_list = []

                    if max_nb_images > -1 and nb_downloaded >= max_nb_images:
                        return            

class LayerSource:
    def __init__(self, 
            wms_service: Union[owslib.wms.wms111.WebMapService_1_1_1, owslib.wms.wms130.WebMapService_1_3_0],
            layernames: List[str],
            layerstyles: List[str] = None,
            bands: List[int] = None,
            random_sleep: int = 0):
        self.wms_service = wms_service
        self.layernames = layernames
        self.layerstyles = layerstyles
        self.bands = bands
        self.random_sleep = random_sleep

def getmap_to_file(
        layersources: List[LayerSource],
        output_dir: Path,
        crs: Union[str, pyproj.CRS],
        bbox,
        size,
        image_format: str = FORMAT_GEOTIFF,
        image_format_save: str = None,
        output_filename: str = None,
        transparent: bool = False,
        tiff_compress: str = 'lzw',
        image_pixels_ignore_border: int = 0,
        force: bool = False,
        layername_in_filename: bool = False) -> Optional[Path]:
    """

    Args
        random_sleep: sleep a random time between 0 and this amount of seconds
                      between requests tot the WMS server
    """
    ##### Init #####
    # If no seperate save format is specified, use the standard image_format
    if image_format_save is None:
        image_format_save = image_format

    # If crs is specified as str, convert to CRS
    if isinstance(crs, str):
        crs = pyproj.CRS(crs)

    # If there isn't a filename supplied, create one...
    if output_filename is None:
        layername = None
        if layername_in_filename:
            for layersource in layersources:
                if layername is None:
                    layername = '_'.join(layersource.layernames)
                else:
                    layername += f"_{'_'.join(layersource.layernames)}"

        output_filename = create_filename(
                crs=crs, 
                bbox=bbox, size=size, 
                image_format=image_format_save,
                layername=layername)

    # Create full output filepath
    output_filepath = output_dir / output_filename

    # If force is false and file exists already, stop...
    if force is False and output_filepath.exists(): 
        if output_filepath.stat().st_size > 0:
            logger.debug(f"File already exists, skip: {output_filepath}")
            return None
        else:
            output_filepath.unlink()

    logger.debug(f"Get image to {output_filepath}")

    if not output_dir.exists():
        output_dir.mkdir()

    ##### Get image(s), read the band to keep and save #####
    image_data_output = None
    image_profile_output = None
    response = None
    for layersource in layersources:
        # Get image from server, and retry up to 10 times...
        nb_retries = 0
        time_sleep = 0
        image_retrieved = False
        while image_retrieved is False:
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
                if crs.to_epsg() == 3059:
                    bbox_for_getmap = (bbox_for_getmap[1], bbox_for_getmap[0], 
                                    bbox_for_getmap[3], bbox_for_getmap[2])

                response = layersource.wms_service.getmap(
                        layers=layersource.layernames,
                        styles=layersource.layerstyles,
                        srs=f"epsg:{crs.to_epsg()}",
                        bbox=bbox_for_getmap,
                        size=size_for_getmap,
                        format=image_format,
                        transparent=transparent)
                logger.debug(f"Finished doing request {response.geturl()}")
                
                # If a random sleep was specified... apply it
                if layersource.random_sleep > 0:
                    time.sleep(random.uniform(0, layersource.random_sleep))
                
                # Image was retrieved... so stop loop
                image_retrieved = True
            except owslib.util.ServiceException as ex:
                raise Exception(f"WMS Service gave an exception: {ex}") from ex
            except Exception as ex:
                # Retry 10 times... and increase sleep time every time
                if nb_retries < 10:
                    nb_retries += 1
                    time_sleep += 5                
                    time.sleep(time_sleep)
                    continue
                else:
                    message = f"Retried 10 times and didn't work, with layers: {layersource.layernames}, styles: {layersource.layernames}"
                    logger.exception(message)
                    raise Exception(message) from ex

        # Write image to temp file...
        # If all bands need to be kept, just save to output
        with rio.MemoryFile(response.read()) as memfile:
            with memfile.open() as image_ds:
                image_profile_curr = image_ds.profile

                # Read the data we need from the memoryfile 
                if layersource.bands is None:
                    # If no specific bands specified, read them all...
                    if image_data_output is None:
                        image_data_output = image_ds.read()
                    else:
                        image_data_output = np.append(image_data_output, image_ds.read(), axis=0)
                elif len(layersource.bands) == 1 and layersource.bands[0] == -1:
                    # If 1 band, -1 specified: dirty hack to use greyscale version of rgb image
                    image_data_tmp = image_ds.read()
                    image_data_grey = np.mean(image_data_tmp, axis=0).astype(image_data_tmp.dtype)
                    new_shape = (1, image_data_grey.shape[0], image_data_grey.shape[1])
                    image_data_grey = np.reshape(image_data_grey, new_shape)
                    if image_data_output is None:
                        image_data_output = image_data_grey
                    else:
                        image_data_output = np.append(image_data_output, image_data_grey, axis=0)
                else:
                    # If bands specified, only read the bands to keep...
                    for band in layersource.bands:
                        # Read the band needed + reshape
                        # Remark: rasterio uses 1-based indexing instead of 0-based!!! 
                        image_data_curr = image_ds.read(band+1)
                        new_shape = (1, image_data_curr.shape[0], image_data_curr.shape[1])
                        image_data_curr = np.reshape(image_data_curr, new_shape)

                        # Set or append to image_data_output
                        if image_data_output is None:
                            image_data_output = image_data_curr
                        else:
                            image_data_output = np.append(image_data_output, image_data_curr, axis=0)

                # Set output profile (number of bands will be corrected later on if needed) 
                if image_profile_output is None:
                    image_profile_output = image_profile_curr

    # Write output file
    # evade pyLance warning
    assert image_profile_output is not None
    assert isinstance(image_data_output, np.ndarray)

    # Set the number of bands to write correctly...
    if(image_format_save in [FORMAT_JPEG, FORMAT_PNG]
       and image_data_output.shape[0] == 2):
        zero_band = np.zeros(
            shape=(1, image_data_output.shape[1], image_data_output.shape[2]), 
            dtype=image_data_output.dtype)
        image_data_output = np.append(image_data_output, zero_band, axis=0)

    assert isinstance(image_data_output, np.ndarray)
    image_profile_output.update(count=image_data_output.shape[0])
    image_profile_output = get_cleaned_write_profile(image_profile_output)
    with rio.open(str(output_filepath), 'w', **image_profile_output) as image_file:
        image_file.write(image_data_output)
    
    # If an aux.xml file was written, remove it again...
    output_aux_path = output_filepath.parent / f"{output_filepath.name}.aux.xml"
    if output_aux_path.exists() is True:
        output_aux_path.unlink()

    ##### Make the output image compliant with image_format_save #####

    # If geotiff is asked, check if the the coordinates are embedded...
    if image_format_save == FORMAT_GEOTIFF:
        # Read output image to check if coördinates are there
        with rio.open(str(output_filepath)) as image_ds:
            image_profile_orig = image_ds.profile
            image_transform_affine = image_ds.transform

            if image_pixels_ignore_border == 0:
                image_data_output = image_ds.read()
            else:
                image_data_output = image_ds.read(window=rio_windows(
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
                image_profile_gtiff = rio_profiles.DefaultGTiffProfile.defaults

                # Copy appropriate info from source file
                image_profile_gtiff.update(
                        count=image_profile_orig['count'], width=size[0], height=size[1],
                        nodata=image_profile_orig['nodata'], dtype=image_profile_orig['dtype'])
                image_profile = image_profile_gtiff
            else:
                image_profile = image_profile_orig
                
            # Set the asked compression
            image_profile.update(compress=tiff_compress)

            # For some coordinate systems apparently the axis ordered is configured wrong in LibOWS :-(
            crs_pixel_x_size = (bbox[2]-bbox[0])/size[0]
            crs_pixel_y_size = (bbox[1]-bbox[3])/size[1]

            logger.debug(f"Coordinates to put in geotiff:\n" +
                         f"    - x-component of the pixel width, W-E: {crs_pixel_x_size}\n" +
                         f"    - y-component of the pixel width, W-E (0 if image is exactly N up): 0\n" +
                         f"    - top-left x: {bbox[0]}\n" +
                         f"    - x-component of the pixel height, N-S (0 if image is exactly N up): \n" +
                         f"    - y-component of the pixel height, N-S: {crs_pixel_y_size}\n" +
                         f"    - top-left y: {bbox[3]}")

            # Add transform and crs to the profile
            image_profile.update(
                    transform=rio_transform.Affine(
                            crs_pixel_x_size, 0, bbox[0],
                            0 , crs_pixel_y_size, bbox[3]),
                    crs=crs)

            # Delete output file, and write again
            output_filepath.unlink()
            with rio.open(str(output_filepath), 'w', **image_profile) as image_file:
                image_file.write(image_data_output)

    else:
        # For file formats that doesn't support coordinates, we add a worldfile       
        crs_pixel_x_size = (bbox[2]-bbox[0])/size[0]
        crs_pixel_y_size = (bbox[1]-bbox[3])/size[1]

        path_noext = output_filepath.parent / output_filepath.stem
        ext_world = get_world_ext_for_image_format(image_format_save)
        output_worldfile_filepath = Path(str(path_noext) + ext_world)
        
        with output_worldfile_filepath.open('w') as wld_file:
            wld_file.write(f"{crs_pixel_x_size}")
            wld_file.write("\n0.000")
            wld_file.write("\n0.000")
            wld_file.write(f"\n{crs_pixel_y_size}")
            wld_file.write(f"\n{bbox[0]}")
            wld_file.write(f"\n{bbox[3]}")
        
        # If the image format to save is different, or if a border needs to be ignored
        if(image_format != image_format_save 
           or image_pixels_ignore_border > 0):

            # Read image 
            with rio.open(str(output_filepath)) as image_ds:
                image_profile_orig = image_ds.profile
                image_transform_affine = image_ds.transform

                # If border needs to be ignored, only read data we are interested in
                if image_pixels_ignore_border == 0:
                    image_data_output = image_ds.read()
                else:
                    image_data_output = image_ds.read(window=rio_windows(
                            image_pixels_ignore_border, image_pixels_ignore_border, 
                            size[0], size[1]))
            
            # If same save format, reuse profile
            if image_format == image_format_save:
                image_profile_curr = image_profile_orig
                if image_pixels_ignore_border != 0:
                    image_profile_curr.update(width=size[0], height=size[1])
            else:
                if image_format_save == FORMAT_TIFF:
                    driver = 'GTiff'
                    compress=tiff_compress
                else:
                    raise Exception(f"Unsupported image_format_save: {image_format_save}")
                image_profile_curr = rio_profiles.Profile(
                        width=size[0], height=size[1], count=image_profile_orig['count'],
                        nodata=image_profile_orig['nodata'], dtype=image_profile_orig['dtype'],
                        compress=compress, driver=driver)

            # Delete output file, and write again
            output_filepath.unlink()
            image_profile_curr = get_cleaned_write_profile(image_profile_curr)
            with rio.open(str(output_filepath), 'w', **image_profile_curr) as image_file:
                image_file.write(image_data_output)               

            #raise Exception(f"Different save format not supported between {image_format} and {image_format_save}")

    return output_filepath

def create_filename(
        crs: pyproj.CRS,
        bbox,
        size,
        image_format: str,
        layername: str = None):
    
    # Get image extension based on format
    image_ext = get_ext_for_image_format(image_format)

    # Use different file names for projected vs geographic crs
    if crs.is_projected:
        output_filename = f"{bbox[0]:06.0f}_{bbox[1]:06.0f}_{bbox[2]:06.0f}_{bbox[3]:06.0f}_{size[0]}_{size[1]}"
    else:
        output_filename = f"{bbox[0]:09.4f}_{bbox[1]:09.4f}_{bbox[2]:09.4f}_{bbox[3]:09.4f}_{size[0]}_{size[1]}"

    # Add layername if it is not None    
    if layername is not None:
        output_filename += "_" + layername
    
    # Add extension
    output_filename += image_ext

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

def get_cleaned_write_profile(profile: rio_profiles.Profile) -> Union[dict, rio_profiles.Profile]:

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

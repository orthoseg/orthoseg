# -*- coding: utf-8 -*-
"""
Helper module to support some tasks regarding using OWS services.

@author: Pieter Roggemans
"""

import logging
import os
import time
import random
import math

from owslib.wms import WebMapService
import rasterio as rio
import fiona
import shapely.geometry as sh_geom

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
                        wms_server_layers: [str],
                        srs: str,
                        output_image_dir: str,
                        image_gen_bounds: (float, float, float, float) = None,
                        image_gen_roi_filepath: str = None,                         
                        grid_xmin: float = 0.0,
                        grid_ymin: float = 0.0,
                        image_srs_pixel_x_size: int = 0.25,
                        image_srs_pixel_y_size: int = 0.25,
                        image_pixel_width: int = 1000,
                        image_pixel_height: int = 1000,                                 
                        format: str = FORMAT_GEOTIFF,
                        tiff_compress: str = 'lzw',
                        transparent: str = False,
                        wms_server_layers_styles: [str] = ['default'],
                        force: bool = False):
    
    srs_width = math.fabs(image_pixel_width*image_srs_pixel_x_size)   # tile width in units of crs => 500 m
    srs_height = math.fabs(image_pixel_height*image_srs_pixel_y_size) # tile height in units of crs => 500 m

    # Read the 
    roi_rows = None    
    if image_gen_roi_filepath:
        # Open vector layer
        logger.info(f"Open vector file {image_gen_roi_filepath}")
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
        
        # If the generate_window wasn't specified, calculate the bounds
        # based on the roi (but make sure they fit the grid!)
        if not image_gen_bounds:
            image_gen_bounds = (roi_data.bounds[0]-((roi_data.bounds[0]-grid_xmin)%srs_width),
                                roi_data.bounds[1]-((roi_data.bounds[1]-grid_ymin)%srs_height),
                                roi_data.bounds[2]+(grid_xmin-((roi_data.bounds[2]-grid_xmin)%srs_width)),
                                roi_data.bounds[3]+(grid_ymin-((roi_data.bounds[3]-grid_ymin)%srs_height)))
            logger.info(f"roi_data.bounds: {roi_data.bounds}, image_gen_bounds: {image_gen_bounds}")

    # Check if the image_gen_bounds are compatible with the grid...
    error_message = None
    if((image_gen_bounds[0]-grid_xmin)%srs_width != 0):
        error_message += f"image_gen_bounds[0] (xmin) is not compatible with grid!\n"
    elif((image_gen_bounds[2]-grid_xmin)%srs_width != 0):
        error_message += f"image_gen_bounds[2] (xmax) is not compatible with grid!\n"
    elif((image_gen_bounds[1]-grid_ymin)%srs_height != 0):
        error_message += f"image_gen_bounds[1] (ymin) is not compatible with grid!\n"
    elif((image_gen_bounds[3]-grid_ymin)%srs_height != 0):
        error_message += f"image_gen_bounds[3] (ymax)is not compatible with grid!\n"
    
    # If there was an error, stop!
    if error_message:
        logger.critical(error_message)
        raise Exception(error_message)

    dx = math.fabs(image_gen_bounds[0] - image_gen_bounds[2]) # area width in units of crs
    dy = math.fabs(image_gen_bounds[1] - image_gen_bounds[3]) # area height in units of crs
    
    cols = int(math.ceil(dx / srs_width)) + 1
    rows = int(math.ceil(dy / srs_height)) + 1
    
    print(f"Number rows: {rows}, number columns: {cols}")
    
    is_srs_projected = rio.crs.CRS.from_string(srs).is_projected
    counter = 0
    
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)
    
    wms = WebMapService(wms_server_url, version='1.3.0')
    
    # Loop through all columns and get the images...
    counter = 0
    for col in range(0, cols):
        image_xmin = col * srs_width + image_gen_bounds[0]
        image_xmax = (col + 1) * srs_width + image_gen_bounds[0]
        
        # Put all the images of this column in a dir
        if is_srs_projected:
            output_dir = os.path.join(output_image_dir, f"{image_xmin:06.0f}")
        else:
            output_dir = os.path.join(output_image_dir, f"{image_xmin:09.4f}")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    
        for row in range(0, rows):
            counter += 1
            logger.info(f"Process image {counter} out of {cols*rows}: {counter/(cols*rows):.2f} %")
            image_ymin = row * srs_height + image_gen_bounds[1]
            image_ymax = (row + 1) * srs_height + image_gen_bounds[1]
            
            # If roi was provided, check first if the current image overlaps
            if roi_rows:
                image_shape = sh_geom.box(image_xmin, image_ymin, image_xmax, image_ymax)
                intersects = False
                # Loop through the roi rows. If one of them overlaps -> OK.
                for roi_row in roi_rows:
                    # Check first on bounds to improve performance
                    if(roi_row['geom_bounds'].intersection(image_shape) 
                       and roi_row['geom'].intersection(image_shape)):
                        intersects = True
                        break 
                if not intersects:
                    logger.info("    -> image doesn't overlap with roi, so skip")
                    continue
    
            # Now really get the image
            getmap_to_file(wms=wms,
                    layers=wms_server_layers,
                    output_dir=output_dir,
                    srs=srs,
                    bbox=(image_xmin, image_ymin, image_xmax, image_ymax),
                    size=(image_pixel_width, image_pixel_height),
                    format=format,
                    tiff_compress=tiff_compress,
                    transparent=transparent)
        
def getmap_to_file(wms: WebMapService,
                   layers: [str],
                   output_dir: str,
                   srs: str,
                   bbox,
                   size,
                   format: str = FORMAT_GEOTIFF,
                   output_filename: str = None,
                   transparent: bool = False,
                   tiff_compress: str = 'lzw',
                   layers_styles: [str] = ['default'],
                   random_sleep: float = 2.0,
                   force: bool = False) -> str:
    """
    
    Args
        random_sleep: sleep a random time between 0 and this amount of seconds 
                      between requests tot the WMS server
    """
    # If there isn't a filename supplied, create one...
    if output_filename is None:
        # Choose image extension based on format    
        if format == FORMAT_GEOTIFF:
            image_ext = FORMAT_GEOTIFF_EXT
        elif format == FORMAT_TIFF:
            image_ext = FORMAT_TIFF_EXT
        elif format == FORMAT_JPEG:
            image_ext = FORMAT_JPEG_EXT
        elif format == FORMAT_PNG:
            image_ext = FORMAT_PNG_EXT
        else:
            raise Exception("Notimplemented exception!")
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
        logger.info(f"File already exists, skip: {output_filepath}")
        return output_filepath
    
    # Retry 10 times...
    nb_retries = 0    
    while nb_retries <= 10:
        try:
            img = wms.getmap(layers=layers,
                             styles=layers_styles,
                             srs=srs,
                             bbox=bbox,
                             size=size,
                             format=format,
                             transparent=transparent)
            if random_sleep:
                time.sleep(random.uniform(0, random_sleep))
            break
            
        except OSError as ex:
            if nb_retries >= 10:
                raise
        
            nb_retries += 1
            if "[WinError 10048]" in str(ex):
                logger.info(f"Exception while trying calculate_sentinel_timeseries, retry! (Full exception message {ex})")
                time.sleep(10)
                continue
            else:
                logger.critical(f"Exception [WinError {ex.winerror}] unknown, so don't retry (Full exception message {ex})")
                raise
    
    # Write image to file...    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_filepath, 'wb') as image_file:
        image_file.write(img.read())

    # If geotiff is asked, check if the the coordinates are embedded...            
    if format == FORMAT_GEOTIFF:
        # Read output image to check if coördinates are there
        with rio.open(output_filepath) as image_ds:
            image_profile_orig = image_ds.profile
            image_transform_affine = image_ds.affine
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
                        driver=rio.profiles.DefaultGTiffProfile.driver,
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
    
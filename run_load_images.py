# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:30:48 2018

@author: pierog
"""

import os

import log_helper
import ows_helper

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Log dir
log_dir = "c:\\errorlog\\python"

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():

    # TODO: change script so it uses ini files!
    # TODO: change script in general: is spaghetti!
    
    # Main initialisation of the logging
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info("Start loading images")

    # Summer pictures
    wms_server_url_summer = "http://geoservices.informatievlaanderen.be/raadpleegdiensten/OMZ/wms?"
    wms_layernames_summer = ['OMZRGB15VL']
    
    # Winter pixtures
    wms_server_url_winter = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'
    wms_layernames_winter = ['ofw']    
    
    image_base_dir = "X:\\Monitoring\\OrthoSeg\\_input_images"
    
    # Set to True to load random test images...
    load_random_test_images = False
#    load_random_test_images = False
    if load_random_test_images:
        image_pixel_width = 512
        image_pixel_height = image_pixel_width
        image_format = ows_helper.FORMAT_JPEG
        pixels_overlap = 0
        
        image_season = 'zomer'
        image_year = 2015
        
        column_start = 1
        nb_images_to_skip = 50
        
    else:
        image_pixel_width = 1024
        image_pixel_height = image_pixel_width
        image_format = ows_helper.FORMAT_JPEG
        pixels_overlap = 128
        
        image_season = 'winter'
        image_year = 2018

        column_start = 0
        nb_images_to_skip = 0

    # Get the wms server url and layername
    if image_year == 2015:
        if image_season == 'zomer':
            wms_server_url = wms_server_url_summer
            wms_layernames = wms_layernames_summer
        else:
            raise Exception("Unsupported season-year combination: {image_season}-{image_year}")
    elif image_year == 2018:
        if image_season == 'winter':
            wms_server_url = wms_server_url_winter
            wms_layernames = wms_layernames_winter
        else:
            raise Exception("Unsupported season-year combination: {image_season}-{image_year}")
        
    # Prepare the image_subdir
    image_subdir = f"Ortho_{image_season}_{image_year}"
    if load_random_test_images:
        image_subdir += "_testsample"
    
    image_subdir = os.path.join(image_subdir, f"{image_pixel_width}x{image_pixel_height}")
    if pixels_overlap and pixels_overlap > 0:
        image_subdir += f"_{pixels_overlap}pxOverlap"
        
    image_dir = os.path.join(image_base_dir, image_subdir)
    
    # SRS...
    srs = "EPSG:31370"

    roi_filepath = "X:\\GIS\\GIS DATA\\Gewesten\\2005\\Vlaanderen+Brussel\\gew_VLenBR.shp"

    '''
    generate_window_xmin = 22000
    generate_window_ymin = 150000
    generate_window_xmax = 259000
    generate_window_ymax = 245000
    '''

    # til column 612 is done...
    ows_helper.get_images_for_grid(
            wms_server_url=wms_server_url,
            wms_layernames=wms_layernames,
            srs=srs,
            output_image_dir=image_dir,
            image_gen_roi_filepath=roi_filepath,
            image_srs_pixel_x_size=0.25,
            image_srs_pixel_y_size=0.25,
            image_pixel_width=image_pixel_width,
            image_pixel_height=image_pixel_height,
            nb_concurrent_calls=6,
            image_format=image_format,
            pixels_overlap=pixels_overlap,
            random_sleep=0.0,
            column_start=column_start,
            nb_images_to_skip=nb_images_to_skip)

if __name__ == '__main__':
    main()

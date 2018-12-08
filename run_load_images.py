# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:30:48 2018

@author: pierog
"""

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

    # Main initialisation of the logging
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info("Start loading images")
    
    # Set to True to load random test images...
    load_random_test_images = False
#    load_random_test_images = False
    if load_random_test_images:
        image_pixel_width = 512
        image_pixel_height = image_pixel_width
        pixels_overlap = 0
        nb_images_to_skip = 50
        image_dir = f"X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\greenhouses\\test_random\\image{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"
    else:
        image_pixel_width = 1024
        image_pixel_height = image_pixel_width
        pixels_overlap = 64
        nb_images_to_skip = 0
        image_dir = f"X:\\GIS\\GIS DATA\\_Tmp\\Ortho_2018_autosegment_cache\\{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"

    WMS_SERVER_URL = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'
    wms_server_layers = ['ofw']
    srs = "EPSG:31370"

    roi_filepath = "X:\\GIS\\GIS DATA\\Gewesten\\2005\\Vlaanderen+Brussel\\gew_VLenBR.shp"

    '''
    generate_window_xmin = 22000
    generate_window_ymin = 150000
    generate_window_xmax = 259000
    generate_window_ymax = 245000
    '''

    ows_helper.get_images_for_grid(
            wms_server_url=WMS_SERVER_URL,
            wms_server_layers=wms_server_layers,
            srs=srs,
            output_image_dir=image_dir,
            image_gen_roi_filepath=roi_filepath,
            image_srs_pixel_x_size=0.25,
            image_srs_pixel_y_size=0.25,
            image_pixel_width=image_pixel_width,
            image_pixel_height=image_pixel_height,
            format=ows_helper.FORMAT_JPEG,
            pixels_overlap=pixels_overlap,
            random_sleep=2.0,
            max_nb_images_to_download=50000,
            column_start=0,
            nb_images_to_skip=nb_images_to_skip)

if __name__ == '__main__':
    main()

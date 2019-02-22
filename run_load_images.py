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

    # TODO: change script so it uses ini files!
    
    # Main initialisation of the logging
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info("Start loading images")
    
    # Set to True to load random test images...
    load_random_test_images = True
#    load_random_test_images = False
    if load_random_test_images:
        image_pixel_width = 512
        image_pixel_height = image_pixel_width
        image_format = ows_helper.FORMAT_JPEG
        pixels_overlap = 0
#        image_dir = f"X:\\Monitoring\\OrthoSeg\\trees\\test_random\\image{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"
        #image_dir = "X:\\Monitoring\\OrthoSeg\\trees\\training\\test-random\\image"
        image_dir = f"X:\\Monitoring\\OrthoSeg\\_input_images\\Ortho_zomer_2015_testsample\\image{image_pixel_width}x{image_pixel_height}"

        column_start = 1
        nb_images_to_skip = 50
    else:
        image_pixel_width = 1024
        image_pixel_height = image_pixel_width
        image_format = ows_helper.FORMAT_GEOTIFF
        pixels_overlap = 128
        image_dir = f"X:\\Monitoring\\OrthoSeg\\_input_images\\Ortho_2018\\{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"

        column_start = 0
        nb_images_to_skip = 0

    # Winter
    wms_server_url = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'
    wms_layernames = ['ofw']
    
    '''
    # Zomer
    wms_server_url = "http://geoservices.informatievlaanderen.be/raadpleegdiensten/OMZ/wms?"
    wms_layernames = ['OMZRGB15VL']
    '''
    
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
            image_format=image_format,
            pixels_overlap=pixels_overlap,
            random_sleep=0.0,
            column_start=column_start,
            nb_images_to_skip=nb_images_to_skip)

if __name__ == '__main__':
    main()

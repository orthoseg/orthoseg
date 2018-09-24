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

    image_dir = "\\\\dg3.be\\alp\\Datagis\\Ortho_AGIV_2018_ofw"
    image_dir = "X:\\GIS\\GIS DATA\\_Tmp\\Ortho_2018_autosegment_cache\\1024x1024_50pxOverlap"

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
            image_pixel_width=1024,
            image_pixel_height=1024,
            format=ows_helper.FORMAT_JPEG,
            pixels_overlap=50,
            random_sleep=2.0,
            max_nb_images=20000)

if __name__ == '__main__':
    main()

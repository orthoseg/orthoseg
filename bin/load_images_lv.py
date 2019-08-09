# -*- coding: utf-8 -*-
"""
Script to load images from a WMS server.
"""

import os
# TODO: the init of this doensn't seem to work properly... should be solved somewhere else?
os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"
    
import logging

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, '.')

import orthoseg.helpers.config as conf
import orthoseg.helpers.log as log_helper
import orthoseg.helpers.ows as ows_helper

def load_images(load_testsample_images: bool = False):

    # Read the configuration
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    segment_config_filepaths=[os.path.join(scriptdir, 'general.ini'), 
                              os.path.join(scriptdir, 'load_images_lv.ini'),
                              os.path.join(scriptdir, 'local_overrule.ini')]
    conf.read_config(segment_config_filepaths)
    
    # Main initialisation of the logging
    logger = log_helper.main_log_init(conf.dirs['log_dir'], __name__)      
    logger.info("Start loading images")
    logger.info(f"Config used: \n{conf.pformat_config()}")
    logger.setLevel(logging.DEBUG)

    # Specify window to generate
    generate_window_xmin = None

    generate_window_xmin = 529920
    generate_window_ymin = 259840
    generate_window_xmax = 599808
    generate_window_ymax = 332800
         
    # Use different setting depending if testsample or all images
    if load_testsample_images:
        output_image_dir=conf.dirs['predictsample_image_dir']

        # Use the same image size as for the training, that is the most 
        # convenient to check the quality
        image_pixel_width = int(conf.train['image_pixel_width'])
        image_pixel_height = int(conf.train['image_pixel_height'])
        image_pixel_x_size = float(conf.train['image_pixel_x_size'])
        image_pixel_y_size = float(conf.train['image_pixel_y_size'])
        image_pixels_overlap = 0
        image_format = ows_helper.FORMAT_JPEG
        
        # To create the testsample, fetch only on every ... images
        column_start = 1
        nb_images_to_skip = 50
        
    else:
        output_image_dir=conf.dirs['predict_image_dir']
        
        # Get the image size for the predict
        image_pixel_width = int(conf.predict['image_pixel_width'])
        image_pixel_height = int(conf.predict['image_pixel_height'])
        image_pixel_x_size = float(conf.train['image_pixel_x_size'])
        image_pixel_y_size = float(conf.train['image_pixel_y_size'])
        image_pixels_overlap = int(conf.predict['image_pixels_overlap'])
        image_format = ows_helper.FORMAT_JPEG
        
        # For the real prediction dataset, no skipping obviously...
        column_start = 0
        nb_images_to_skip = 0
    
    # Now start loading the images
    if generate_window_xmin is not None:
        image_gen_bounds = (generate_window_xmin, generate_window_ymin, generate_window_xmax, generate_window_ymax) 
    else:
        image_gen_bounds = None

    ows_helper.get_images_for_grid(
            wms_server_url=conf.general['wms_server_url'],
            wms_layernames=conf.general['wms_layername'],
            srs=conf.general['projection'],
            output_image_dir=output_image_dir,
            image_gen_bounds=image_gen_bounds,
            image_gen_roi_filepath=conf.files['roi_filepath'],
            image_srs_pixel_x_size=image_pixel_x_size,
            image_srs_pixel_y_size=image_pixel_y_size,
            image_pixel_width=image_pixel_width,
            image_pixel_height=image_pixel_height,
            nb_concurrent_calls=6,
            image_format=image_format,
            pixels_overlap=image_pixels_overlap,
            column_start=column_start,
            nb_images_to_skip=nb_images_to_skip)

if __name__ == '__main__':
    
    load_images(load_testsample_images=False)

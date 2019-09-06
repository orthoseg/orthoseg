# -*- coding: utf-8 -*-
"""
Script to load images from a WMS server.
"""

import os
# TODO: the init of this doensn't seem to work properly... should be solved somewhere else?
os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, '.')

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper
from orthoseg.util import ows_util

def load_images(
            config_filepaths: [str],
            load_testsample_images: bool = False):

    ##### Init #####
    # Main initialisation of the logging
    conf.read_config(config_filepaths)
    logger = log_helper.main_log_init(conf.dirs['log_dir'], __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")
   
    # Use different setting depending if testsample or all images
    if load_testsample_images:
        output_image_dir=conf.dirs['predictsample_image_input_dir']

        # Use the same image size as for the training, that is the most 
        # convenient to check the quality
        image_pixel_width = conf.train.getint('image_pixel_width')
        image_pixel_height = conf.train.getint('image_pixel_height')
        image_pixel_x_size = conf.train.getfloat('image_pixel_x_size')
        image_pixel_y_size = conf.train.getfloat('image_pixel_y_size')
        image_pixels_overlap = 0
        image_format = ows_util.FORMAT_JPEG
        
        # To create the testsample, fetch only on every ... images
        column_start = 1
        nb_images_to_skip = 50
        
    else:
        output_image_dir=conf.dirs['predict_image_input_dir']
        
        # Get the image size for the predict
        image_pixel_width = conf.predict.getint('image_pixel_width')
        image_pixel_height = conf.predict.getint('image_pixel_height')
        image_pixel_x_size = conf.predict.getfloat('image_pixel_x_size')
        image_pixel_y_size = conf.predict.getfloat('image_pixel_y_size')
        image_pixels_overlap = conf.predict.getint('image_pixels_overlap')
        image_format = ows_util.FORMAT_JPEG
        
        # For the real prediction dataset, no skipping obviously...
        column_start = 0
        nb_images_to_skip = 0
    
    predict_datasource_code = conf.predict['image_datasource_code']
    wms_server_url = conf.image_datasources[predict_datasource_code]['wms_server_url']
    wms_layernames = conf.image_datasources[predict_datasource_code]['wms_layernames']
    wms_layerstyles = conf.image_datasources[predict_datasource_code]['wms_layerstyles']
    projection = conf.image_datasources[predict_datasource_code]['projection']
    bbox = conf.image_datasources[predict_datasource_code]['bbox']
    grid_xmin = conf.image_datasources[predict_datasource_code]['grid_xmin']
    grid_ymin = conf.image_datasources[predict_datasource_code]['grid_ymin']
    image_pixels_ignore_border = conf.image_datasources[predict_datasource_code]['image_pixels_ignore_border']
    ows_util.get_images_for_grid(
            wms_server_url=wms_server_url,
            wms_layernames=wms_layernames,
            wms_layerstyles=wms_layerstyles,
            srs=projection,
            output_image_dir=output_image_dir,
            image_gen_bounds=bbox,
            image_gen_roi_filepath=conf.files['roi_filepath'],
            grid_xmin=grid_xmin,
            grid_ymin=grid_ymin,
            image_srs_pixel_x_size=image_pixel_x_size,
            image_srs_pixel_y_size=image_pixel_y_size,
            image_pixel_width=image_pixel_width,
            image_pixel_height=image_pixel_height,
            image_pixels_ignore_border=image_pixels_ignore_border,
            nb_concurrent_calls=6,
            image_format=image_format,
            pixels_overlap=image_pixels_overlap,
            column_start=column_start,
            nb_images_to_skip=nb_images_to_skip)

if __name__ == '__main__':
    raise Exception("Not implemented!")

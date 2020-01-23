# -*- coding: utf-8 -*-
"""
Script to load images from a WMS server.
"""

import argparse
from pathlib import Path
import shlex
import sys

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper
from orthoseg.util import ows_util

def load_images_argstr(argstr):
    args = shlex.split(argstr)
    load_images_args(args)

def load_images_args(args):

    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument("--config_dir", type=str, required=True,
            help="The config dir to use")
    required.add_argument("--config_filename", type=str, required=True,
            help="The config file to use")
    
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    # Add back help 
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')

    # Interprete arguments
    args = parser.parse_args(args)

    ##### Run! #####
    load_images(
            config_dir=Path(args.config_dir),
            config_filename=args.config_filename)

def load_images(
        config_dir: Path,
        config_filename: str,
        load_testsample_images: bool = False):

    ##### Init #####   
    # Load config
    config_filepaths = conf.get_needed_config_files(
            config_dir=config_dir, 
            config_filename=config_filename)
    layer_config_filepath = config_dir / 'image_layers.ini'
    conf.read_config(config_filepaths, layer_config_filepath)
    
    # Main initialisation of the logging
    global logger
    logger = log_helper.main_log_init(conf.dirs.getpath('log_training_dir'), __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Use different setting depending if testsample or all images
    if load_testsample_images:
        output_image_dir=conf.dirs.getpath('predictsample_image_input_dir')

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
        output_image_dir=conf.dirs.getpath('predict_image_input_dir')
        
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
    
    predict_layer = conf.predict['image_layer']
    wms_server_url = conf.image_layers[predict_layer]['wms_server_url']
    wms_version = conf.image_layers[predict_layer]['wms_version']
    wms_layernames = conf.image_layers[predict_layer]['wms_layernames']
    wms_layerstyles = conf.image_layers[predict_layer]['wms_layerstyles']
    nb_concurrent_calls = conf.image_layers[predict_layer]['nb_concurrent_calls']
    random_sleep = conf.image_layers[predict_layer]['random_sleep']
    projection = conf.image_layers[predict_layer]['projection']
    bbox = conf.image_layers[predict_layer]['bbox']
    grid_xmin = conf.image_layers[predict_layer]['grid_xmin']
    grid_ymin = conf.image_layers[predict_layer]['grid_ymin']
    image_pixels_ignore_border = conf.image_layers[predict_layer]['image_pixels_ignore_border']
    roi_filepath = conf.image_layers[predict_layer]['roi_filepath']

    ows_util.get_images_for_grid(
            wms_server_url=wms_server_url,
            wms_version=wms_version,
            wms_layernames=wms_layernames,
            wms_layerstyles=wms_layerstyles,
            srs=projection,
            output_image_dir=output_image_dir,
            image_gen_bounds=bbox,
            image_gen_roi_filepath=roi_filepath,
            grid_xmin=grid_xmin,
            grid_ymin=grid_ymin,
            image_srs_pixel_x_size=image_pixel_x_size,
            image_srs_pixel_y_size=image_pixel_y_size,
            image_pixel_width=image_pixel_width,
            image_pixel_height=image_pixel_height,
            image_pixels_ignore_border=image_pixels_ignore_border,
            nb_concurrent_calls=nb_concurrent_calls,
            random_sleep=random_sleep,
            image_format=image_format,
            pixels_overlap=image_pixels_overlap,
            column_start=column_start,
            nb_images_to_skip=nb_images_to_skip)

if __name__ == '__main__':
    load_images_args(sys.argv[1:])

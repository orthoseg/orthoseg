# -*- coding: utf-8 -*-
"""
Script to load images from a WMS server.
"""

import argparse
from pathlib import Path
import shlex
import sys

import pyproj

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper
from orthoseg.util import ows_util

logger = None

def load_images_argstr(argstr):
    args = shlex.split(argstr)
    load_images_args(args)

def load_images_args(args):

    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument("--configfile", type=str, required=True,
            help="The config file to use")
    
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    # Add back help 
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')

    # Interprete arguments
    args = parser.parse_args(args)

    ##### Run! #####
    load_images(projectconfig_path=Path(args.configfile))

def load_images(
        projectconfig_path: Path,
        imagelayerconfig_path: Path = None,
        load_testsample_images: bool = False):
    """
    Load and cache images for a segmentation project.
    
    Args:
        projectconfig_path (Path): Path to the projects config file.
        imagelayerconfig_path (Path, optional): Path to the imagelayer config file. If not specified, 
            the path specified in files.image_layers_config_filepath in the project config will be used. 
            Defaults to None.
        load_testsample_images (bool, optional): True to only load testsample images. Defaults to False.
    """

    ##### Init #####   
    # Load config
    config_filepaths = conf.search_projectconfig_files(projectconfig_path=projectconfig_path)
    conf.read_project_config(config_filepaths, imagelayerconfig_path)
    
    # Main initialisation of the logging
    log_helper.clean_log_dir(
            log_dir=conf.dirs.getpath('log_dir'),
            nb_logfiles_tokeep=conf.logging.getint('nb_logfiles_tokeep'))     
    global logger
    logger = log_helper.main_log_init(conf.dirs.getpath('log_dir'), __name__)
    logger.debug(f"Config used: \n{conf.pformat_config()}")

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
    
    # Get the download cron schedule
    download_cron_schedule = conf.download['cron_schedule']

    # Get the layer info
    predict_layer = conf.predict['image_layer']
    wms_server_url = conf.image_layers[predict_layer]['wms_server_url']
    wms_version = conf.image_layers[predict_layer]['wms_version']
    wms_layernames = conf.image_layers[predict_layer]['wms_layernames']
    wms_layerstyles = conf.image_layers[predict_layer]['wms_layerstyles']
    nb_concurrent_calls = conf.image_layers[predict_layer]['nb_concurrent_calls']
    random_sleep = conf.image_layers[predict_layer]['random_sleep']
    crs = pyproj.CRS.from_user_input(conf.image_layers[predict_layer]['projection'])
    bbox = conf.image_layers[predict_layer]['bbox']
    grid_xmin = conf.image_layers[predict_layer]['grid_xmin']
    grid_ymin = conf.image_layers[predict_layer]['grid_ymin']
    image_pixels_ignore_border = conf.image_layers[predict_layer]['image_pixels_ignore_border']
    roi_filepath = conf.image_layers[predict_layer]['roi_filepath']

    # Now we are ready to get the images...
    ows_util.get_images_for_grid(
            wms_server_url=wms_server_url,
            wms_version=wms_version,
            wms_layernames=wms_layernames,
            wms_layerstyles=wms_layerstyles,
            crs=crs,
            output_image_dir=output_image_dir,
            image_gen_bounds=bbox,
            image_gen_roi_filepath=roi_filepath,
            grid_xmin=grid_xmin,
            grid_ymin=grid_ymin,
            image_crs_pixel_x_size=image_pixel_x_size,
            image_crs_pixel_y_size=image_pixel_y_size,
            image_pixel_width=image_pixel_width,
            image_pixel_height=image_pixel_height,
            image_pixels_ignore_border=image_pixels_ignore_border,
            nb_concurrent_calls=nb_concurrent_calls,
            random_sleep=random_sleep,
            cron_schedule=download_cron_schedule,
            image_format=image_format,
            pixels_overlap=image_pixels_overlap,
            column_start=column_start,
            nb_images_to_skip=nb_images_to_skip)

if __name__ == '__main__':
    load_images_args(sys.argv[1:])

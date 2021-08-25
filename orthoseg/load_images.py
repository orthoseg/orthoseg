# -*- coding: utf-8 -*-
"""
Script to load images from a WMS server.
"""

import argparse
import logging
from pathlib import Path
import shlex
import sys
import traceback

import pyproj

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import email_helper
from orthoseg.util import log as log_util
from orthoseg.util import ows_util

#-------------------------------------------------------------
# First define/init general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def load_images_argstr(argstr):
    args = shlex.split(argstr)
    load_images_args(args)

def load_images_args(args):

    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument('-c', '--config', type=str, required=True,
            help="The config file to use")
    
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    # Add back help 
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')

    # Interprete arguments
    args = parser.parse_args(args)

    ##### Run! #####
    load_images(config_path=Path(args.config))

def load_images(
        config_path: Path,
        load_testsample_images: bool = False):
    """
    Load and cache images for a segmentation project.
    
    Args:
        config_path (Path): Path to the projects config file.
        load_testsample_images (bool, optional): True to only load testsample 
            images. Defaults to False.
    """

    ##### Init #####   
    # Load the config and save in a bunch of global variables zo it 
    # is accessible everywhere 
    conf.read_orthoseg_config(config_path)
    
    # Init logging
    log_util.clean_log_dir(
            log_dir=conf.dirs.getpath('log_dir'),
            nb_logfiles_tokeep=conf.logging.getint('nb_logfiles_tokeep'))     
    global logger
    logger = log_util.main_log_init(conf.dirs.getpath('log_dir'), __name__)      
    
    # Log + send email
    message = f"Start load_images for config {config_path.stem}"
    logger.info(message)
    logger.debug(f"Config used: \n{conf.pformat_config()}")    
    email_helper.sendmail(message)
    
    try:
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
        layersources = conf.image_layers[predict_layer]['layersources']
        nb_concurrent_calls = conf.image_layers[predict_layer]['nb_concurrent_calls']
        crs = pyproj.CRS.from_user_input(conf.image_layers[predict_layer]['projection'])
        bbox = conf.image_layers[predict_layer]['bbox']
        grid_xmin = conf.image_layers[predict_layer]['grid_xmin']
        grid_ymin = conf.image_layers[predict_layer]['grid_ymin']
        image_pixels_ignore_border = conf.image_layers[predict_layer]['image_pixels_ignore_border']
        roi_filepath = conf.image_layers[predict_layer]['roi_filepath']

        # Now we are ready to get the images...
        ows_util.get_images_for_grid(
                layersources=layersources,
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
                cron_schedule=download_cron_schedule,
                image_format=image_format,
                pixels_overlap=image_pixels_overlap,
                column_start=column_start,
                nb_images_to_skip=nb_images_to_skip)

        # Log and send mail
        message = f"Completed load_images for config {config_path.stem}"
        logger.info(message)
        email_helper.sendmail(message)
    except Exception as ex:
        message = f"ERROR while running load_images for task {config_path.stem}"
        logger.exception(message)
        email_helper.sendmail(subject=message, body=f"Exception: {ex}\n\n {traceback.format_exc()}")
        raise Exception(message) from ex

def main():
    try:
        load_images_args(sys.argv[1:])
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise

# If the script is ran directly...
if __name__ == '__main__':
    main()
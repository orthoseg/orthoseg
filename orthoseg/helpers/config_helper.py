# -*- coding: utf-8 -*-
"""
Module with specific helper functions to manage the configuration of orthoseg.
"""

import configparser
import json
import logging
from pathlib import Path
import pprint

from orthoseg.util import config_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

# Define the chars that cannot be used in codes that are use in filenames.
# Remark: '_' cannot be used because '_' is used as devider to parse filenames, and if it is 
# used in codes as well the parsing becomes a lot more difficult. 
illegal_chars_in_codes = ['_', ',', '.', '?', ':']

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def read_orthoseg_config(config_path: Path):

    # Determine list of config files that should be loaded
    config_paths = config_util.get_config_files(config_path)
    # Load them
    global config
    config = config_util.read_config_ext(config_paths)

    # Now do orthoseg-specific checks, inits,... on config
    global config_filepaths_used
    config_filepaths_used = config_paths

    # Now set global variables to each section as shortcuts
    global general
    general = config['general']
    global model
    model = config['model'] 
    global download
    download = config['download']
    global train
    train = config['train']
    global predict
    predict = config['predict']
    global postprocess
    postprocess = config['postprocess']
    global dirs 
    dirs = config['dirs']
    global files
    files = config['files']
    global logging
    logging = config['logging']
    global email
    email = config['email']

    # Some checks to make sure the config is loaded properly
    segment_subject = general.get('segment_subject')
    if segment_subject is None or segment_subject  == 'MUST_OVERRIDE':
        raise Exception(f"Projectconfig parameter general.segment_subject needs to be overruled to a proper name in a specific project config file, \nwith config_filepaths {config_paths}")
    elif any(illegal_character in segment_subject for illegal_character in illegal_chars_in_codes):
        raise Exception(f"Projectconfig parameter general.segment_subject ({segment_subject}) should not contain any of the following characters: {illegal_chars_in_codes}")

    # If the projects_dir parameter is a relative path, resolve it towards the location of
    # the project config file.
    projects_dir = dirs.getpath('projects_dir')
    if not projects_dir.is_absolute():
        projects_dir_absolute = (config_paths[-1].parent / projects_dir).resolve()
        logger.info(f"Parameter dirs.projects_dir was relative, so is now resolved to {projects_dir_absolute}")
        dirs['projects_dir'] = projects_dir_absolute.as_posix()

    # Read the layer config
    layer_config_filepath = files.getpath('image_layers_config_filepath')
    global layer_config_filepath_used
    layer_config_filepath_used = layer_config_filepath

    global image_layers
    image_layers = read_layer_config(layer_config_filepath=layer_config_filepath)

def read_layer_config(layer_config_filepath: Path) -> dict:
    # Init
    if not layer_config_filepath.exists():
        raise Exception(f"Layer config file not found: {layer_config_filepath}")

    # Read config file...
    layer_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')],
                        'dict': lambda x: None if x is None else json.loads(x),
                        'path': lambda x: Path(x)})
    layer_config.read(layer_config_filepath)
    
    # Prepare data
    image_layers = {}
    for image_layer in layer_config.sections():
        # First check if the image_layer code doesn't contain 'illegal' characters
        if any(illegal_char in image_layer for illegal_char in illegal_chars_in_codes):
            raise Exception(f"Section name [{image_layer}] in layer config should not contain any of the following characters: {illegal_chars_in_codes}, in {layer_config_filepath}")

        # Init layer with all parameters in the section as dict
        image_layers[image_layer] = dict(layer_config[image_layer])
        
        # If the layer source(s) are specified in a json parameter, parse it
        if 'layersources' in image_layers[image_layer]:
            image_layers[image_layer]['layersources'] = layer_config[image_layer].getdict('layersources')
            
            # Give default values to some optional properties of a server
            for layersource in image_layers[image_layer]['layersources']:
                if 'random_sleep' not in layersource:
                    layersource['random_sleep'] = 0
            
        else:
            # If not, the layersource should be specified in seperate parameters
            layersource = {}
            layersource['wms_server_url'] = layer_config[image_layer].get('wms_server_url')
            layersource['wms_version'] = layer_config[image_layer].get('wms_version', fallback='1.3.0')
            # The layer names and layer styles are lists
            layersource['layernames'] = layer_config[image_layer].getlist('wms_layernames')
            layersource['layerstyles'] = layer_config[image_layer].getlist('wms_layerstyles')
            # Some more properties
            layersource['bands'] = layer_config[image_layer].getlist('bands', fallback=None)
            layersource['random_sleep'] = layer_config[image_layer].getint('random_sleep', fallback=0)
            image_layers[image_layer]['layersources'] = [layersource]
        
        # Read nb_concurrent calls param
        image_layers[image_layer]['nb_concurrent_calls'] = (
                layer_config[image_layer].getint('nb_concurrent_calls', fallback=6))
            
        # Check if a region of interest is specified as file or bbox
        roi_filepath = layer_config[image_layer].getpath('roi_filepath', fallback=None)
        image_layers[image_layer]['roi_filepath'] = roi_filepath
        bbox_tuple = None
        if layer_config.has_option(image_layer, 'bbox'):
            bbox_list = layer_config[image_layer].getlist('bbox')
            bbox_tuple = (float(bbox_list[0]), float(bbox_list[1]), 
                            float(bbox_list[2]), float(bbox_list[3]))
            image_layers[image_layer]['bbox'] = bbox_tuple
        image_layers[image_layer]['bbox'] = bbox_tuple

        # Check if the grid xmin and xmax are specified            
        grid_xmin = 0
        if layer_config.has_option(image_layer, 'grid_xmin'):
            grid_xmin = layer_config[image_layer].getfloat('grid_xmin')                
        image_layers[image_layer]['grid_xmin'] = grid_xmin
        grid_ymin = 0
        if layer_config.has_option(image_layer, 'grid_ymin'):
            grid_ymin = layer_config[image_layer].getfloat('grid_ymin')
        image_layers[image_layer]['grid_ymin'] = grid_ymin

        # Check if a image_pixels_ignore_border is specified
        image_pixels_ignore_border = layer_config[image_layer].getint(
                'image_pixels_ignore_border', fallback=0)
        image_layers[image_layer]['image_pixels_ignore_border'] = image_pixels_ignore_border                                
    return image_layers

def pformat_config():
    message = f"Config files used: {pprint.pformat(config_filepaths_used)} \n"
    message += f"Layer config file used: {layer_config_filepath_used} \n"
    message += "Config info listing:\n"
    message += pprint.pformat(config_util.as_dict(config))
    message += "Layer config info listing:\n"
    message += pprint.pformat(image_layers)
    return message
    
# If the script is ran directly...
if __name__ == '__main__':
    raise Exception("Not implemented")
    
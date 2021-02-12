# -*- coding: utf-8 -*-
"""
Module with specific helper functions to manage the configuration of orthoseg.
"""

import configparser
import json
import logging
from pathlib import Path
import pprint
from typing import List

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def read_project_config(
        config_filepaths: List[Path],
        layer_config_filepath: Path = None):
    
    # Define the chars that cannot be used in codes that are use in filenames.
    # Remark: '_' cannot be used because '_' is used as devider to parse filenames, and if it is 
    # used in codes as well the parsing becomes a lot more difficult. 
    illegal_chars_in_codes = ['_', ',', '.', '?', ':']

    # Log config filepaths that don't exist...
    for config_filepath in config_filepaths:
        if not config_filepath.exists():
            logger.warning(f"config_filepath does not exist: {config_filepath}")

    # Read the configuration
    def safe_math_eval(string):
        if string is None:
            return None
            
        allowed_chars = "0123456789+-*(). /"
        for char in string:
            if char not in allowed_chars:
                raise Exception("Unsafe eval")

        return eval(string)

    global config
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')],
                        'listfloat': lambda x: [float(i.strip()) for i in x.split(',')],
                        'dict': lambda x: None if x is None else json.loads(x),
                        'path': lambda x: None if x is None else Path(x),
                        'eval': lambda x: safe_math_eval(x)},
            allow_no_value=True)

    config.read(config_filepaths)
    global config_filepaths_used
    config_filepaths_used = config_filepaths

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

    # Some checks to make sure the config is loaded properly
    segment_subject = general.get('segment_subject')
    if segment_subject is None or segment_subject  == 'MUST_OVERRIDE':
        raise Exception(f"Projectconfig parameter general.segment_subject needs to be overruled to a proper name in a specific project config file, \nwith config_filepaths {config_filepaths}")
    elif any(illegal_character in segment_subject for illegal_character in illegal_chars_in_codes):
        raise Exception(f"Projectconfig parameter general.segment_subject ({segment_subject}) should not contain any of the following characters: {illegal_chars_in_codes}")

    # If the projects_dir parameter is a relative path, resolve it towards the location of
    # the project config file.
    projects_dir = dirs.getpath('projects_dir')
    if not projects_dir.is_absolute():
        projects_dir_absolute = (config_filepaths[-1].parent / projects_dir).resolve()
        logger.info(f"Parameter dirs.projects_dir was relative, so is now resolved to {projects_dir_absolute}")
        dirs['projects_dir'] = projects_dir_absolute.as_posix()
        
    # Read the layer config
    if layer_config_filepath is None:
        layer_config_filepath = files.getpath('image_layers_config_filepath')

    if not layer_config_filepath.exists():
        raise Exception(f"Layer config file not found: {layer_config_filepath}")

    global layer_config
    layer_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')],
                        'path': lambda x: Path(x)})
    layer_config.read(layer_config_filepath)
    global layer_config_filepath_used
    layer_config_filepath_used = layer_config_filepath

    global image_layers
    image_layers = {}
    for image_layer in layer_config.sections():
        # First check if the image_layer code doesn't exist 'illegal' characters
        if any(illegal_char in image_layer for illegal_char in illegal_chars_in_codes):
            raise Exception(f"Section name [{image_layer}] in layer config should not contain any of the following characters: {illegal_chars_in_codes}, in {layer_config_filepath}")

        image_layers[image_layer] = dict(layer_config[image_layer])
        
        # The layer names and layer styles are lists
        wms_layernames = layer_config[image_layer].getlist('wms_layernames')
        image_layers[image_layer]['wms_layernames'] = wms_layernames
        wms_layerstyles = layer_config[image_layer].getlist('wms_layerstyles')
        image_layers[image_layer]['wms_layerstyles'] = wms_layerstyles

        # Give default values to some other properties of a server
        nb_concurrent_calls = layer_config[image_layer].getint('nb_concurrent_calls', fallback=6)
        image_layers[image_layer]['nb_concurrent_calls'] = nb_concurrent_calls
        random_sleep = layer_config[image_layer].getint('random_sleep', fallback=6)
        image_layers[image_layer]['random_sleep'] = random_sleep

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
            
def pformat_config():
    message = f"Config files used: {pprint.pformat(config_filepaths_used)} \n"
    message += f"Layer config file used: {layer_config_filepath_used} \n"
    message += "Config info listing:\n"
    message += pprint.pformat(as_dict())
    message += "Layer config info listing:\n"
    message += pprint.pformat(image_layers)
    return message
    
def as_dict():
    """
    Converts a ConfigParser object into a dictionary.

    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.
    """
    the_dict = {}
    for section in config.sections():
        the_dict[section] = {}
        for key, val in config.items(section):
            the_dict[section][key] = val
    return the_dict

def search_projectconfig_files(
        projectconfig_path: Path,
        projectconfig_defaults_overrule_path: Path = None,
        projectconfig_defaults_path: Path = None) -> List[Path]:

    config_filepaths = []
    # First the default settings, because they can be overridden by the other 2
    if projectconfig_defaults_path is None:
        install_dir = Path(__file__).resolve().parent.parent
        projectconfig_defaults_path = install_dir / 'project_defaults.ini'
        if projectconfig_defaults_path.exists():
            config_filepaths.append(projectconfig_defaults_path)
        else:
            logger.warn(f"No default projectconfig found, so won't be used, but this could give problems when upgrading to new versions: {projectconfig_defaults_path}")

    # Then the settings on the projectsdir level
    if projectconfig_defaults_overrule_path is not None:
        config_filepaths.append(projectconfig_defaults_overrule_path)
    else:
        # Check if a config file exists on the default overrule location
        projects_dir = projectconfig_path.parent.parent
        projectconfig_defaults_overrule_path = projects_dir / 'project_defaults_overrule.ini'
        if projectconfig_defaults_overrule_path.exists():
            config_filepaths.append(projectconfig_defaults_overrule_path)
        else: 
            logger.warn(f"No (optional) project_defaults.ini file found on projectdir level, so won't be used: {projectconfig_defaults_overrule_path}")     

    # Specific settings for the subject
    config_filepaths.append(projectconfig_path)

    return config_filepaths

# If the script is ran directly...
if __name__ == '__main__':
    raise Exception("Not implemented")
    
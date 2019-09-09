# -*- coding: utf-8 -*-
"""
Module with specific helper functions to manage the configuration of orthoseg.
"""

import os
import configparser
import pprint

def read_config(config_filepaths: []):
        
    # Log config filepaths that don't exist...
    for config_filepath in config_filepaths:
        if not os.path.exists(config_filepath):
            print(f"config_filepath does not exist: {config_filepath}")

    # Read the configuration
    global config
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')]})

    config.read(config_filepaths)
    global config_filepaths_used
    config_filepaths_used = config_filepaths

    # Now set global variables to each section as shortcuts
    global general
    general = config['general']
    global email
    email = config['email']
    global train
    train = config['train']
    global predict
    predict = config['predict']
    global model
    model = config['model']
    global dirs 
    dirs = config['dirs']
    global files
    files = config['files']

    # Create a dictionary of the configured image datasources 
    global image_datasources
    image_datasources = {}
    for section in config.sections():
        if section.startswith('image_datasource_'):
            image_datasource_code = section.replace('image_datasource_', '')
            image_datasources[image_datasource_code] = dict(config[section])
            
            # The layer names and layer styles are lists
            wms_layernames = config[section].getlist('wms_layernames')
            image_datasources[image_datasource_code]['wms_layernames'] = wms_layernames
            wms_layerstyles = config[section].getlist('wms_layerstyles')
            image_datasources[image_datasource_code]['wms_layerstyles'] = wms_layerstyles

            # Geve default values to some other properties of a server
            nb_concurrent_calls = config[section].getint('nb_concurrent_calls', 6)
            image_datasources[image_datasource_code]['nb_concurrent_calls'] = nb_concurrent_calls
            random_sleep = config[section].getint('random_sleep', 6)
            image_datasources[image_datasource_code]['random_sleep'] = random_sleep
            
            # Check if a bbox is specified
            bbox_tuple = None
            if config.has_option(section, 'bbox'):
                bbox_list = config[section].getlist('bbox')
                bbox_tuple = (float(bbox_list[0]), float(bbox_list[1]), 
                              float(bbox_list[2]), float(bbox_list[3]))
                image_datasources[image_datasource_code]['bbox'] = bbox_tuple
            image_datasources[image_datasource_code]['bbox'] = bbox_tuple

            # Check if the grid xmin and xmax are specified            
            grid_xmin = 0
            if config.has_option(section, 'grid_xmin'):
                grid_xmin = config[section].getfloat('grid_xmin')                
            image_datasources[image_datasource_code]['grid_xmin'] = grid_xmin
            grid_ymin = 0
            if config.has_option(section, 'grid_ymin'):
                grid_ymin = config[section].getfloat('grid_ymin')
            image_datasources[image_datasource_code]['grid_ymin'] = grid_ymin

            # Check if a image_pixels_ignore_border is specified
            image_pixels_ignore_border = 0
            if config.has_option(section, 'image_pixels_ignore_border'):
                image_pixels_ignore_border = config[section].getint('image_pixels_ignore_border')
            image_datasources[image_datasource_code]['image_pixels_ignore_border'] = image_pixels_ignore_border                                
            
def pformat_config():
    message = f"Config files used: {pprint.pformat(config_filepaths_used)} \n"
    message += "Config info listing:\n"
    message += pprint.pformat({section: dict(config[section]) for section in config.sections()})
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

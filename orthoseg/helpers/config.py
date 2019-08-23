# -*- coding: utf-8 -*-
"""
Module that manages the configuration of a segmentation
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
            
            # The layer names ad layer styles are lists
            wms_layer_names = config[section].getlist('wms_layer_names')
            image_datasources[image_datasource_code]['wms_layer_names'] = wms_layer_names
            wms_layer_styles = config[section].getlist('wms_layer_styles')
            image_datasources[image_datasource_code]['wms_layer_styles'] = wms_layer_styles

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

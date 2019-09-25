# -*- coding: utf-8 -*-
"""
Module with specific helper functions to manage the configuration of orthoseg.
"""

import os
import configparser
import pprint

def read_config(
        config_filepaths: [],
        layer_config_filepath: str):
        
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

    # Read the layer config
    global layer_config
    layer_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')]})
    layer_config.read(layer_config_filepath)
    global layer_config_filepath_used
    layer_config_filepath_used = layer_config_filepath

    global image_layers
    image_layers = {}
    for image_layer in layer_config.sections():
        image_layers[image_layer] = dict(layer_config[image_layer])
        
        # The layer names and layer styles are lists
        wms_layernames = layer_config[image_layer].getlist('wms_layernames')
        image_layers[image_layer]['wms_layernames'] = wms_layernames
        wms_layerstyles = layer_config[image_layer].getlist('wms_layerstyles')
        image_layers[image_layer]['wms_layerstyles'] = wms_layerstyles

        # Give default values to some other properties of a server
        nb_concurrent_calls = layer_config[image_layer].getint('nb_concurrent_calls', 6)
        image_layers[image_layer]['nb_concurrent_calls'] = nb_concurrent_calls
        random_sleep = layer_config[image_layer].getint('random_sleep', 6)
        image_layers[image_layer]['random_sleep'] = random_sleep

        # Check if a region of interest is specified as file or bbox
        roi_filepath = layer_config[image_layer].get('roi_filepath', None)
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
        image_pixels_ignore_border = layer_config[image_layer].getint('image_pixels_ignore_border', 0)
        image_layers[image_layer]['image_pixels_ignore_border'] = image_pixels_ignore_border                                
            
def pformat_config():
    message = f"Config files used: {pprint.pformat(config_filepaths_used)} \n"
    message += f"Layer config file used: {layer_config_filepath_used} \n"
    message += "Config info listing:\n"
    message += pprint.pformat({section: dict(config[section]) for section in config.sections()})
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

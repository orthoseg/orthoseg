# -*- coding: utf-8 -*-
"""
Module that manages the configuration of a segmentation

@author: Pieter Roggemans
"""

import configparser
import pprint

def read_config(config_filepaths: []):
        
    # Read the configuration
    global config
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

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
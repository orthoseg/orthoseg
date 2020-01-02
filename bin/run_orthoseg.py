# -*- coding: utf-8 -*-
"""
Process the jobs as scheduled in the run_jobs_config file.
"""

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import List

import pandas as pd

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
[sys.path.append(i) for i in ['.', '..']]
from orthoseg.helpers import config_helper as conf 
from orthoseg.helpers import log_helper

def orthoseg_argstr(argstr):
    args = shlex.split(argstr)
    orthoseg_args(args)

def orthoseg_args(args):

    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument("--action", type=str, required=True,
            help="The action you want to perform")
    required.add_argument("--config", type=str, required=True,
            help="The config to perform the action with")
    
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    # Add back help 
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')

    # Interprete arguments
    args = parser.parse_args(args)

    ##### Run! #####
    orthoseg(action=args.action, config=args.config)

def orthoseg(
        action: str,
        config: str):

    # Prepare the path to the job dir,...
    
    script_dir = Path(os.path.abspath(__file__)).parent
    base_dir = script_dir.parent
    config_dir = base_dir / "config"

    # Get needed config + load it
    print(f"Start {action} on {config}")
    config_filepaths = get_needed_config_files(config_dir=config_dir, config=config)
    layer_config_filepath = config_dir / 'image_layers.ini'
    conf.read_config(config_filepaths, layer_config_filepath)
    
    # Main initialisation of the logging
    global logger
    logger = log_helper.main_log_init(conf.dirs.getpath('log_training_dir'), __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Now start the appropriate action 
    try:
        if(action == 'train'):
            import orthoseg.run_train as train
            train.run_training_session()
        elif(action == 'predict'):
            import orthoseg.run_predict as pred
            pred.run_prediction()
        elif(action == 'load_images'):
            import orthoseg.run_load_images as load_images
            load_images.load_images()
        elif(action == 'load_testsample_images'):
            import orthoseg.run_load_images as load_images
            load_images.load_images(load_testsample_images=True)
        elif(action == 'postprocess'):
            import orthoseg.run_postprocess as postp
            postp.postprocess_predictions()
        else:
            raise Exception(f"Unsupported action: {action}")
    except Exception as ex:
        message = f"OrthoSeg ERROR in action {action} on {config}"
        logger.exception(message)
        raise Exception(message) from ex

def get_needed_config_files(
        config_dir: Path,
        config: str = None) -> List[Path]:

    # General settings need to be first in list
    config_filepaths = [config_dir / 'general.ini']

    # Then specific settings depending on the OS
    if os.name == 'posix':
        config_filepaths.append(config_dir / 'general_posix.ini')
    elif os.name == 'nt':
        None
    else: 
        raise Exception(f"Unsupported os.name: {os.name}")

    # Specific settings for the subject if one is specified
    if(config is not None):
        config_filepath = config_dir / config
        if not os.path.exists(config_filepath):
            raise Exception(f"Config file specified does not exist: {config_filepath}")
        config_filepaths.append(config_filepath)

    # Local overrule settings
    config_filepaths.append(config_dir / 'local_overrule.ini')

    return config_filepaths

if __name__ == '__main__':
    orthoseg_args(sys.argv[1:])

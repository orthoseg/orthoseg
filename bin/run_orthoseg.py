# -*- coding: utf-8 -*-
"""
Process the jobs as scheduled in the run_jobs_config file.
"""

import configparser
from email.message import EmailMessage
import glob 
import os
import smtplib
import shlex
import sys

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
[sys.path.append(i) for i in ['.', '..']]

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"

import pandas as pd

from orthoseg.helpers import config_helper as conf 
from orthoseg.helpers import log_helper

import argparse

def orthoseg_argstr(argstr):
    #args = list(filter(None, argstr.split(' ')))
    args = shlex.split(argstr)
    orthoseg_args(args)

def orthoseg_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, help="The action you want to perform")
    parser.add_argument("--subject", type=str, help="The subject to perform the action on")
    args = parser.parse_args(args)

    orthoseg(action=args.action, subject=args.subject)

def orthoseg(
        action: str,
        subject: str):

    # Prepare the path to the job dir,...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir, _ = os.path.split(script_dir)
    config_dir = os.path.join(base_dir, "config")

    # Get needed config + load it
    print(f"Start {action} on {subject}")
    config_filepaths = get_needed_config_files(config_dir=config_dir, subject=subject)
    conf.read_config(config_filepaths)
    
    # Main initialisation of the logging
    global logger
    logger = log_helper.main_log_init(conf.dirs['log_training_dir'], __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Now start the appropriate action 
    try:
        if(action == 'train'):
            import orthoseg.run_train as train
            train.run_training_session(config_filepaths=config_filepaths)
        elif(action == 'predict'):
            import orthoseg.run_predict as pred
            pred.run_prediction(config_filepaths=config_filepaths)
        elif(action == 'load_images'):
            import orthoseg.run_load_images as load_images
            load_images.load_images(config_filepaths=config_filepaths)
        elif(action == 'load_testsample_images'):
            import orthoseg.run_load_images as load_images
            load_images.load_images(config_filepaths=config_filepaths, load_testsample_images=True)
        elif(action == 'postprocess'):
            import orthoseg.run_postprocess as postp
            postp.postprocess_predictions(config_filepaths=config_filepaths)
        elif(action == 'test'):
            return 'TEST OK'
        else:
            raise Exception(f"Unsupported action: {action}")
    except Exception as ex:
        message = f"OrthoSeg ERROR in action {action} on {subject}"
        logger.exception(message)
        raise Exception(message) from ex

def get_needed_config_files(
        config_dir: str,
        subject: str = None) -> []:

    # General settings need to be first in list
    config_filepaths = [os.path.join(config_dir, 'general.ini')]

    # Then specific settings depending on the OS
    if os.name == 'posix':
        config_filepaths.append(os.path.join(config_dir, 'general_posix.ini'))
    elif os.name == 'nt':
        None
    else: 
        raise Exception(f"Unsupported os.name: {os.name}")

    # Specific settings for the subject if one is specified
    if(subject is not None):
        config_filepaths.append(os.path.join(config_dir, f"{subject}.ini"))

    # Local overrule settings
    config_filepaths.append(os.path.join(config_dir, 'local_overrule.ini'))

    return config_filepaths

if __name__ == '__main__':
    orthoseg_args(sys.argv[1:])

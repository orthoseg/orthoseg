# -*- coding: utf-8 -*-
"""
Process the jobs in the job directory.
"""

import configparser
import glob 
import os
import sys

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
[sys.path.append(i) for i in ['.', '..']]

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"

import pandas as pd

import orthoseg.helpers.config as conf 
import orthoseg.train as train
import orthoseg.predict as pred

def run_orthoseg():

    # Prepare the path to the job dir,...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_config_filepath = os.path.join(script_dir, "run_orthoseg_config.csv")
    base_dir, _ = os.path.split(script_dir)
    config_dir = os.path.join(base_dir, "config")

    # Read the jobs that need to be ran in the run_jobs file
    run_config_df = pd.read_csv(run_config_filepath)
    mandatory_columns = ['to_run', 'subject', 'action']
    missing_columns = set(mandatory_columns).difference(set(run_config_df.columns))
    if(len(missing_columns) > 0):
        raise Exception(f"Missing columns in {run_config_filepath}: {missing_columns}")
    
    # Loop over jobs to run and treat them
    for run_info in run_config_df.itertuples(): 
        if(run_info.to_run == 0):
            continue

        # Get needed config, and go for it!
        config_filepaths = get_needed_config_files(config_dir=config_dir,
                                                   subject=run_info.subject)
        if(run_info.action == 'train'):
            train.run_training_session(segment_config_filepaths=config_filepaths)
        elif(run_info.action == 'predict'):
            pred.run_prediction(segment_config_filepaths=config_filepaths)
        else:
            raise Exception(f"Unsupported action: {run_info.action}")

def get_needed_config_files(config_dir: str,
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
    run_orthoseg()
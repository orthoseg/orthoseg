# -*- coding: utf-8 -*-
"""
Process the jobs as scheduled in the run_jobs_config file.
"""

import configparser
from email.message import EmailMessage
import glob 
import os
import smtplib
import sys

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
[sys.path.append(i) for i in ['.', '..']]

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"

import pandas as pd

from orthoseg.helpers import config_helper as conf 
from orthoseg.helpers import log_helper

def run_jobs():

    # Prepare the path to the job dir,...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_config_filepath = os.path.join(script_dir, "run_job_config.csv")
    base_dir, _ = os.path.split(script_dir)
    config_dir = os.path.join(base_dir, "config")

    # Read the jobs that need to be ran in the run_jobs file
    run_config_df = read_run_job_config(run_config_filepath)

    ##### Run jobs! #####
    # Loop over jobs to run and treat them
    for run_info in run_config_df.itertuples(): 
        if(run_info.to_run == 0):
            continue

        # Get needed config
        print(f"Start {run_info.action} on {run_info.subject}")
        config_filepaths = get_needed_config_files(config_dir=config_dir, subject=run_info.subject)

        # Read the configuration
        conf.read_config(config_filepaths)
        
        # Main initialisation of the logging
        global logger
        logger = log_helper.main_log_init(conf.dirs['log_training_dir'], __name__)      
        logger.info(f"Config used: \n{conf.pformat_config()}")

        # Now start the appropriate action 
        try:
            if(run_info.action == 'train'):
                import orthoseg.run_train as train
                train.run_training_session(config_filepaths=config_filepaths)
            elif(run_info.action == 'predict'):
                import orthoseg.run_predict as pred
                pred.run_prediction(config_filepaths=config_filepaths)
            elif(run_info.action == 'load_images'):
                import orthoseg.run_load_images as load_images
                load_images.load_images(config_filepaths=config_filepaths)
            elif(run_info.action == 'load_testsample_images'):
                import orthoseg.run_load_images as load_images
                load_images.load_images(config_filepaths=config_filepaths, load_testsample_images=True)
            elif(run_info.action == 'postprocess'):
                import orthoseg.run_postprocess as postp
                postp.postprocess_predictions(config_filepaths=config_filepaths)
            else:
                raise Exception(f"Unsupported action: {run_info.action}")
            sendmail(f"OrthoSeg completed action {run_info.action} on {run_info.subject}")
        except Exception as ex:
            message = f"OrthoSeg ERROR in action {run_info.action} on {run_info.subject}"
            logger.exception(message)
            sendmail(message)
            raise Exception(message) from ex

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

def read_run_job_config(filepath):
    
    # Read
    run_config_df = pd.read_csv(filepath)

    # Trim column names and string columns
    for column in run_config_df.columns:
        column_stripped = column.strip()
        if column != column_stripped:
            run_config_df.rename(columns={column:column_stripped}, inplace=True)
        if run_config_df[column_stripped].dtype in ('str', 'object'):
            run_config_df[column_stripped] = run_config_df[column_stripped].astype(str).str.strip()

    # Check mandatory columns
    mandatory_columns = ['subject', 'to_run', 'action']
    missing_columns = set(mandatory_columns).difference(set(run_config_df.columns))
    if(len(missing_columns) > 0):
        raise Exception(f"Missing column(s) in {filepath}: {missing_columns}")
    
    return run_config_df

def sendmail(
        subject: str, 
        body: str = None,
        stop_on_error: bool = False):

    try:
        # Create message
        msg = EmailMessage()
        msg.add_header('from', conf.email['from'])
        msg.add_header('to', conf.email['to'])
        msg.add_header('subject', subject)
        if body is not None:
            msg.set_payload(body)

        # Send the email
        server = smtplib.SMTP(conf.email['smtp_server'])
        #server.login("MrDoe", "PASSWORD")
        server.send_message(msg)
        server.quit()
    except Exception as ex:
        if stop_on_error is False:
            logger.exception("Error sending email")
        else:
            raise Exception("Error sending email") from ex
        

if __name__ == '__main__':
    run_jobs()

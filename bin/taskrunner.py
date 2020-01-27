# -*- coding: utf-8 -*-
"""
Process the tasks as configured in a tasks csv file.
"""

import argparse
import configparser
from email.message import EmailMessage
import json
import logging
import logging.config
from pathlib import Path
import smtplib
from typing import List

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, '.')

import pandas as pd

def run_tasks(config_filepaths: List[Path]):

    ##### Init #####
    # Read the taskrunner configuration
    global global_config
    global_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')],
                        'dict': lambda x: json.loads(x),
                        'path': lambda x: Path(x)})
    global_config.read(config_filepaths)

    # Init logging
    logging.config.dictConfig(global_config['logging'].getdict('logconfig'))
    global logger
    logger = logging.getLogger()
    logger.info(f"Config files used for taskrunner: {config_filepaths}")
    
    # Get the default config dir
    config_dir = Path(global_config['dirs'].get('config_dir'))
    stop_on_error = global_config['dirs'].getboolean('stop_on_error')

    # Read the tasks that need to be ran in the run_tasks file
    tasks_filepath = global_config['files'].getpath('tasks_path')
    tasks_df = get_tasks(tasks_filepath)

    # Loop over tasks to run
    for task in tasks_df.itertuples(): 
        if(task.active == 0):
            continue

        logger.info(f"Start action {task.action} for config {task.config}")
        try:
            if(task.action == 'train'):
                from orthoseg import train
                train.train(config_dir=config_dir, config_filename=task.config)
            elif(task.action == 'predict'):
                from orthoseg import predict
                predict.predict(config_dir=config_dir, config_filename=task.config)
            elif(task.action == 'load_images'):
                from orthoseg import load_images
                load_images.load_images(config_dir=config_dir, config_filename=task.config)
            elif(task.action == 'load_testsample_images'):
                from orthoseg import load_images
                load_images.load_images(
                        config_dir=config_dir, 
                        config_filename=task.config, 
                        load_testsample_images=True)
            elif(task.action == 'postprocess'):
                from orthoseg import postprocess
                postprocess.postprocess(config_dir=config_dir, config_filename=task.config)
            else:
                raise Exception(f"Unsupported action: {task.action}")

            message = f"Completed action {task.action} for config {task.config}"
            logger.info(message)
            sendmail(message)
        except Exception as ex:
            message = f"ERROR in task with action {task.action} for config {task.config}"
            logger.exception(message)
            sendmail(subject=message, body=f"Exception: {ex}")
            if stop_on_error:
                raise Exception(message) from ex

def get_tasks(filepath: Path):
    
    # Read
    tasks_df = pd.read_csv(filepath)

    # Trim column names and string columns
    for column in tasks_df.columns:
        column_stripped = column.strip()
        if column != column_stripped:
            tasks_df.rename(columns={column:column_stripped}, inplace=True)
        if tasks_df[column_stripped].dtype in ('str', 'object'):
            tasks_df[column_stripped] = tasks_df[column_stripped].astype(str).str.strip()

    # Check mandatory columns
    mandatory_columns = ['active', 'config', 'action']
    missing_columns = set(mandatory_columns).difference(set(tasks_df.columns))
    if(len(missing_columns) > 0):
        raise Exception(f"Missing column(s) in {filepath}: {missing_columns}")
    
    return tasks_df

def sendmail(
        subject: str, 
        body: str = None,
        stop_on_error: bool = False):

    mail_from = global_config['email'].get('from', None)
    mail_to = global_config['email'].get('to', None)
    mail_server = global_config['email'].get('server', None)
    mail_server_username = global_config['email'].get('username', None)
    mail_server_password = global_config['email'].get('password', None)

    # If one of the necessary parameters not provided, log subject
    if(mail_from is None
       or mail_to is None
       or mail_server is None):
        logger.warning(f"Mail global_config not provided to send email with subject: {subject}")
        return

    try:
        # Create message
        msg = EmailMessage()
        msg.add_header('from', mail_from)
        msg.add_header('to', mail_to)
        msg.add_header('subject', subject)
        if body is not None:
            msg.set_payload(body)

        # Send the email
        server = smtplib.SMTP(mail_server)
        if(mail_server_username is not None
           and mail_server_password is not None):
            server.login(mail_server_username, mail_server_password)
        server.send_message(msg)
        server.quit()
    except Exception as ex:
        if stop_on_error is False:
            logger.exception("Error sending email")
        else:
            raise Exception("Error sending email") from ex
        
if __name__ == '__main__':
    
    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('-c', '--configfile',
            help='The config file to use. If not provided, taskrunner tries to use ./taskrunner.ini or ./taskrunner_sample.ini.')
    # Add back help         
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')
    args = parser.parse_args()

    # Determine config_filepaths
    script_dir = Path(__file__).resolve().parent
    config_defaults_filepath = script_dir / 'taskrunner_defaults.ini'
    config_filepaths = [config_defaults_filepath]
    if args.configfile is not None:
        # config_filepath is provided, so use it as well
        config_filepaths.append(Path(args.configfile))
    else:
        # If the default config file name exist, use it as well
        config_filepath = script_dir / 'taskrunner.ini'
        if config_filepath.exists():
            config_filepaths.append(config_filepath)
        else:
            print("Only taskrunner_defaults.ini settings will be used. If you want to overrule settings, follow the guidelines specified in this file on how to do this.")

    # Run!
    run_tasks(config_filepaths=config_filepaths)

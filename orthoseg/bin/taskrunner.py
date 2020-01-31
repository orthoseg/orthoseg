# -*- coding: utf-8 -*-
"""
Process the tasks as configured in a tasks csv file.
"""

import argparse
import configparser
import datetime
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

def run_tasks(
        tasks_path: Path,
        config_filepaths: List[Path],
        stop_on_error: bool = False):

    ##### Init #####
    # Read the taskrunner configuration
    global runner_config
    runner_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')],
                        'dict': lambda x: json.loads(x),
                        'path': lambda x: Path(x)})
    runner_config.read(config_filepaths)

    # Init logging
    logconfig_dict = runner_config['logging'].getdict('logconfig')

    # If there are file handlers, replace possible placeholders + make sure log dir exists
    for handler in logconfig_dict['handlers']:
        if "filename" in logconfig_dict['handlers'][handler]:
            # Format the filename
            log_path = Path(logconfig_dict['handlers'][handler]['filename'].format(
                    iso_datetime=f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"))

            # If the log_path is a relative path, resolve it towards the location of
            # the tasks file.
            if not log_path.is_absolute():
                log_path = tasks_path.parent / log_path
                print(f"Parameter logconfig.handlers.{handler}.filename was relative, so is now resolved to {log_path}")

            logconfig_dict['handlers'][handler]['filename'] = log_path.as_posix()

            # Also make sure the log dir exists
            log_path.parent.mkdir(parents=True, exist_ok=True)

    # Now load the log config
    logging.config.dictConfig(logconfig_dict)
    global logger
    logger = logging.getLogger()
    logger.info(f"Config files used for taskrunner: {config_filepaths}")
    
    # Read the tasks that need to be ran in the run_tasks file
    tasks_df = get_tasks(tasks_path)

    # Loop over tasks to run
    for task in tasks_df.itertuples(): 
        # If the task is not active, skip
        if(task.active == 0):
            continue

        # Format the projectfile_path
        projectfile_path = Path(task.config)
        # If the projectfile path is not absolute, treat is as relative to the tasks csv file...
        if not projectfile_path.is_absolute():
            projectfile_path = tasks_path.parent / projectfile_path

        # Now we are ready to start the action        
        logger.info(f"Start action {task.action} for config {projectfile_path}")
        try:
            if(task.action == 'train'):
                from orthoseg import train
                train.train(projectconfig_path=projectfile_path)
            elif(task.action == 'predict'):
                from orthoseg import predict
                predict.predict(projectconfig_path=projectfile_path)
            elif(task.action == 'load_images'):
                from orthoseg import load_images
                load_images.load_images(projectconfig_path=projectfile_path)
            elif(task.action == 'load_testsample_images'):
                from orthoseg import load_images
                load_images.load_images(
                        projectconfig_path=projectfile_path, 
                        load_testsample_images=True)
            elif(task.action == 'postprocess'):
                from orthoseg import postprocess
                postprocess.postprocess(projectconfig_path=projectfile_path)
            else:
                raise Exception(f"Unsupported action: {task.action}")

            message = f"Completed action {task.action} for config {projectfile_path}"
            logger.info(message)
            sendmail(message)
        except Exception as ex:
            message = f"ERROR in task with action {task.action} for config {projectfile_path}"
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

    if runner_config['email'].getboolean('enabled', fallback=False):
        return
    mail_from = runner_config['email'].get('from', None)
    mail_to = runner_config['email'].get('to', None)
    mail_server = runner_config['email'].get('server', None)
    mail_server_username = runner_config['email'].get('username', None)
    mail_server_password = runner_config['email'].get('password', None)

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
    optional.add_argument('-t', '--tasksfile',
            help='The tasks file to use. If not provided, taskrunner tries to use ../../projects/tasks.csv relative to taskrunner.py')
    # Add back help         
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')
    args = parser.parse_args()

    # If tasks file is not specified, use default location
    if args.tasksfile is not None:
        tasks_path = Path(args.tasksfile)
    else:
        script_dir = Path(__file__).resolve().parent
        tasks_path = script_dir / '../../projects/tasks.csv'
    
    # Find the config filepaths to use
    config_filepaths = []
    script_dir = Path(__file__).resolve().parent
    config_defaults_path = script_dir / '../project_defaults.ini'
    if config_defaults_path.exists():
        config_filepaths.append(config_defaults_path)
    else:
        print(f"Warning: default project settings not found: {config_defaults_path}")
    config_defaults_overrule_path = tasks_path.parent / 'project_defaults_overrule.ini'
    if config_defaults_overrule_path.exists():
        config_filepaths.append(config_defaults_overrule_path)
    else:
        print(f"Warning: default overule project settings not found: {config_defaults_overrule_path}")

    # Run!
    run_tasks(
            tasks_path=tasks_path,
            config_filepaths=config_filepaths)

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
import os
from pathlib import Path
import pprint
import smtplib
import traceback
from typing import List

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

import pandas as pd

from orthoseg.util import git_downloader

def main():
    
    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('-t', '--tasksfile',
            help='The path to the tasks .csv file to use.')
    optional.add_argument('-s', '--sampleprojects', action='store_true',
            help="Download the sample projects to ~/orthoseg/sample_projects if this directory doesn't exist and run.")
    # Add back help         
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')
    args = parser.parse_args()

    # If tasks file is specified, use it
    if args.tasksfile is not None:
        tasks_path = Path(args.tasksfile)
    elif args.sampleprojects is True:
        # If sampleprojects is true, download sampleproject and run
        orthoseg_dir = Path.home() / 'orthoseg'
        projects_dir = orthoseg_dir / 'sample_projects'
        if not projects_dir.exists():
            print("Download sample projects")
            git_repo_dir = 'https://github.com/theroggy/orthoseg/tree/master/sample_projects'
            git_downloader.download(repo_url=git_repo_dir, output_dir=orthoseg_dir)
        tasks_path = projects_dir / 'tasks.csv'
    else:
        parser.print_help()
        sys.exit(1)
    
    # Find the config filepaths to use
    config_filepaths = []
    script_dir = Path(__file__).resolve().parent
    config_defaults_path = script_dir / 'project_defaults.ini'
    if config_defaults_path.exists():
        config_filepaths.append(config_defaults_path)
    else:
        print(f"Warning: default project settings not found: {config_defaults_path}")
    config_defaults_overrule_path = tasks_path.parent / 'project_defaults_overrule.ini'
    if config_defaults_overrule_path.exists():
        config_filepaths.append(config_defaults_overrule_path)
    else:
        print(f"Warning: default overrule project settings not found: {config_defaults_overrule_path}")

    # Run!
    run_tasks(
            tasks_path=tasks_path,
            config_filepaths=config_filepaths)

def run_tasks(
        tasks_path: Path,
        config_filepaths: List[Path],
        stop_on_error: bool = False):

    ##### Init #####
    # Read the configuration
    global runner_config
    runner_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')],
                        'dict': lambda x: json.loads(x),
                        'path': lambda x: None if x is None else Path(x)},
            allow_no_value=True)
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
    #logger.info(f"Config files used for orthoseg task loop: {config_filepaths}")
        
    # Make sure GDAL is properly set up
    if os.environ.get('GDAL_DATA') is None:
        logger.info(f"environment: {pprint.pformat(os.environ)}")
        #os.environ['GDAL_DATA'] = r"C:\Tools\miniconda3\envs\orthoseg\Library\share\gdal"
        os.environ['GDAL_DATA'] = r"C:\Users\pierog\Miniconda3\envs\orthosegdev\Library\share\gdal"
        logger.warn(f"Environment variable GDAL_DATA was not set, so set to {os.environ['GDAL_DATA']}")
    if os.environ.get('PROJ_LIB') is None:
        #os.environ['PROJ_LIB'] = r"C:\Tools\miniconda3\envs\orthoseg\Library\share\proj"
        os.environ['PROJ_LIB'] = r"C:\Users\pierog\Miniconda3\envs\orthosegdev\Library\share\proj"
        logger.warn(f"Environment variable PROJ_LIB was not set, so set to {os.environ['PROJ_LIB']}")
    
    proj_db_path = Path(os.environ['PROJ_LIB']) / 'proj.db'
    if not proj_db_path.exists():
        raise Exception(f"There must be something wrong with the GDAL installation. proj.db file not found in {os.environ['PROJ_LIB']}")
        
    # Read the tasks that need to be ran in the run_tasks file
    tasks_df = get_tasks(tasks_path)

    # Get the cancel filepath from the config
    cancel_filepath = Path(runner_config['files']['cancel_filepath'])

    # Loop over tasks to run
    for task in tasks_df.itertuples(): 

        # If the cancel file exists, stop processing...
        if cancel_filepath.exists():
            logger.info(f"Cancel file found, so stop: {cancel_filepath}")
            break

        # If the task is not active, skip
        if(task.active == 0):
            continue

        # Format the projectfile_path
        projectfile_path = Path(task.config)
        # If the projectfile path is not absolute, treat is as relative to the tasks csv file...
        if not projectfile_path.is_absolute():
            projectfile_path = tasks_path.parent / projectfile_path

        # Now we are ready to start the action        
        message = f"Start action {task.action} for config {projectfile_path}"
        logger.info(message)
        sendmail(message)
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
            sendmail(subject=message, body=f"Exception: {ex}\n\n {traceback.format_exc()}")
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

    if not runner_config['email'].getboolean('enabled', fallback=False):
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
    main()

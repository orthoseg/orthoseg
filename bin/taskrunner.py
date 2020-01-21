# -*- coding: utf-8 -*-
"""
Process the tasks as configured in the run_tasks_config.csv file.
"""

import argparse
import configparser
from email.message import EmailMessage
import json
import logging
import logging.config
import os
from pathlib import Path
import smtplib

import pandas as pd

def run_tasks(
        tasks_filepath: Path,
        config_filepath: Path,
        logconfig_filepath: Path):

    ##### Init #####
    run_local = True

    # Init logging
    with open(logconfig_filepath, 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    global logger
    logger = logging.getLogger()

    # Read the taskrunner configuration
    global config
    config = configparser.ConfigParser()
    config.read(config_filepath)

    # Read the tasks that need to be ran in the run_tasks file
    run_config_df = get_tasks(tasks_filepath)

    # Loop over tasks to run
    for run_info in run_config_df.itertuples(): 
        if(run_info.active == 0):
            continue

        if run_local:
            # Run local (possible to debug,...)
            if run_info.command == 'bin/run_orthoseg.py':
                try:                 
                    import run_orthoseg as run_orthoseg
                    run_orthoseg.orthoseg_argstr(run_info.argumentstring)
                    sendmail(f"Completed task {run_info.command} {run_info.argumentstring}")
                except Exception as ex:
                    message = f"ERROR in task {run_info.command} {run_info.argumentstring}"
                    sendmail(subject=message, body=f"Exception: {ex}")
                    raise Exception(message) from ex
            else:
                raise Exception(f"Unknown command: {run_info.command}")
        else:
            # Run the tasks by command
            # TODO: support running on remote machine over ssh?
            try:
                # TODO: make the running script cancellable?
                # Remark: this path will depend on the python environment the task 
                # needs to run in
                python_path = r"C:\Tools\anaconda3\envs\orthoseg\python.exe"
                fullcommand = f"{python_path} {run_info.command} {run_info.argumentstring}"
                returncode = os.system(fullcommand)
                if returncode == 0:
                    sendmail(f"Completed task {run_info.command} {run_info.argumentstring}")
                else:
                    raise Exception(f"Error: returncode: {returncode} returned for {fullcommand}")

            except Exception as ex:
                message = f"ERROR in task {run_info.command} {run_info.argumentstring}"
                sendmail(subject=message, body=f"Exception: {ex}")
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
    mandatory_columns = ['command', 'active', 'argumentstring']
    missing_columns = set(mandatory_columns).difference(set(tasks_df.columns))
    if(len(missing_columns) > 0):
        raise Exception(f"Missing column(s) in {filepath}: {missing_columns}")
    
    return tasks_df

def sendmail(
        subject: str, 
        body: str = None,
        stop_on_error: bool = False):

    mail_from = config['email'].get('from', None)
    mail_to = config['email'].get('to', None)
    mail_server = config['email'].get('server', None)
    mail_server_username = config['email'].get('username', None)
    mail_server_password = config['email'].get('password', None)

    # If one of the necessary parameters not provided, log subject
    if(mail_from is None
       or mail_to is None
       or mail_server is None):
        logger.info(f"Mail config not provided to send email with subject: {subject}")
        return

    try:
        # Create message
        # TODO: email adress shouldn't be hardcoded... I suppose
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
    optional.add_argument('-t', '--tasks_csv',
            help='Specify the tasks csv filepath. If not provided, ./taskrunner_tasks.csv or ./taskrunner_tasks_sample.csv is assumed.')
    optional.add_argument('-c', '--configfile',
            help='Specify the file to use  the config files. If not provided, ../config is assumed.')            
    # Add back help         
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')
    
    # Interprete arguments
    args = parser.parse_args()
    tasks_csv=args.tasks_csv

    # Prepare the path to the dir where the config can be found,...
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # If the tasks csv is not provided, fallback to default file names...
    if tasks_csv is not None:
        tasks_filepath = tasks_csv
    else: 
        tasks_filepath = script_dir / 'taskrunner_tasks.csv'

    # If there doesn't exist a 'taskrunner_tasks.csv', we run in sample mode!
    sample_suffix = ''
    if not tasks_filepath.exists():
        sample_suffix = '_sample'
        tasks_filepath = script_dir / f"taskrunner_tasks{sample_suffix}.csv"

    config_filepath = script_dir / f"taskrunner{sample_suffix}.ini"
    logconfig_filepath = script_dir / f"taskrunner_logconfig{sample_suffix}.json"

    run_tasks(
            tasks_filepath=tasks_filepath,
            config_filepath=config_filepath,
            logconfig_filepath=logconfig_filepath)

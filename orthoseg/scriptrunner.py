# -*- coding: utf-8 -*-
"""
Run the scripts in a directory.
"""

import argparse
import configparser
import datetime
from email.message import EmailMessage
import logging
import logging.config
from pathlib import Path
import smtplib
import subprocess
import time
from typing import List, Optional

runner_config = None
logger = logging.getLogger()

def main():
    
    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('-d', '--script_dir',
            help='Directory containing the scripts to run.')
    optional.add_argument('-w', '--watch', action='store_true', default=False,
            help='Watch the directory forever for files getting in it.')
    optional.add_argument('-c', '--config',
            help='Path to a config file with parameters that need to overrule the defaults.')
    
    # Add back help         
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')
    args = parser.parse_args()
    
    ### Init stuff ###
    script_dir = Path(args.script_dir)
    if not script_dir.exists():
        raise Exception(f"script dir {script_dir} does not exist")

    # Load the scriptrunner config
    load_config(args.config)

    error_dir = script_dir / 'error'
    done_dir = script_dir / 'done'
    log_path = script_dir / 'log' / f"{datetime.datetime.now():%Y-%m-%d}_scriptrunner.log"
    init_logging(log_path=log_path)

    # Loop over scripts to be ran
    wait_message_printed = False
    while True:

        # List the scripts in the dir
        script_paths = []
        script_patterns = ['*.bat', '*.sh']
        for script_pattern in script_patterns:
            script_paths.extend(list(script_dir.glob(script_pattern)))

        # If no scripts found, sleep or stop...
        if len(script_paths) == 0:
            if args.watch is False:
                logger.info(f"No scripts found (anymore) in {script_dir}, so stop")
                break
            else:
                if wait_message_printed is False:
                    logger.info(f"No scripts to run in {script_dir}, so watch script dir...")
                    wait_message_printed = True
                time.sleep(10)
                continue

        # Get next script alphabetically
        script_path = sorted(script_paths)[0]
        
        try:
            # Run the script and print output in realtime
            wait_message_printed = False
            logger.info(f"Run script {script_path}")
            cmd = [script_path]

            '''
            process = subprocess.Popen(cmd, shell=True, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                    creationflags=subprocess.CREATE_NO_WINDOW)
            
            keep_running = True
            while keep_running:
                if process.stdout is not None:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        keep_running = False
                    if output:
                        logger.info(output.strip())
                if process.stderr is not None:
                    err = process.stderr.readline()
                    if err == '' and process.poll() is not None:
                        keep_running = False
                    if err:
                        logger.error(err.strip())
            
            # If error code != 0, an error occured
            rc = process.poll()
            '''
            
            result = subprocess.run(cmd)
            rc = result.returncode

            if rc != 0:
                # Script gave an error, so move to error dir
                logger.error(f"Script {script_path} gave error return code: {rc}")
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / script_path.name
                if error_path.exists():
                    error_path.unlink()
                script_path.rename(target=error_path)
            else:
                # Move the script to the done dir 
                done_dir.mkdir(parents=True, exist_ok=True)
                done_path = done_dir / script_path.name
                if done_path.exists():
                    done_path.unlink()
                script_path.rename(target=done_path)

        except Exception as ex:
            logger.exception(f"Error running script {script_path}")

            # If the script still exists, move it to error dir
            if script_path.exists():
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / script_path.name
                if error_path.exists():
                    error_path.unlink()
                script_path.rename(target=error_path)

def load_config(config_path: str) -> configparser.ConfigParser:

    # Load defaults first
    script_dir = Path(__file__).resolve().parent.parent
    config_paths = [script_dir / 'project_defaults.ini']

    # If a config path is specified, this config should overrule the defaults
    if config_path is not None:
        config_paths.append(Path(config_path))

    # Load!
    scriptrunner_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')]},
            allow_no_value=True)
    scriptrunner_config.read(config_paths)

    return scriptrunner_config

def init_logging(log_path: Optional[Path]):
    
    # Make sure the log dir exists
    if log_path is not None and not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
    # Get root logger
    logger = logging.getLogger('')
    
    # Set the general maximum log level...
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    
    # Remove all handlers and add the ones I want again, so a new log file is created for each run
    # Remark: the function removehandler doesn't seem to work?
    logger.handlers = []
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    #ch.setFormatter(logging.Formatter('%(levelname)s|%(name)s|%(message)s'))
    #ch.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    #                                  datefmt='%H:%M:%S,uuu'))
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s',
                                      datefmt='%H:%M:%S')) 
    logger.addHandler(ch)
    
    if log_path is not None:
        fh = logging.FileHandler(filename=str(log_path))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s'))
        logger.addHandler(fh)
        
    return logger

if __name__ == '__main__':
    main()

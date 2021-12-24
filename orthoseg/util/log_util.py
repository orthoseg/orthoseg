# -*- coding: utf-8 -*-
"""
Module with specific helper functions to manage the logging of orthoseg.

TODO: maybe it is cleaner to replace most code here by a config dict?
"""

import json
import logging
import logging.config
import datetime
from pathlib import Path
import tempfile

#-------------------------------------------------------------
# Init logging
#-------------------------------------------------------------

def init_logging_dictConfig(
        logconfig_path: Path = None, 
        logconfig_dict: dict = None,
        log_basedir: Path = None,
        loggername: str = None) -> logging.Logger:
    """
    Initializes the logging based on input in dictConfig format. 
    
    The input can be a dict or a json file.

    The added value of this function is:
        - it will format {iso_datetime} placeholders in handlers.*.filename
        - it will resolve relative file paths in handlers.*.filename to tempdir.

    Args:
        logconfig_path (Path, optional): Json file containing the dictConfig info. Defaults to None.
        logconfig_dict (dict, optional): Dict containing the dictConfig info. Defaults to None.
    """   
    # Init logging
    if logconfig_dict is None:
        if logconfig_path is not None and logconfig_path.exists():
            # Load from json file
            with open(logconfig_path, 'r') as logconfig_file:
                logconfig_dict = json.load(logconfig_file)
        else:
             raise Exception(f"If the logconfig_dict is None, the logconfig_path should point to an existing file: {logconfig_path}")

    # If there are file handlers, replace possible placeholders + make sure log dir exists
    assert logconfig_dict is not None
    for handler in logconfig_dict['handlers']:
        if "filename" in logconfig_dict['handlers'][handler]:
            # Format the filename
            log_path = Path(logconfig_dict['handlers'][handler]['filename'].format(
                    iso_datetime=f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"))

            # If the log_path is a relative path, resolve it to temp dir.
            if not log_path.is_absolute():
                if log_basedir is not None:
                    log_path = log_basedir / log_path
                else:
                    log_path = Path(tempfile.gettempdir()) / log_path
                print(f"Parameter logconfig.handlers.{handler}.filename was relative, so is now resolved to {log_path}")

            logconfig_dict['handlers'][handler]['filename'] = str(log_path)

            # Also make sure the log dir exists
            log_path.parent.mkdir(parents=True, exist_ok=True)

    # Now load the log config
    logging.config.dictConfig(logconfig_dict)

    return logging.getLogger(loggername)

def main_log_init(
        log_dir: Path,
        log_basefilename: str):

    # Check input parameters
    if not log_dir:
        raise Exception(f"Error: log_dir is mandatory!")
    
    # Make sure the log dir exists
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    log_filepath = log_dir / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{log_basefilename}.log"
    fh = logging.FileHandler(filename=str(log_filepath))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s'))
    logger.addHandler(fh)
    
    return logger

def clean_log_dir(
        log_dir: Path,
        nb_logfiles_tokeep: int,
        pattern: str = '*.*'):
    """
    Clean a log dir.

    Args:
        log_dir (Path): dir with log files to clean.
        nb_logfiles_tokeep (int): the number of log files to keep. 
        pattern (str, optional): pattern of the file names of the log files. 
            Defaults to '*.*'.
    """
    # Check input params
    if log_dir is None or log_dir.exists() is False or nb_logfiles_tokeep is None:
        return
    
    # List log files and remove the ones that are too much 
    files = sorted(list(log_dir.glob(pattern)), reverse=True)
    if len(files) > nb_logfiles_tokeep:
        for file_index in range(nb_logfiles_tokeep, len(files)):
            files[file_index].unlink()

# If the script is ran directly...
if __name__ == '__main__':
    raise Exception("Not implemented")
    
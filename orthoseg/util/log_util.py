"""Module with specific helper functions to manage the logging of orthoseg.

TODO: maybe it is cleaner to replace most code here by a config dict?
"""

import datetime
import json
import logging
import logging.config
import tempfile
from pathlib import Path


class LoggingContext:
    """Context handler to temporary log using a specific level/handler."""

    def __init__(self, logger, level=None, handler=None, close=True):
        """Constructor of LoggingContext.

        Args:
            logger (_type_): the logger to use.
            level (_type_, optional): level to use. Defaults to None.
            handler (_type_, optional): handler to use. Defaults to None.
            close (bool, optional): close the handler when context ends.
                Defaults to True.
        """
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        """Enter."""
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        """Exit.

        Args:
            et (_type_): _description_
            ev (_type_): _description_
            tb (_type_): _description_
        """
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


def init_logging_dictConfig(
    logconfig_path: Path | None = None,
    logconfig_dict: dict | None = None,
    log_basedir: Path | None = None,
    loggername: str | None = None,
) -> logging.Logger:
    """Initializes the logging based on input in dictConfig format.

    The input can be a dict or a json file.

    The added value of this function is:
        - it will format {iso_datetime} placeholders in handlers.*.filename
        - it will resolve relative file paths in handlers.*.filename to tempdir.

    Args:
        logconfig_path (Path, optional): Json file containing the dictConfig info.
            Defaults to None.
        logconfig_dict (dict, optional): Dict containing the dictConfig info.
            Defaults to None.
        log_basedir (Path, optional): directory to log to.
        loggername (str, optional): name of the logger.
    """
    # Init logging
    if logconfig_dict is None:
        if logconfig_path is not None and logconfig_path.exists():
            # Load from json file
            with logconfig_path.open() as logconfig_file:
                logconfig_dict = json.load(logconfig_file)
        else:
            raise ValueError(
                "If the logconfig_dict is None, the logconfig_path should point to "
                f"an existing file: {logconfig_path}"
            )

    # If there are file handlers, replace placeholders + make sure log dir exists
    assert logconfig_dict is not None
    for handler in logconfig_dict["handlers"]:
        if "filename" in logconfig_dict["handlers"][handler]:
            # Format the filename
            log_path = Path(
                logconfig_dict["handlers"][handler]["filename"].format(
                    iso_datetime=f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
                )
            )

            # If the log_path is a relative path, resolve it to temp dir.
            if not log_path.is_absolute():
                if log_basedir is not None:
                    log_path = log_basedir / log_path
                else:
                    log_path = Path(tempfile.gettempdir()) / log_path
                print(
                    f"Parameter logconfig.handlers.{handler}.filename was relative, "
                    f"so is now resolved to {log_path}"
                )

            logconfig_dict["handlers"][handler]["filename"] = str(log_path)

            # Also make sure the log dir exists
            log_path.parent.mkdir(parents=True, exist_ok=True)

    # Now load the log config
    logging.config.dictConfig(logconfig_dict)

    return logging.getLogger(loggername)


def main_log_init(log_dir: Path, log_basefilename: str):
    """Initialize logging.

    Args:
        log_dir (Path): directory to log to.
        log_basefilename (str): base file name to use for the logging files.

    Returns:
        logging.Logger: the logger.
    """
    # Check input parameters
    if not log_dir:
        raise ValueError("Error: log_dir is mandatory!")

    # Make sure the log dir exists
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    logger = logging.getLogger("")

    # Set the general maximum log level...
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.flush()
        handler.close()

    # Remove all handlers and add the ones I want again, so a new log file is created
    # for each run
    # Remark: the function removehandler doesn't seem to work?
    logger.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # ch.setFormatter(logging.Formatter('%(levelname)s|%(name)s|%(message)s'))
    # ch.setFormatter(logging.Formatter(
    #     '%(asctime)s|%(levelname)s|%(name)s|%(message)s', datefmt='%H:%M:%S,uuu')
    # )
    ch.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(ch)

    log_filepath = (
        log_dir / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{log_basefilename}.log"
    )
    fh = logging.FileHandler(filename=str(log_filepath))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
    logger.addHandler(fh)

    return logger


def clean_log_dir(log_dir: Path, nb_logfiles_tokeep: int, pattern: str = "*.*"):
    """Clean a log dir.

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
    files = sorted(log_dir.glob(pattern), reverse=True)
    if len(files) > nb_logfiles_tokeep:
        for file_index in range(nb_logfiles_tokeep, len(files)):
            files[file_index].unlink()

# -*- coding: utf-8 -*-
"""
Module with specific helper functions to manage the configuration of orthoseg.
"""

import configparser
import json
import logging
from pathlib import Path
import tempfile
from typing import List, Optional

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Define the chars that cannot be used in codes that are use in filenames.
# Remark: '_' cannot be used because '_' is used as devider to parse filenames, and if
# it is used in codes as well the parsing becomes a lot more difficult.
illegal_chars_in_codes = ["_", ",", ".", "?", ":"]

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def get_config_files(config_path: Path) -> List[Path]:

    # Init
    # First check input param
    config_path = config_path.expanduser()
    if not config_path.exists():
        raise Exception(f"Config file specified does not exist: {config_path}")

    # Collect the config files to use. The "hardcoded" defaults should always
    # be loaded.
    script_dir = Path(__file__).resolve().parent.parent
    config_filepaths = [script_dir / "project_defaults.ini"]

    # Load possible extra config files from the project config file
    basic_projectconfig = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        converters={"list": lambda x: [i.strip() for i in x.split(",")]},
        allow_no_value=True,
    )
    basic_projectconfig.read(config_path)

    default_basedir = config_path.parent
    extra_config_files_to_load = basic_projectconfig["general"].getlist(
        "extra_config_files_to_load"
    )
    if extra_config_files_to_load is not None:
        for config_file in extra_config_files_to_load:
            config_file_formatted = Path(
                config_file.format(
                    task_filepath=config_path, jobs_dir=config_path.parent
                )
            ).expanduser()
            if not config_file_formatted.is_absolute():
                config_file_formatted = (
                    default_basedir / config_file_formatted
                ).resolve()
            config_filepaths.append(Path(config_file_formatted))

    # Finally add the specific project config file...
    config_filepaths.append(config_path)

    return config_filepaths


def read_config_ext(config_paths: List[Path]) -> configparser.ConfigParser:

    # Log config filepaths that don't exist...
    for config_filepath in config_paths:
        if not config_filepath.exists():
            logger.warning(f"config_filepath does not exist: {config_filepath}")

    # Now we are ready to read the entire configuration
    def parse_boolean_ext(input) -> Optional[bool]:
        if input is None:
            return None

        if input in ("True", "true", "1", 1):
            return True
        elif input in ("False", "false", "0", 0):
            return False
        elif input in ("False", "false", "0", 0):
            return None

    def safe_math_eval(string):
        """
        Function to evaluate a mathematical expression safely.
        """
        if string is None:
            return None

        allowed_chars = "0123456789+-*(). /"
        for char in string:
            if char not in allowed_chars:
                raise Exception("Unsafe eval")

        return eval(string)

    def to_path(pathlike: str) -> Optional[Path]:
        if pathlike is None:
            return None
        else:
            if "{tempdir}" in pathlike:
                return Path(pathlike.format(tempdir=tempfile.gettempdir()))
            else:
                return Path(pathlike)

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        converters={
            "list": lambda x: [i.strip() for i in x.split(",")],
            "listint": lambda x: [int(i.strip()) for i in x.split(",")],
            "listfloat": lambda x: [float(i.strip()) for i in x.split(",")],
            "dict": lambda x: None if x is None else json.loads(x),
            "path": lambda x: to_path(x),
            "eval": lambda x: safe_math_eval(x),
            "boolean_ext": lambda x: parse_boolean_ext(x),
        },
        allow_no_value=True,
    )

    config.read(config_paths)
    return config


def as_dict(config: configparser.ConfigParser):
    """
    Converts a ConfigParser object into a dictionary.

    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.
    """
    the_dict = {}
    for section in config.sections():
        the_dict[section] = {}
        for key, val in config.items(section):
            the_dict[section][key] = val
    return the_dict

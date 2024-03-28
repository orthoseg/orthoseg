"""
Module to make it easy to start a validating session.
"""

import argparse
import logging
from pathlib import Path
import shutil
import sys
import traceback
from typing import List


from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import email_helper
from orthoseg.lib import prepare_traindatasets as prep
from orthoseg.util import log_util

# Get a logger...
logger = logging.getLogger(__name__)


def _validate_args(args) -> argparse.Namespace:
    # Interprete arguments
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-c", "--config", type=str, required=True, help="The config file to use"
    )

    # Optional arguments
    optional = parser.add_argument_group("Optional arguments")
    # Add back help
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    optional.add_argument(
        "config_overrules",
        nargs="*",
        help=(
            "Supply any number of config overrules like this: "
            "<section>.<parameter>=<value>"
        ),
    )

    return parser.parse_args(args)


def validate(config_path: Path, config_overrules: List[str] = []):
    """
    Run a validating session for the config specified.

    Args:
        config_path (Path): Path to the config file to use.
        config_overrules (List[str], optional): list of config options that will
            overrule other ways to supply configuration. They should be specified in the
            form of "<section>.<parameter>=<value>". Defaults to [].
    """
    # Init
    # Load the config and save in a bunch of global variables zo it
    # is accessible everywhere
    conf.read_orthoseg_config(config_path, overrules=config_overrules)

    # Init logging
    log_util.clean_log_dir(
        log_dir=conf.dirs.getpath("log_dir"),
        nb_logfiles_tokeep=conf.logging.getint("nb_logfiles_tokeep"),
    )
    global logger
    logger = log_util.main_log_init(conf.dirs.getpath("log_dir"), __name__)

    # Log start
    logger.info(f"Start validate for config {config_path.stem}")
    logger.debug(f"Config used: \n{conf.pformat_config()}")

    try:
        # Now create the train datasets (train, validation, test)
        training_dir, traindata_id = conf.prepare_traindatasets()

        # Send mail that we are starting train
        email_helper.sendmail(f"Start validate for config {config_path.stem}")
        logger.info(
            f"Traindata dir to use is {training_dir}, with traindata_id: {traindata_id}"
        )
    except Exception as ex:
        message = f"ERROR while running validate for task {config_path.stem}"
        logger.exception(message)
        if isinstance(ex, prep.ValidationError):
            message_body = f"Validation error: {ex.to_html()}"
        else:
            message_body = f"Exception: {ex}<br/><br/>{traceback.format_exc()}"
        email_helper.sendmail(subject=message, body=message_body)
        raise Exception(message) from ex
    finally:
        if conf.tmp_dir is not None:
            shutil.rmtree(conf.tmp_dir, ignore_errors=True)


def main():
    """
    Run validate.
    """
    try:
        # Interprete arguments
        args = _validate_args(sys.argv[1:])

        # Run!
        validate(config_path=Path(args.config), config_overrules=args.config_overrules)
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

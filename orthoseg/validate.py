"""Module to make it easy to start a validating session."""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from orthoseg.helpers import config_helper as conf, email_helper
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


def validate(config_path: Path, config_overrules: list[str] = []):
    """Run a validating session for the config specified.

    Args:
        config_path (Path): Path to the config file to use.
        config_overrules (list[str], optional): list of config options that will
            overrule other ways to supply configuration. They should be specified in the
            form of "<section>.<parameter>=<value>". Defaults to [].
    """
    # Init
    # Load the config and save in a bunch of global variables so it
    # is accessible everywhere
    conf.read_orthoseg_config(config_path, overrules=config_overrules)

    # Init logging
    log_util.clean_log_dir(
        log_dir=conf.dirs.getpath("log_dir"),
        nb_logfiles_tokeep=conf.logging_conf.getint("nb_logfiles_tokeep"),
    )
    global logger
    logger = log_util.main_log_init(conf.dirs.getpath("log_dir"), __name__)

    # Log start
    logger.info(f"Start validate for config {config_path.stem}")
    logger.debug(f"Config used: \n{conf.pformat_config()}")

    try:
        # Create the output dir's if they don't exist yet...
        for dir in [
            conf.dirs.getpath("project_dir"),
            conf.dirs.getpath("training_dir"),
        ]:
            if dir and not dir.exists():
                dir.mkdir()

        train_label_infos = conf.get_train_label_infos()
        classes = conf.determine_classes()

        # Now create the train datasets (train, validation, test)
        logger.info("Prepare train, validation and test data")
        training_dir, traindata_id = prep.prepare_traindatasets(
            label_infos=train_label_infos,
            classes=classes,
            image_layers=conf.image_layers,
            training_dir=conf.dirs.getpath("training_dir"),
            labelname_column=conf.train.get("labelname_column"),
            image_pixel_x_size=conf.train.getfloat("image_pixel_x_size"),
            image_pixel_y_size=conf.train.getfloat("image_pixel_y_size"),
            image_pixel_width=conf.train.getint("image_pixel_width"),
            image_pixel_height=conf.train.getint("image_pixel_height"),
            ssl_verify=conf.general["ssl_verify"],
            only_validate=True,
        )

        # Send mail that validate was successful
        email_helper.sendmail(f"Validate for config {config_path.stem} was successful")
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
        conf.remove_tmp_dir()


def main():
    """Run validate."""
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

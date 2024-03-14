"""
Module to make it easy to start a validating session.
"""

import argparse
import logging
from pathlib import Path
import shlex
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


def _validate_argstr(argstr):
    args = shlex.split(argstr)
    _validate_args(args)


def _validate_args(args):
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

    # Interprete arguments
    args = parser.parse_args(args)

    # Run!
    validate(config_path=Path(args.config), config_overrules=args.config_overrules)


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
        # First check if the segment_subject has a valid name
        segment_subject = conf.general["segment_subject"]
        if segment_subject == "MUST_OVERRIDE":
            raise Exception(
                "segment_subject must be overridden in the subject specific config file"
            )
        elif "_" in segment_subject:
            raise Exception(f"segment_subject cannot contain '_': {segment_subject}")

        # Create the output dir's if they don't exist yet...
        for dir in [
            conf.dirs.getpath("project_dir"),
            conf.dirs.getpath("training_dir"),
        ]:
            if dir and not dir.exists():
                dir.mkdir()

        # If the training data doesn't exist yet, create it
        # -------------------------------------------------
        train_label_infos = conf.get_train_label_infos()
        if train_label_infos is None or len(train_label_infos) == 0:
            raise ValueError(
                "No valid label file config found in train.label_datasources or "
                f"with patterns {conf.train.get('labelpolygons_pattern')} and "
                f"{conf.train.get('labellocations_pattern')}"
            )

        # Determine classes
        try:
            classes = conf.train.getdict("classes")

            # If the burn_value property isn't supplied for the classes, add them
            for class_id, (classname) in enumerate(classes):
                if "burn_value" not in classes[classname]:
                    classes[classname]["burn_value"] = class_id
        except Exception as ex:
            raise Exception(
                f"Error reading classes: {conf.train.get('classes')}"
            ) from ex

        # Now create the train datasets (train, validation, test)
        force_model_traindata_id = conf.train.getint("force_model_traindata_id")
        if force_model_traindata_id > -1:
            training_dir = (
                conf.dirs.getpath("training_dir") / f"{force_model_traindata_id:02d}"
            )
            traindata_id = force_model_traindata_id
        else:
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
            )

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
        _validate_args(sys.argv[1:])
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

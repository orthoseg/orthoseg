"""
Module with functions for post-processing prediction masks towards polygons.
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
from orthoseg.lib import postprocess_predictions as postp
import orthoseg.model.model_helper as mh
from orthoseg.util import log_util

# Get a logger...
logger = logging.getLogger(__name__)


def _postprocess_argstr(argstr):
    args = shlex.split(argstr)
    _postprocess_args(args)


def _postprocess_args(args):
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
    postprocess(config_path=Path(args.config), config_overrules=args.config_overrules)


def postprocess(config_path: Path, config_overrules: List[str] = []):
    """
    Postprocess the output of a prediction for the config specified.

    Args:
        config_path (Path): Path to the config file.
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

    # Log start + send email
    message = f"Start postprocess for config {config_path.stem}"
    logger.info(message)
    logger.debug(f"Config used: \n{conf.pformat_config()}")
    email_helper.sendmail(message)

    try:
        # Create base filename of model to use
        # TODO: is force data version the most logical, or rather implement
        #       force weights file or ?
        traindata_id = None
        force_model_traindata_id = conf.train.getint("force_model_traindata_id")
        if force_model_traindata_id is not None and force_model_traindata_id > -1:
            traindata_id = force_model_traindata_id

        # Get the best model that already exists for this train dataset
        trainparams_id = conf.train.getint("trainparams_id")
        best_model = mh.get_best_model(
            model_dir=conf.dirs.getpath("model_dir"),
            segment_subject=conf.general["segment_subject"],
            traindata_id=traindata_id,
            trainparams_id=trainparams_id,
        )
        if best_model is None:
            raise Exception(f"No best model found in {conf.dirs.getpath('model_dir')}")

        # Input file  the "most recent" prediction result dir for this subject
        output_vector_dir = conf.dirs.getpath("output_vector_dir")
        output_vector_name = (
            f"{best_model['basefilename']}_{best_model['epoch']}_"
            f"{conf.predict['image_layer']}"
        )
        output_vector_path = output_vector_dir / f"{output_vector_name}.gpkg"

        # Prepare some parameters for the postprocessing
        nb_parallel = conf.general.getint("nb_parallel")

        dissolve = conf.postprocess.getboolean("dissolve")
        dissolve_tiles_path = conf.postprocess.getpath("dissolve_tiles_path")
        reclassify_query = conf.postprocess.get("reclassify_to_neighbour_query")
        if reclassify_query is not None:
            reclassify_query = reclassify_query.replace("\n", " ")

        simplify_algorithm = conf.postprocess.get("simplify_algorithm")
        simplify_tolerance = conf.postprocess.geteval("simplify_tolerance")
        simplify_lookahead = conf.postprocess.get("simplify_lookahead")
        if simplify_lookahead is not None:
            simplify_lookahead = int(simplify_lookahead)

        # Go!
        postp.postprocess_predictions(
            input_path=output_vector_path,
            output_path=output_vector_path,
            dissolve=dissolve,
            dissolve_tiles_path=dissolve_tiles_path,
            reclassify_to_neighbour_query=reclassify_query,
            simplify_algorithm=simplify_algorithm,
            simplify_tolerance=simplify_tolerance,
            simplify_lookahead=simplify_lookahead,
            nb_parallel=nb_parallel,
            force=False,
        )

        # Log and send mail
        message = f"Completed postprocess for config {config_path.stem}"
        logger.info(message)
        email_helper.sendmail(message)
    except Exception as ex:
        message = f"ERROR while running postprocess for task {config_path.stem}"
        logger.exception(message)
        email_helper.sendmail(
            subject=message, body=f"Exception: {ex}\n\n {traceback.format_exc()}"
        )
        raise Exception(message) from ex
    finally:
        shutil.rmtree(conf.tmp_dir, ignore_errors=True)


def main():
    """
    Run postprocess.
    """
    try:
        _postprocess_args(sys.argv[1:])
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

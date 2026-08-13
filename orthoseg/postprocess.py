"""Module with functions for post-processing prediction masks towards polygons."""

import argparse
import logging
import sys
import traceback
from pathlib import Path

import orthoseg.model.model_helper as mh
from orthoseg.helpers import config_helper as conf, email_helper
from orthoseg.lib import postprocess_predictions as postp
from orthoseg.util import log_util

# Get a logger...
logger = logging.getLogger(__name__)


def _apply_postprocess(input_path: Path, output_path: Path) -> None:
    """Apply postprocessing to the prediction output file using config settings.

    Reads postprocessing config from the [postprocess] section and applies
    the full postprocessing pipeline to the vector file.

    Args:
        input_path (Path): Path to the input vector file to postprocess.
        output_path (Path): Path to the output vector file.
    """
    # Prepare some parameters for the postprocessing
    nb_parallel = conf.general.getint("nb_parallel", -1)

    keep_original_file = conf.postprocess.getboolean("keep_original_file", True)
    keep_intermediary_files = conf.postprocess.getboolean(
        "keep_intermediary_files", False
    )
    dissolve = conf.postprocess.getboolean("dissolve", True)
    dissolve_tiles_path = conf.postprocess.getpath("dissolve_tiles_path")
    reclassify_query = conf.postprocess.get("reclassify_to_neighbour_query")
    if reclassify_query is not None:
        reclassify_query = reclassify_query.replace("\n", " ")

    simplify_algorithm = conf.postprocess.get("simplify_algorithm")
    simplify_tolerance = conf.postprocess.geteval("simplify_tolerance")
    simplify_lookahead = conf.postprocess.get("simplify_lookahead")
    if simplify_lookahead is not None:
        simplify_lookahead = int(simplify_lookahead)
    clip_path = conf.postprocess.getpath("clip_path")
    output_style_path = conf.postprocess.getpath("output_style_path")

    # Apply postprocessing
    postp.postprocess_predictions(
        input_path=input_path,
        output_path=output_path,
        clip_path=clip_path,
        dissolve=dissolve,
        dissolve_tiles_path=dissolve_tiles_path,
        reclassify_to_neighbour_query=reclassify_query,
        simplify_algorithm=simplify_algorithm,
        simplify_tolerance=simplify_tolerance,
        simplify_lookahead=simplify_lookahead,
        output_style_path=output_style_path,
        keep_original_file=keep_original_file,
        keep_intermediary_files=keep_intermediary_files,
        nb_parallel=nb_parallel,
        force=False,
    )


def _postprocess_args(args) -> argparse.Namespace:
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
            "Supply any number of config overrules like this: <section>.<key>=<value>"
        ),
    )

    return parser.parse_args(args)


def postprocess(config_path: Path, config_overrules: list[str] | None = None) -> Path:
    """Postprocess the output of a prediction for the config specified.

    Args:
        config_path (Path): Path to the config file.
        config_overrules (list[str], optional): list of config options that will
            overrule other ways to supply configuration. They should be specified in the
            form of "<section>.<key>=<value>". Defaults to None.

    Returns:
        Path: The path to the postprocessed output file.

    """
    # Init
    # Load the config and save in a bunch of global variables zo it
    # is accessible everywhere
    conf.read_orthoseg_config(config_path, overrules=config_overrules)

    # Init logging
    log_util.clean_log_dir(
        log_dir=conf.dirs.getpath("log_dir"),
        nb_logfiles_tokeep=conf.logging_conf.getint("nb_logfiles_tokeep"),
    )
    global logger  # noqa: PLW0603
    logger = log_util.main_log_init(conf.dirs.getpath("log_dir"), __name__)

    # Log start + send email
    image_layer = conf.predict["image_layer"]
    logger.info(f"Start postprocess for {config_path.stem} on {image_layer}")
    logger.debug(f"Config used: \n{conf.pformat_config()}")
    model_name = None

    try:
        # Create base filename of model to use
        # TODO: is force data version the most logical, or rather implement
        #       force weights file or ?
        traindata_id = None
        force_model_traindata_id = conf.train.getint("force_model_traindata_id")
        if force_model_traindata_id is not None and force_model_traindata_id > -1:
            traindata_id = force_model_traindata_id

        # Get the best model that already exists for this train dataset
        architecture_id = conf.model.getint("architecture_id")
        trainparams_id = conf.train.getint("trainparams_id")
        best_model = mh.get_best_model(
            model_dir=conf.dirs.getpath("model_dir"),
            segment_subject=conf.general["segment_subject"],
            traindata_id=traindata_id,
            architecture_id=architecture_id,
            trainparams_id=trainparams_id,
        )
        if best_model is None:
            raise RuntimeError(
                f"No best model found in {conf.dirs.getpath('model_dir')}"
            )

        model_name = best_model.basefilename
        message = f"Start postprocess for {model_name} on {image_layer}"
        email_helper.sendmail(message)

        # Input file  the "most recent" prediction result dir for this subject
        output_vector_dir = conf.dirs.getpath("output_vector_dir")
        image_layer = conf.predict["image_layer"]
        output_stem = f"{best_model.base_output_name}_{image_layer}"
        output_predict_path = output_vector_dir / f"{output_stem}_predict.gpkg"
        output_postp_path = output_vector_dir / f"{output_stem}.gpkg"

        # Backward compat: fall back to old-style name that had epoch before image_layer
        if not output_predict_path.exists():
            name = f"{best_model.base_output_legacy_name}_{image_layer}.gpkg"
            output_legacy_path = output_vector_dir / name
            if output_legacy_path.exists():
                output_predict_path = output_legacy_path

        # Apply postprocessing
        _apply_postprocess(output_predict_path, output_postp_path)

        # Log and send mail
        message = f"Completed postprocess for {model_name} on {image_layer}"
        logger.info(message)
        email_helper.sendmail(message)

        return output_postp_path

    except Exception as ex:
        if model_name is None:
            model_name = config_path.name
        message = f"ERROR in postprocess for {model_name} on {image_layer}"
        logger.exception(message)
        email_helper.sendmail(
            subject=message, body=f"Exception: {ex}\n\n {traceback.format_exc()}"
        )
        raise RuntimeError(f"{message}: {ex}") from ex
    finally:
        conf.remove_run_tmp_dir()


def main():
    """Run postprocess."""
    try:
        # Interprete arguments
        args = _postprocess_args(sys.argv[1:])

        # Run!
        postprocess(
            config_path=Path(args.config), config_overrules=args.config_overrules
        )

    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

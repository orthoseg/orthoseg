"""
High-level API to run a segmentation.
"""

import argparse
import logging
from pathlib import Path
import pprint
import shlex
import shutil
import sys
import traceback
from typing import List

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import tensorflow as tf

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import email_helper
from orthoseg.lib import predicter
import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh
from orthoseg.util import log_util

# Get a logger...
logger = logging.getLogger(__name__)


def _predict_argstr(argstr):
    args = shlex.split(argstr)
    _predict_args(args)


def _predict_args(args):
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
    predict(config_path=Path(args.config), config_overrules=args.config_overrules)


def predict(config_path: Path, config_overrules: List[str] = []):
    """
    Run a prediction for the config specified.

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

    # Log start + send email
    message = f"Start predict for config {config_path.stem}"
    logger.info(message)
    logger.debug(f"Config used: \n{conf.pformat_config()}")
    email_helper.sendmail(message)

    try:
        # Read some config, and check if values are ok
        image_layer = conf.image_layers[conf.predict["image_layer"]]
        if image_layer is None:
            raise Exception(
                f"image_layer to predict is not specified in config: {image_layer}"
            )
        input_image_dir = conf.dirs.getpath("predict_image_input_dir")
        if not input_image_dir.exists():
            raise Exception(f"input image dir doesn't exist: {input_image_dir}")

        # TODO: add something to delete old data, predictions???

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

        # Check if a model was found
        if best_model is None:
            message = (
                f"No model found in model_dir: {conf.dirs.getpath('model_dir')} for "
                f"traindata_id: {traindata_id}"
            )
            logger.critical(message)
            raise Exception(message)
        else:
            model_weights_filepath = best_model["filepath"]
            logger.info(f"Best model found: {model_weights_filepath}")

        # Load the hyperparams of the model
        # TODO: move the hyperparams filename formatting to get_models...
        hyperparams_path = (
            best_model["filepath"].parent
            / f"{best_model['basefilename']}_hyperparams.json"
        )
        hyperparams = mh.HyperParams(path=hyperparams_path)

        # Validate the image prediction size for the model architecture
        overlap = conf.predict.getint("image_pixels_overlap", 0)
        input_width_pred = conf.predict.getint("image_pixel_width") + 2 * overlap
        input_height_pred = conf.predict.getint("image_pixel_height") + 2 * overlap
        mf.check_image_size(
            architecture=hyperparams.architecture.architecture,
            input_width=input_width_pred,
            input_height=input_height_pred,
        )

        # Prepare output subdir to be used for predictions
        predict_out_subdir = f"{best_model['basefilename']}"
        if trainparams_id > 0:
            predict_out_subdir += f"_{trainparams_id}"
        predict_out_subdir += f"_{best_model['epoch']}"

        # Load model to predict with
        # --------------------------
        model = None
        # Try optimizing model with tensorrt. Not supported on Windows
        # -> disabled for now till I have access again to a linux machine
        """
        if os.name != "nt":
            try:
                # Try import
                from tensorflow.python.compiler.tensorrt import trt_convert as trt

                # Import didn't fail, so optimize model
                logger.info(
                    "Tensorrt is available, so try to create and use optimized model"
                )
                savedmodel_optim_dir = (
                    best_model["filepath"].parent
                    / f"{best_model['filepath'].stem}_optim"
                )
                if not savedmodel_optim_dir.exists():
                    # If base model not yet in savedmodel format
                    savedmodel_dir = (
                        best_model["filepath"].parent / best_model["filepath"].stem
                    )
                    if not savedmodel_dir.exists():
                        logger.info(
                            f"SavedModel format not yet available, so load "
                            f"model + weights from {best_model['filepath']}"
                        )
                        model = mf.load_model(best_model["filepath"], compile=False)
                        logger.info(f"Now save again as savedmodel to {savedmodel_dir}")
                        tf.saved_model.save(model, str(savedmodel_dir))
                        model = None

                    # Now optimize model
                    logger.info(f"Optimize + save model to {savedmodel_optim_dir}")
                    converter = trt.TrtGraphConverterV2(
                        input_saved_model_dir=str(savedmodel_dir),
                        is_dynamic_op=True,
                        precision_mode="FP16",
                    )
                    converter.convert()
                    converter.save(savedmodel_optim_dir)

                logger.info(
                    f"Load optimized model + weights from {savedmodel_optim_dir}"
                )
                model = tf.keras.models.load_model(str(savedmodel_optim_dir))

            except ImportError:
                logger.info("Tensorrt is not available, so load unoptimized model")
            except Exception as ex:
                logger.info(
                    "An error occured trying to use tensorrt, "
                    f"so load unoptimized model. Error: {ex}"
                )
        """

        # If model isn't loaded yet... load!
        if model is None:
            model = mf.load_model(best_model["filepath"], compile=False)

        # Prepare the model for predicting
        nb_gpu = len(tf.config.experimental.list_physical_devices("GPU"))
        batch_size = conf.predict.getint("batch_size")
        if nb_gpu <= 1:
            model_for_predict = model
            logger.info(f"Predict using single GPU or CPU, with nb_gpu: {nb_gpu}")
        else:
            # If multiple GPU's available, create multi_gpu_model
            try:
                model_for_predict = model
                logger.warning(
                    "Predict using multiple GPUs NOT IMPLEMENTED AT THE MOMENT"
                )

                # logger.info(
                #     f"Predict using multiple GPUs: {nb_gpu}, batch size becomes: "
                #     f"{batch_size*nb_gpu}"
                # )
                # batch_size *= nb_gpu
            except ValueError:
                logger.info("Predict using single GPU or CPU")
                model_for_predict = model

        # Prepare params for the inline postprocessing of the prediction
        min_probability = conf.predict.getfloat("min_probability")
        postprocess = {}
        simplify_algorithm = conf.predict.get("simplify_algorithm")
        if simplify_algorithm is not None and simplify_algorithm != (""):
            postprocess["simplify"] = {}
            simplify = postprocess["simplify"]

            simplify["simplify_algorithm"] = simplify_algorithm
            simplify["simplify_tolerance"] = conf.predict.geteval("simplify_tolerance")
            simplify["simplify_lookahead"] = conf.predict.getint("simplify_lookahead")
            simplify["simplify_topological"] = conf.predict.getboolean_ext(
                "simplify_topological"
            )
        postprocess["filter_background_modal_size"] = conf.predict.getint(
            "filter_background_modal_size"
        )
        query = conf.predict.get("reclassify_to_neighbour_query")
        if query is not None:
            query = query.replace("\n", " ")
        postprocess["reclassify_to_neighbour_query"] = query
        logger.info(f"Inline postprocessing:\n{pprint.pformat(postprocess)}")

        # Prepare the output dirs/paths
        predict_output_dir = Path(
            f"{str(conf.dirs.getpath('predict_image_output_basedir'))}_"
            f"{predict_out_subdir}"
        )
        output_vector_dir = conf.dirs.getpath("output_vector_dir")
        output_vector_name = (
            f"{best_model['basefilename']}_{best_model['epoch']}_"
            f"{conf.predict['image_layer']}"
        )
        output_vector_path = output_vector_dir / f"{output_vector_name}.gpkg"

        # Predict for entire dataset
        nb_parallel = conf.general.getint("nb_parallel")
        predicter.predict_dir(
            model=model_for_predict,
            input_image_dir=input_image_dir,
            output_image_dir=predict_output_dir,
            output_vector_path=output_vector_path,
            classes=hyperparams.architecture.classes,
            min_probability=min_probability,
            postprocess=postprocess,
            border_pixels_to_ignore=conf.predict.getint("image_pixels_overlap"),
            projection_if_missing=image_layer["projection"],
            input_mask_dir=None,
            batch_size=batch_size,
            evaluate_mode=False,
            cancel_filepath=conf.files.getpath("cancel_filepath"),
            nb_parallel_postprocess=nb_parallel,
            max_prediction_errors=conf.predict.getint("max_prediction_errors"),
        )

        # Log and send mail
        message = f"Completed predict for config {config_path.stem}"
        logger.info(message)
        email_helper.sendmail(message)
    except Exception as ex:
        message = f"ERROR while running predict for task {config_path.stem}"
        logger.exception(message)
        email_helper.sendmail(
            subject=message, body=f"Exception: {ex}\n\n {traceback.format_exc()}"
        )
        raise Exception(message) from ex
    finally:
        shutil.rmtree(conf.tmp_dir, ignore_errors=True)


def main():
    """
    Run predict.
    """
    try:
        _predict_args(sys.argv[1:])
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

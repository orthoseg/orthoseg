"""High-level API to run a segmentation."""

import argparse
import logging
import pprint
import sys
import traceback
from pathlib import Path
from typing import Any

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import tensorflow as tf

import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh
from orthoseg.helpers import config_helper as conf, email_helper
from orthoseg.lib import cleanup, predicter
from orthoseg.util import log_util

# Get a logger...
logger = logging.getLogger(__name__)


def _predict_args(args) -> argparse.Namespace:
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


def predict(config_path: Path, config_overrules: list[str] = []):
    """Run a prediction for the config specified.

    Args:
        config_path (Path): Path to the config file to use.
        config_overrules (list[str], optional): list of config options that will
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
        nb_logfiles_tokeep=conf.logging_conf.getint("nb_logfiles_tokeep"),
    )
    global logger
    logger = log_util.main_log_init(conf.dirs.getpath("log_dir"), __name__)
    logger.info(f"Start predict for config {config_path.stem}")
    logger.debug(f"Config used: \n{conf.pformat_config()}")

    try:
        # Read some config, and check if values are ok
        image_layer = conf.predict["image_layer"]
        image_layer_info = conf.image_layers[image_layer]
        if image_layer_info is None:
            raise Exception(
                f"image_layer to predict is not specified in config: {image_layer_info}"
            )
        input_image_dir = conf.dirs.getpath("predict_image_input_dir")

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
        postprocess: dict[str, Any] = {}
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
            f"{conf.dirs['predict_image_output_basedir']}_{predict_out_subdir}"
        )
        output_vector_dir = conf.dirs.getpath("output_vector_dir")
        output_vector_name = (
            f"{best_model['basefilename']}_{best_model['epoch']}_{image_layer}"
        )
        output_vector_path = output_vector_dir / f"{output_vector_name}.gpkg"

        # Start predict for entire dataset
        # --------------------------------
        # Send email
        email_helper.sendmail(f"Start predict for {config_path.stem} on {image_layer}")

        # Check if the layer to predict is configured in the image_layers
        predict_layer = conf.predict["image_layer"]
        if predict_layer not in conf.image_layers:
            raise ValueError(f"{predict_layer=} is not configured in image_layers")
        image_layer = conf.image_layers[predict_layer]

        use_cache = image_layer.get("use_cache", "yes")
        if use_cache == "ifavailable":
            use_cache = (
                "yes"
                if input_image_dir is not None and input_image_dir.exists()
                else "no"
            )

        # Predict!
        if use_cache == "yes":
            # Predict from a directory with (cached) images
            predicter.predict_dir(
                model=model_for_predict,
                input_image_dir=input_image_dir,
                output_image_dir=predict_output_dir,
                output_vector_path=output_vector_path,
                classes=hyperparams.architecture.classes,
                min_probability=min_probability,
                postprocess=postprocess,
                border_pixels_to_ignore=conf.predict.getint("image_pixels_overlap"),
                projection_if_missing=image_layer_info["projection"],
                input_mask_dir=None,
                batch_size=batch_size,
                evaluate_mode=False,
                cancel_filepath=conf.files.getpath("cancel_filepath"),
                nb_parallel_postprocess=conf.general.getint("nb_parallel"),
                max_prediction_errors=conf.predict.getint("max_prediction_errors"),
            )
        else:
            # Predict directly from an image/layer
            predicter.predict_layer(
                model=model_for_predict,
                image_layer=image_layer,
                image_pixel_width=conf.predict.getint("image_pixel_width"),
                image_pixel_height=conf.predict.getint("image_pixel_height"),
                image_pixel_x_size=conf.predict.getfloat("image_pixel_x_size"),
                image_pixel_y_size=conf.predict.getfloat("image_pixel_y_size"),
                image_pixels_overlap=conf.predict.getint("image_pixels_overlap", 0),
                output_image_dir=predict_output_dir,
                output_vector_path=output_vector_path,
                classes=hyperparams.architecture.classes,
                min_probability=min_probability,
                postprocess=postprocess,
                projection_if_missing=image_layer_info["projection"],
                input_mask_dir=None,
                batch_size=batch_size,
                evaluate_mode=False,
                cancel_filepath=conf.files.getpath("cancel_filepath"),
                ssl_verify=conf.general["ssl_verify"],
                nb_parallel_postprocess=conf.general.getint("nb_parallel"),
                max_prediction_errors=conf.predict.getint("max_prediction_errors"),
            )

        # Log and send mail
        message = f"Completed predict for {config_path.stem} on {image_layer}"
        logger.info(message)
        email_helper.sendmail(message)

        # Cleanup old data
        cleanup.clean_models(
            model_dir=conf.dirs.getpath("model_dir"),
            versions_to_retain=conf.cleanup.getint("model_versions_to_retain"),
            simulate=conf.cleanup.getboolean("simulate"),
        )
        cleanup.clean_training_data_directories(
            training_dir=conf.dirs.getpath("training_dir"),
            versions_to_retain=conf.cleanup.getint("training_versions_to_retain"),
            simulate=conf.cleanup.getboolean("simulate"),
        )
        cleanup.clean_predictions(
            output_vector_dir=conf.dirs.getpath("output_vector_dir"),
            versions_to_retain=conf.cleanup.getint("prediction_versions_to_retain"),
            simulate=conf.cleanup.getboolean("simulate"),
        )
    except Exception as ex:
        message = f"ERROR in predict for {config_path.stem} on {image_layer}"
        logger.exception(message)
        email_helper.sendmail(
            subject=message, body=f"Exception: {ex}\n\n {traceback.format_exc()}"
        )
        raise RuntimeError(message) from ex
    finally:
        conf.remove_run_tmp_dir()


def main():
    """Run predict."""
    try:
        # Interprete arguments
        args = _predict_args(sys.argv[1:])

        # Run!
        predict(config_path=Path(args.config), config_overrules=args.config_overrules)
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

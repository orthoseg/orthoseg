"""Module to make it easy to start a training session."""

import argparse
import gc
import logging
import sys
import traceback
from pathlib import Path

from tensorflow import keras as kr

from orthoseg.helpers import config_helper as conf, email_helper
from orthoseg.lib import predicter, prepare_traindatasets as prep, trainer
from orthoseg.model import model_factory as mf, model_helper as mh
from orthoseg.util import log_util

# Get a logger...
logger = logging.getLogger(__name__)


def _train_args(args) -> argparse.Namespace:
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


def train(config_path: Path, config_overrules: list[str] | None = None):
    """Run a training session for the config specified.

    Args:
        config_path (Path): Path to the config file to use.
        config_overrules (list[str], optional): list of config options that will
            overrule other ways to supply configuration. They should be specified in the
            form of "<section>.<parameter>=<value>". Defaults to None.
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

    # Log start
    logger.info(f"Start train for config {config_path.stem}")
    logger.debug(f"Config used: \n{conf.pformat_config()}")

    try:
        # Create the output dir's if they don't exist yet...
        for output_dir in [
            conf.dirs.getpath("project_dir"),
            conf.dirs.getpath("training_dir"),
        ]:
            if output_dir and not output_dir.exists():
                output_dir.mkdir()

        train_label_infos = conf.get_train_label_infos()

        # Determine the projection of (the first) train layer... it will be used for all
        train_image_layer = train_label_infos[0].image_layer
        train_projection = conf.image_layers[train_image_layer]["projection"]

        classes = conf.determine_classes()

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
        email_helper.sendmail(f"Start train for {config_path.stem} ({traindata_id=})")
        logger.info(
            f"Traindata dir to use is {training_dir}, with traindata_id: {traindata_id}"
        )
        traindata_dir = training_dir / "train"
        validationdata_dir = training_dir / "validation"
        testdata_dir = training_dir / "test"

        # Check if training is needed
        # Get hyper parameters from the config
        # TODO: activation_function should probably not be specified!!!!!!
        architectureparams = mh.ArchitectureParams(
            architecture=conf.model["architecture"],
            classes=list(classes),
            nb_channels=conf.model.getint("nb_channels"),
            architecture_id=conf.model.getint("architecture_id"),
            activation_function="softmax",
        )
        trainparams = mh.TrainParams(
            trainparams_id=conf.train.getint("trainparams_id"),
            image_augmentations=conf.train.getdict("image_augmentations"),
            mask_augmentations=conf.train.getdict("mask_augmentations"),
            class_weights=[classes[classname]["weight"] for classname in classes],
            batch_size=conf.train.getint("batch_size_fit"),
            optimizer=conf.train.get("optimizer"),
            optimizer_params=conf.train.getdict("optimizer_params"),
            loss_function=conf.train.get("loss_function"),
            monitor_metric=conf.train.get("monitor_metric"),
            monitor_metric_mode=conf.train.get("monitor_metric_mode"),
            save_format=conf.train.get("save_format"),
            save_best_only=conf.train.getboolean("save_best_only"),
            save_min_accuracy=conf.train.getfloat("save_min_accuracy"),
            nb_epoch=conf.train.getint("max_epoch"),
            nb_epoch_with_freeze=conf.train.getint("nb_epoch_with_freeze"),
            earlystop_patience=conf.train.getint("earlystop_patience"),
            earlystop_monitor_metric=conf.train.get("earlystop_monitor_metric"),
            earlystop_monitor_metric_mode=conf.train.get(
                "earlystop_monitor_metric_mode"
            ),
            log_tensorboard=conf.train.getboolean("log_tensorboard"),
            log_csv=conf.train.getboolean("log_csv"),
        )

        # Check if there exists already a model for this train dataset + hyperparameters
        model_dir = conf.dirs.getpath("model_dir")
        segment_subject = conf.general["segment_subject"]
        best_model_curr_train_version = mh.get_best_model(
            model_dir=model_dir,
            segment_subject=segment_subject,
            traindata_id=traindata_id,
            architecture_id=architectureparams.architecture_id,
            trainparams_id=trainparams.trainparams_id,
        )

        # Determine if training is needed,...
        resume_train = conf.train.getboolean("resume_train")
        if resume_train is False:
            # If no (best) model found, training needed!
            if best_model_curr_train_version is None:
                train_needed = True
            elif conf.train.getboolean("force_train") is True:
                train_needed = True
            else:
                logger.info(
                    "JUST PREDICT, no training: resume_train is false + model found"
                )
                train_needed = False
        else:  # noqa: PLR5501
            # We want to preload an existing model and models were found
            if best_model_curr_train_version is not None:
                logger.info(
                    "PRELOAD model + continue TRAINING: "
                    f"{best_model_curr_train_version['filename']}"
                )
                train_needed = True
            else:
                message = "STOP: preload_existing_model is true but no model was found!"
                logger.error(message)
                raise Exception(message)

        # Train!!!
        min_probability = conf.predict.getfloat("min_probability")
        nb_parallel_postprocess = conf.general.getint("nb_parallel")

        if train_needed is True:
            # If a model already exists, use it to predict (possibly new) training and
            # validation dataset. This way it is possible to have a quick check on
            # errors in (new) added labels in the datasets.

            # Get the current best model that already exists for this subject
            best_recent_model = mh.get_best_model(
                model_dir=model_dir,
                segment_subject=segment_subject,
                architecture_id=architectureparams.architecture_id,
                trainparams_id=trainparams.trainparams_id,
            )
            if best_recent_model is not None:
                try:
                    # TODO: move the hyperparams filename formatting to get_models...
                    logger.info(
                        f"Load model + weights from {best_recent_model['filepath']}"
                    )
                    best_model = mf.load_model(
                        best_recent_model["filepath"], compile_model=False
                    )
                    best_hyperparams_path = (
                        best_recent_model["filepath"].parent
                        / f"{best_recent_model['basefilename']}_hyperparams.json"
                    )
                    best_hyperparams = mh.HyperParams(path=best_hyperparams_path)
                    logger.info("Loaded model, weights and params")

                    # Prepare output subdir to be used for predictions
                    predict_out_subdir = best_recent_model["filepath"].stem

                    # Predict training dataset
                    predicter.predict_dir(
                        model=best_model,
                        input_image_dir=traindata_dir / "image",
                        output_image_dir=traindata_dir / predict_out_subdir,
                        output_vector_path=None,
                        projection_if_missing=train_projection,
                        input_mask_dir=traindata_dir / "mask",
                        batch_size=conf.train.getint("batch_size_predict"),
                        evaluate_mode=True,
                        classes=best_hyperparams.architecture.classes,
                        min_probability=min_probability,
                        cancel_filepath=conf.files.getpath("cancel_filepath"),
                        nb_parallel_postprocess=nb_parallel_postprocess,
                        max_prediction_errors=conf.predict.getint(
                            "max_prediction_errors"
                        ),
                    )

                    # Predict validation dataset
                    predicter.predict_dir(
                        model=best_model,
                        input_image_dir=validationdata_dir / "image",
                        output_image_dir=validationdata_dir / predict_out_subdir,
                        output_vector_path=None,
                        projection_if_missing=train_projection,
                        input_mask_dir=validationdata_dir / "mask",
                        batch_size=conf.train.getint("batch_size_predict"),
                        evaluate_mode=True,
                        classes=best_hyperparams.architecture.classes,
                        min_probability=min_probability,
                        cancel_filepath=conf.files.getpath("cancel_filepath"),
                        nb_parallel_postprocess=nb_parallel_postprocess,
                        max_prediction_errors=conf.predict.getint(
                            "max_prediction_errors"
                        ),
                    )
                    del best_model
                except Exception as ex:
                    logger.warning(f"Exception trying to predict with old model: {ex}")

            # Now we can really start training
            logger.info("Start training")
            model_preload_filepath = None
            if best_model_curr_train_version is not None:
                model_preload_filepath = best_model_curr_train_version["filepath"]
            elif conf.train.getboolean("preload_with_previous_traindata"):
                best_model_for_architecture = mh.get_best_model(
                    model_dir=model_dir, segment_subject=segment_subject
                )
                if best_model_for_architecture is not None:
                    model_preload_filepath = best_model_for_architecture["filepath"]

            # Combine all hyperparameters in hyperparams object
            hyperparams = mh.HyperParams(
                architecture=architectureparams, train=trainparams
            )

            trainer.train(
                traindata_dir=traindata_dir,
                validationdata_dir=validationdata_dir,
                model_save_dir=model_dir,
                segment_subject=segment_subject,
                traindata_id=traindata_id,
                hyperparams=hyperparams,
                model_preload_filepath=model_preload_filepath,
                image_width=conf.train.getint("image_pixel_width"),
                image_height=conf.train.getint("image_pixel_height"),
                save_augmented_subdir=conf.train.get("save_augmented_subdir"),
            )

            # Now get the best model found during training
            best_model_curr_train_version = mh.get_best_model(
                model_dir=model_dir,
                segment_subject=segment_subject,
                traindata_id=traindata_id,
            )

        # Assert to evade typing warnings
        assert best_model_curr_train_version is not None

        # Now predict on the train,... data
        logger.info(
            "PREDICT test data with best model: "
            f"{best_model_curr_train_version['filename']}"
        )

        # Load prediction model...
        logger.info(
            f"Load model + weights from {best_model_curr_train_version['filepath']}"
        )
        model = mf.load_model(
            best_model_curr_train_version["filepath"], compile_model=False
        )
        logger.info("Loaded model + weights")

        # Prepare output subdir to be used for predictions
        predict_out_subdir = best_model_curr_train_version["filepath"].stem

        # Predict training dataset
        predicter.predict_dir(
            model=model,
            input_image_dir=traindata_dir / "image",
            output_image_dir=traindata_dir / predict_out_subdir,
            output_vector_path=None,
            projection_if_missing=train_projection,
            input_mask_dir=traindata_dir / "mask",
            batch_size=conf.train.getint("batch_size_predict"),
            evaluate_mode=True,
            classes=classes,
            min_probability=min_probability,
            cancel_filepath=conf.files.getpath("cancel_filepath"),
            nb_parallel_postprocess=nb_parallel_postprocess,
            max_prediction_errors=conf.predict.getint("max_prediction_errors"),
        )

        # Predict validation dataset
        predicter.predict_dir(
            model=model,
            input_image_dir=validationdata_dir / "image",
            output_image_dir=validationdata_dir / predict_out_subdir,
            output_vector_path=None,
            projection_if_missing=train_projection,
            input_mask_dir=validationdata_dir / "mask",
            batch_size=conf.train.getint("batch_size_predict"),
            evaluate_mode=True,
            classes=classes,
            min_probability=min_probability,
            cancel_filepath=conf.files.getpath("cancel_filepath"),
            nb_parallel_postprocess=nb_parallel_postprocess,
            max_prediction_errors=conf.predict.getint("max_prediction_errors"),
        )

        # Predict test dataset, if it exists
        if testdata_dir is not None and testdata_dir.exists():
            predicter.predict_dir(
                model=model,
                input_image_dir=testdata_dir / "image",
                output_image_dir=testdata_dir / predict_out_subdir,
                output_vector_path=None,
                projection_if_missing=train_projection,
                input_mask_dir=testdata_dir / "mask",
                batch_size=conf.train.getint("batch_size_predict"),
                evaluate_mode=True,
                classes=classes,
                min_probability=min_probability,
                cancel_filepath=conf.files.getpath("cancel_filepath"),
                nb_parallel_postprocess=nb_parallel_postprocess,
                max_prediction_errors=conf.predict.getint("max_prediction_errors"),
                no_images_ok=True,
            )

        # Predict extra test dataset with random images in the roi, to add to
        # train and/or validation dataset if inaccuracies are found
        # -> this is very useful to find false positives to improve the datasets
        if conf.dirs.getpath("predictsample_image_input_dir").exists():
            predicter.predict_dir(
                model=model,
                input_image_dir=conf.dirs.getpath("predictsample_image_input_dir"),
                output_image_dir=conf.dirs.getpath("predictsample_image_output_basedir")
                / predict_out_subdir,
                output_vector_path=None,
                projection_if_missing=train_projection,
                batch_size=conf.train.getint("batch_size_predict"),
                evaluate_mode=True,
                classes=classes,
                min_probability=min_probability,
                cancel_filepath=conf.files.getpath("cancel_filepath"),
                nb_parallel_postprocess=nb_parallel_postprocess,
                max_prediction_errors=conf.predict.getint("max_prediction_errors"),
            )

        # Free resources...
        logger.debug("Free resources")
        if model is not None:
            del model
        kr.backend.clear_session()
        gc.collect()

        # Log and send mail
        message = f"Completed train for {config_path.stem} ({traindata_id=})"
        logger.info(message)
        email_helper.sendmail(message)
    except Exception as ex:
        message = f"ERROR in train for {config_path.stem}"
        logger.exception(message)
        if isinstance(ex, prep.ValidationError):
            message_body = f"Validation error: {ex.to_html()}"
        else:
            message_body = f"Exception: {ex}<br/><br/>{traceback.format_exc()}"
        email_helper.sendmail(subject=message, body=message_body)
        raise RuntimeError(f"{message}: {ex}") from ex
    finally:
        conf.remove_run_tmp_dir()


def main():
    """Run train."""
    try:
        # Interprete arguments
        args = _train_args(sys.argv[1:])

        # Run!
        train(config_path=Path(args.config), config_overrules=args.config_overrules)
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

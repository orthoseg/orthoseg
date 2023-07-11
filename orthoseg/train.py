# -*- coding: utf-8 -*-
"""
Module to make it easy to start a training session.
"""

import argparse
import gc
import logging
import os
from pathlib import Path
import re
import shlex
import sys
import traceback
from typing import List

from tensorflow import keras as kr

# orthoseg is higher in dir hierarchy, add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import email_helper
from orthoseg.lib import prepare_traindatasets as prep
from orthoseg.lib import predicter
from orthoseg.lib import trainer
from orthoseg.model import model_factory as mf
from orthoseg.model import model_helper as mh
from orthoseg.util import log_util

# -------------------------------------------------------------
# First define/init general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def _train_argstr(argstr):
    args = shlex.split(argstr)
    _train_args(args)


def _train_args(args):

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

    # Interprete arguments
    args = parser.parse_args(args)

    # Run!
    train(config_path=Path(args.config))


def train(config_path: Path):
    """
    Run a training session for the config specified.

    Args:
        config_path (Path): Path to the config file to use.
    """
    # Init
    # Load the config and save in a bunch of global variables zo it
    # is accessible everywhere
    conf.read_orthoseg_config(config_path)

    # Init logging
    log_util.clean_log_dir(
        log_dir=conf.dirs.getpath("log_dir"),
        nb_logfiles_tokeep=conf.logging.getint("nb_logfiles_tokeep"),
    )
    global logger
    logger = log_util.main_log_init(conf.dirs.getpath("log_dir"), __name__)

    # Log start
    logger.info(f"Start train for config {config_path.stem}")
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
        # Get the label input info
        label_files_dict = conf.train.getdict("label_datasources", None)
        label_infos = []
        if label_files_dict is not None:
            for label_file_key in label_files_dict:
                label_file = label_files_dict[label_file_key]
                # Add as LabelInfo objects to list
                label_infos.append(
                    prep.LabelInfo(
                        locations_path=Path(label_file["locations_path"]),
                        polygons_path=Path(label_file["data_path"]),
                        image_layer=label_file["image_layer"],
                    )
                )
            if label_infos is None or len(label_infos) == 0:
                raise Exception(
                    "label_datasources is defined in config but contains invalid info!"
                )
        else:
            # Search for the files based on the file name patterns...
            labelpolygons_pattern = conf.train.getpath("labelpolygons_pattern")
            labellocations_pattern = conf.train.getpath("labellocations_pattern")
            label_infos = _search_label_files(
                labelpolygons_pattern, labellocations_pattern
            )
            if label_infos is None or len(label_infos) == 0:
                raise Exception(
                    f"No label files found with patterns {labellocations_pattern} and "
                    f"{labelpolygons_pattern}"
                )

        # Determine the projection of (the first) train layer... it will be used for all
        train_image_layer = label_infos[0].image_layer
        train_projection = conf.image_layers[train_image_layer]["projection"]

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
                label_infos=label_infos,
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
        email_helper.sendmail(f"Start train for config {config_path.stem}")
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
            classes=[classname for classname in classes],
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
        else:
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
                        best_recent_model["filepath"], compile=False
                    )
                    best_hyperparams_path = (
                        best_recent_model["filepath"].parent
                        / f"{best_recent_model['basefilename']}_hyperparams.json"
                    )
                    best_hyperparams = mh.HyperParams(path=best_hyperparams_path)
                    logger.info("Loaded model, weights and params")

                    # Prepare output subdir to be used for predictions
                    predict_out_subdir, _ = os.path.splitext(
                        best_recent_model["filename"]
                    )

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
                        max_prediction_errors=conf.predict.getint(
                            "max_prediction_errors"
                        ),
                    )
                    del best_model
                except Exception as ex:
                    logger.warn(f"Exception trying to predict with old model: {ex}")

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
        model = mf.load_model(best_model_curr_train_version["filepath"], compile=False)
        logger.info("Loaded model + weights")

        # Prepare output subdir to be used for predictions
        predict_out_subdir, _ = os.path.splitext(
            best_model_curr_train_version["filename"]
        )

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
                max_prediction_errors=conf.predict.getint("max_prediction_errors"),
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
                max_prediction_errors=conf.predict.getint("max_prediction_errors"),
            )

        # Free resources...
        logger.debug("Free resources")
        if model is not None:
            del model
        kr.backend.clear_session()
        gc.collect()

        # Log and send mail
        message = f"Completed train for config {config_path.stem}"
        logger.info(message)
        email_helper.sendmail(message)
    except Exception as ex:
        message = f"ERROR while running train for task {config_path.stem}"
        logger.exception(message)
        if isinstance(ex, prep.ValidationError):
            message_body = f"Validation error: {ex.to_html()}"
        else:
            message_body = f"Exception: {ex}<br/><br/>{traceback.format_exc()}"
        email_helper.sendmail(subject=message, body=message_body)
        raise Exception(message) from ex


def _search_label_files(
    labelpolygons_pattern: Path, labellocations_pattern: Path
) -> List[prep.LabelInfo]:

    if not labelpolygons_pattern.parent.exists():
        raise ValueError(f"Label dir {labelpolygons_pattern.parent} doesn't exist")
    if not labellocations_pattern.parent.exists():
        raise ValueError(f"Label dir {labellocations_pattern.parent} doesn't exist")

    label_infos = []
    labelpolygons_pattern_searchpath = Path(
        str(labelpolygons_pattern).format(image_layer="*")
    )
    labelpolygons_paths = list(
        labelpolygons_pattern_searchpath.parent.glob(
            labelpolygons_pattern_searchpath.name
        )
    )
    labellocations_pattern_searchpath = Path(
        str(labellocations_pattern).format(image_layer="*")
    )
    labellocations_paths = list(
        labellocations_pattern_searchpath.parent.glob(
            labellocations_pattern_searchpath.name
        )
    )

    # Loop through all labellocation files
    for labellocations_path in labellocations_paths:
        tokens = _unformat(labellocations_path.stem, labellocations_pattern.stem)
        if "image_layer" not in tokens:
            raise ValueError(
                f"image_layer token not found in {labellocations_path} using pattern "
                f"{labellocations_pattern}"
            )
        image_layer = tokens["image_layer"]

        # Look for the matching (= same image_layer) data file
        found = False
        for labelpolygons_path in labelpolygons_paths:
            tokens = _unformat(labelpolygons_path.stem, labelpolygons_pattern.stem)
            if "image_layer" not in tokens:
                raise ValueError(
                    f"image_layer token not found in {labelpolygons_path} using "
                    f"pattern {labelpolygons_pattern}"
                )

            if tokens["image_layer"] == image_layer:
                found = True
                label_infos.append(
                    prep.LabelInfo(
                        locations_path=labellocations_path,
                        polygons_path=labelpolygons_path,
                        image_layer=image_layer,
                    )
                )
        if found is False:
            raise ValueError(f"No matching data file found for {labellocations_path}")

    return label_infos


def _unformat(string: str, pattern: str) -> dict:
    regex = re.sub(r"{(.+?)}", r"(?P<_\1>.+)", pattern)
    regex_result = re.search(regex, string)
    if regex_result is not None:
        values = list(regex_result.groups())
        keys = re.findall(r"{(.+?)}", pattern)
        _dict = dict(zip(keys, values))
        return _dict
    else:
        raise Exception(f"Error: pattern {pattern} not found in {string}")


def main():
    try:
        _train_args(sys.argv[1:])
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

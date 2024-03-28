"""
Automatic cleanup of 'old' models, predictions and training data directories.
"""

from glob import glob
import logging
import os
import shutil
from typing import List
from orthoseg.helpers import config_helper as conf
from pathlib import Path

from orthoseg.model import model_helper
from orthoseg.util import log_util
from orthoseg.util.data import aidetection_info


def Init_logging(
    config_path: Path,
    config_overrules: List[str] = [],
):
    """
    Init.

    Args:
        config_path (Path): Path to the models directory
        config_overrules (List[str], optional): _description_. Defaults to [].
    """
    # Init logging
    conf.read_orthoseg_config(config_path, overrules=config_overrules)

    log_util.clean_log_dir(
        log_dir=conf.dirs.getpath("log_dir"),
        nb_logfiles_tokeep=conf.logging.getint("nb_logfiles_tokeep"),
    )
    global logger
    logger = log_util.main_log_init(conf.dirs.getpath("log_dir"), __name__)


def clean_old_data(
    path: Path,
    versions_to_retain: int,
    simulate: bool,
):
    """
    Cleanup models, training data directories and predictions.

    Args:
        path (Path): Path to the directories
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted
    """
    clean_models(path=path, versions_to_retain=versions_to_retain, simulate=simulate)
    clean_training_data_directories(
        path=path, versions_to_retain=versions_to_retain, simulate=simulate
    )
    clean_predictions(
        path=path, versions_to_retain=versions_to_retain, simulate=simulate
    )


def clean_models(
    path: Path,
    versions_to_retain: int,
    simulate: bool,
):
    """
    Cleanup models.

    Args:
        path (Path): Path to the directories
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted
    """
    # Get a logger...
    logger = logging.getLogger(__name__)

    # Log start
    # TODO: nakijken of dit klopt? path.stem
    logger.info(f"Start cleanup models for config {path.stem}")

    # path = conf.dirs.getpath("model_dir")
    # versions_to_retain = conf.cleanup.getint("model_versions_to_retain")
    # simulate = conf.cleanup.getboolean("simulate")
    logger.info(f"PATH|{path}")
    logger.info(f"VERSIONS_TO_RETAIN|{versions_to_retain}")
    logger.info(f"SIMULATE|{simulate}")

    models = model_helper.get_models(path)
    traindata_id = [model["traindata_id"] for model in models]
    traindata_id.sort()
    traindata_id_to_cleanup = traindata_id[
        : len(traindata_id) - versions_to_retain
        if len(traindata_id) >= versions_to_retain
        else 0
    ]
    models_to_cleanup = [
        model["basefilename"]
        for model in models
        if model["traindata_id"] in traindata_id_to_cleanup
    ]

    for model in models_to_cleanup:
        file_path = f"{path}/{model}*.*"
        file_list = glob(pathname=file_path)
        for file in file_list:
            if simulate:
                logger.info(f"REMOVE|{file}")
            else:
                try:
                    os.remove(file)
                    logger.info(f"REMOVE|{file}")
                except OSError as ex:
                    message = f"ERROR while deleting file {file}"
                    logger.exception(message)
                    raise Exception(message) from ex

    # Log stop
    # TODO: nakijken of dit klopt? path.stem
    logger.info(f"Cleanup models done for config {path.stem}")


def clean_training_data_directories(
    path: Path,
    versions_to_retain: int,
    simulate: bool,
):
    """
    Cleanup training data directories.

    Args:
        path (Path): Path to the directories
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted
    """
    # Get a logger...
    logger = logging.getLogger(__name__)

    # Log start
    logger.info(f"Start cleanup training data for config {path.stem}")

    # path = conf.dirs.getpath("training_dir")
    # versions_to_retain = conf.cleanup.getint("training_versions_to_retain")
    # simulate = conf.cleanup.getboolean("simulate")
    logger.info(f"PATH|{path}")
    logger.info(f"VERSIONS_TO_RETAIN|{versions_to_retain}")
    logger.info(f"SIMULATE|{simulate}")

    training_dirs = [dir for dir in os.listdir(path) if dir.isnumeric()]
    training_dirs.sort()
    traindata_dirs_to_cleanup = training_dirs[
        : len(training_dirs) - versions_to_retain
        if len(training_dirs) >= versions_to_retain
        else 0
    ]
    for dir in traindata_dirs_to_cleanup:
        if simulate:
            logger.info(f"REMOVE|{path}/{dir}")
        else:
            try:
                shutil.rmtree(f"{path}/{dir}")
                logger.info(f"REMOVE|{path}/{dir}")
            except Exception as ex:
                message = f"ERROR while deleting directory {dir}"
                logger.exception(message)
                raise Exception(message) from ex

    # Log stop
    logger.info(f"Cleanup training data done for config {path.stem}")


def clean_predictions(
    path: Path,
    versions_to_retain: int,
    simulate: bool,
):
    """
    Cleanup predictions.

    Args:
        path (Path): Path to the directories
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted
    """
    # Get a logger...
    logger = logging.getLogger(__name__)

    # Log start
    logger.info(f"Start cleanup predictions for config {path.stem}")

    # path = conf.dirs.getpath("output_vector_dir")
    # versions_to_retain = conf.cleanup.getint("prediction_versions_to_retain")
    # simulate = conf.cleanup.getboolean("simulate")
    logger.info(f"PATH|{path.parent}")
    logger.info(f"VERSIONS_TO_RETAIN|{versions_to_retain}")
    logger.info(f"SIMULATE|{simulate}")

    output_vector_path = path.parent
    prediction_dirs = os.listdir(output_vector_path)
    for prediction_dir in prediction_dirs:
        file_path = f"{output_vector_path / prediction_dir}/*.*"
        file_list = glob(pathname=file_path)
        ai_detection_infos = [aidetection_info(path=Path(file)) for file in file_list]
        postprocessing = [x.postprocessing for x in ai_detection_infos]
        postprocessing = list(dict.fromkeys(postprocessing))
        for p in postprocessing:
            traindata_versions = [
                ai_detection_info.traindata_version
                for ai_detection_info in ai_detection_infos
                if p == ai_detection_info.postprocessing
            ]
            traindata_versions.sort()
            traindata_versions_to_cleanup = traindata_versions[
                : len(traindata_versions) - versions_to_retain
                if len(traindata_versions) >= versions_to_retain
                else 0
            ]
            predictions_to_cleanup = [
                ai_detection_info
                for ai_detection_info in ai_detection_infos
                if ai_detection_info.traindata_version in traindata_versions_to_cleanup
                and ai_detection_info.postprocessing == p
            ]
            for prediction in predictions_to_cleanup:
                if simulate:
                    logger.info(f"REMOVE|{prediction.path}")
                else:
                    try:
                        os.remove(prediction.path)
                        logger.info(f"REMOVE|{prediction.path}")
                    except Exception as ex:
                        message = f"ERROR while deleting file {prediction.path}"
                        logger.exception(message)
                        raise Exception(message) from ex

    # Log stop
    logger.info(f"Cleanup predictions done for config {path.stem}")

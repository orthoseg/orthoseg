"""
Automatic cleanup of 'old' models, predictions and training data directories.
"""

from glob import glob
import os
import shutil
from pathlib import Path

from orthoseg.model import model_helper
from orthoseg.util import log_util
from orthoseg.util.data import aidetection_info


def clean_models(
    config_path: Path,
    path: Path,
    versions_to_retain: int,
    simulate: bool,
):
    """
    Cleanup models.

    Args:
        config_path (Path): config path
        path (Path): Path to the directories
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted
    """
    logger = log_util.main_log_init(path.parent / "log", __name__)
    logger.info(f"CLEANUP|Start cleanup models for config {config_path.stem}")
    logger.info(f"VERSIONS_TO_RETAIN|{versions_to_retain}")
    logger.info(f"SIMULATE|{simulate}")
    logger.info(f"PATH|{path}")

    models = model_helper.get_models(model_dir=path)
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
                logger.info(f"REMOVE|{Path(file).name}")
            else:
                try:
                    os.remove(file)
                    logger.info(f"REMOVE|{Path(file).name}")
                except OSError as ex:
                    message = f"ERROR while deleting file {file}"
                    logger.exception(message)
                    raise Exception(message) from ex

    logger.info(f"CLEANUP|Cleanup models done for config {config_path.stem}")


def clean_training_data_directories(
    config_path: Path,
    path: Path,
    versions_to_retain: int,
    simulate: bool,
):
    """
    Cleanup training data directories.

    Args:
        config_path (Path): config path
        path (Path): Path to the directories
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted
    """
    logger = log_util.main_log_init(path.parent / "log", __name__)
    logger.info(f"CLEANUP|Start cleanup training data for config {config_path.stem}")
    logger.info(f"VERSIONS_TO_RETAIN|{versions_to_retain}")
    logger.info(f"SIMULATE|{simulate}")
    logger.info(f"PATH|{path}")

    training_dirs = [dir for dir in os.listdir(path) if dir.isnumeric()]
    training_dirs.sort()
    traindata_dirs_to_cleanup = training_dirs[
        : len(training_dirs) - versions_to_retain
        if len(training_dirs) >= versions_to_retain
        else 0
    ]
    for dir in traindata_dirs_to_cleanup:
        if simulate:
            logger.info(f"REMOVE|{dir}")
        else:
            try:
                shutil.rmtree(f"{path}/{dir}")
                logger.info(f"REMOVE|{dir}")
            except Exception as ex:
                message = f"ERROR while deleting directory {path}/{dir}"
                logger.exception(message)
                raise Exception(message) from ex

    logger.info(f"CLEANUP|Cleanup training data done for config {config_path.stem}")


def clean_predictions(
    config_path: Path,
    path: Path,
    versions_to_retain: int,
    simulate: bool,
):
    """
    Cleanup predictions.

    Args:
        config_path (Path): config path
        path (Path): Path to the directories
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted
    """
    logger = log_util.main_log_init(path.parent.parent / "log", __name__)
    logger.info(f"CLEANUP|Start cleanup predictions for config {config_path.stem}")
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
        logger.info(f"PATH|{path.parent}/{prediction_dir}")
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
                    logger.info(f"REMOVE|{prediction.path.name}")
                else:
                    try:
                        os.remove(prediction.path)
                        logger.info(f"REMOVE|{prediction.path.name}")
                    except Exception as ex:
                        message = f"ERROR while deleting file {prediction.path}"
                        logger.exception(message)
                        raise Exception(message) from ex

    logger.info(f"CLEANUP|Cleanup predictions done for config {config_path.stem}")

"""
Automatic cleanup of 'old' models, predictions and training data directories.
"""

from glob import glob
import logging
import os
import shutil
from pathlib import Path
from typing import List


from orthoseg.model import model_helper
from orthoseg.util.data import aidetection_info

# Get a logger...
logger = logging.getLogger(__name__)


def clean_models(
    model_dir: Path,
    versions_to_retain: int,
    simulate: bool,
) -> List[str]:
    """
    Cleanup models.

    Args:
        model_dir (Path): Path to the directory with the models to be cleaned
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted

    Raises:
        Exception: ERROR while deleting file

    Returns:
        List[str]: List of models to be cleaned
    """
    logger.info(f"{model_dir=}, {versions_to_retain=}, {simulate=}")
    models_to_cleanup = []

    if model_dir.exists():
        models = model_helper.get_models(model_dir=model_dir)
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
            file_path = f"{model_dir}/{model}*.*"
            file_list = glob(pathname=file_path)
            for file in file_list:
                removed_file = Path(file).name
                if simulate:
                    logger.info(f"{removed_file=}")
                else:
                    try:
                        os.remove(file)
                        logger.info(f"{removed_file=}")
                    except OSError as ex:
                        message = f"ERROR while deleting file {file}"
                        logger.exception(message)
                        raise Exception(message) from ex
    else:
        logger.info(f"Directory {model_dir.name} doesn't exist")
    return models_to_cleanup


def clean_training_data_directories(
    training_dir: Path,
    versions_to_retain: int,
    simulate: bool,
) -> List[str]:
    """
    Cleanup training data directories.

    Args:
        training_dir (Path): Path to the directory with the training data
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted

    Raises:
        Exception: ERROR while deleting file

    Returns:
        List[str]: List of training directories to be cleaned
    """
    logger.info(f"{training_dir=}, {versions_to_retain=}, {simulate=}")
    traindata_dirs_to_cleanup = []

    if training_dir.exists():
        training_dirs = [dir for dir in os.listdir(training_dir) if dir.isnumeric()]
        training_dirs.sort()
        traindata_dirs_to_cleanup = training_dirs[
            : len(training_dirs) - versions_to_retain
            if len(training_dirs) >= versions_to_retain
            else 0
        ]
        for dir in traindata_dirs_to_cleanup:
            removed_dir = dir
            if simulate:
                logger.info(f"{removed_dir=}")
            else:
                try:
                    shutil.rmtree(f"{training_dir}/{dir}")
                    logger.info(f"{removed_dir=}")
                except Exception as ex:
                    message = f"ERROR while deleting directory {training_dir}/{dir}"
                    logger.exception(message)
                    raise Exception(message) from ex
    else:
        logger.info(f"Directory {training_dir.name} doesn't exist")
    return traindata_dirs_to_cleanup


def clean_predictions(
    output_vector_dir: Path,
    versions_to_retain: int,
    simulate: bool,
) -> List[str]:
    """
    Cleanup predictions.

    Args:
        output_vector_dir (Path): Path to the directory containing
                                  the vector predictions
        versions_to_retain (int): Versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted

    Raises:
        Exception: ERROR while deleting file

    Returns:
        List[str]: List of training directories to be cleaned
    """
    predictions_to_cleanup: List[str] = []
    if output_vector_dir.exists():
        file_path = f"{output_vector_dir}/*.*"
        file_list = glob(pathname=file_path)
        try:
            ai_detection_infos = [
                aidetection_info(path=Path(file)) for file in file_list
            ]
            postprocessing = [x.postprocessing for x in ai_detection_infos]
            postprocessing = list(dict.fromkeys(postprocessing))
            logger.info(f"{output_vector_dir=}, {versions_to_retain=}, {simulate=}")
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
                predictions_to_cleanup.extend(
                    [
                        ai_detection_info
                        for ai_detection_info in ai_detection_infos
                        if ai_detection_info.traindata_version
                        in traindata_versions_to_cleanup
                        and ai_detection_info.postprocessing == p
                    ]
                )
            for prediction in predictions_to_cleanup:
                removed_prediction = prediction.path.name
                if simulate:
                    logger.info(f"{removed_prediction=}")
                else:
                    try:
                        os.remove(prediction.path)
                        logger.info(f"{removed_prediction=}")
                    except Exception as ex:
                        message = f"ERROR while deleting file {prediction.path}"
                        logger.exception(message)
                        raise Exception(message) from ex
        except Exception as ex:
            logger.info(f"{ex}")
    else:
        logger.info(f"Directory {output_vector_dir.name} doesn't exist")
    return predictions_to_cleanup


def clean_project_dir(
    model_dir: Path,
    model_versions_to_retain: int,
    training_dir: Path,
    training_versions_to_retain: int,
    output_vector_dir: Path,
    prediction_versions_to_retain: int,
    simulate: bool,
):
    """
    Cleanup project directory.

    Args:
        model_dir (Path): Path to the directory with the models to be cleaned
        model_versions_to_retain (int): Model versions to retain
        training_dir (Path): Path to the directory with the training data to be cleaned
        training_versions_to_retain (int): Training data versions to retain
        output_vector_dir (Path): Path to the directory
                                  with the predictions to be cleaned
        prediction_versions_to_retain (int): Prediction versions to retain
        simulate (bool): Simulate cleanup, files are logged, no files are deleted
    """
    removed = {}
    removed["models"] = clean_models(
        model_dir=model_dir,
        versions_to_retain=model_versions_to_retain,
        simulate=simulate,
    )
    removed["training_dirs"] = clean_training_data_directories(
        training_dir=training_dir,
        versions_to_retain=training_versions_to_retain,
        simulate=simulate,
    )
    output_vector_parent_dir = output_vector_dir.parent
    if output_vector_parent_dir.exists():
        prediction_dirs = os.listdir(output_vector_parent_dir)
        removed["predictions"] = []
        for prediction_dir in prediction_dirs:
            removed["predictions"].extend(
                clean_predictions(
                    output_vector_dir=output_vector_parent_dir / prediction_dir,
                    versions_to_retain=prediction_versions_to_retain,
                    simulate=simulate,
                )
            )
    else:
        logger.info(f"Directory {output_vector_parent_dir.name} doesn't exist")
    return removed

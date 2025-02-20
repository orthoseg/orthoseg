"""Module with functions to clean up old project data."""

import logging
import os
import shutil
from pathlib import Path

from orthoseg.model import model_helper
from orthoseg.util.data import aidetection_info

# Get a logger...
logger = logging.getLogger(__name__)


def clean_models(
    model_dir: Path, versions_to_retain: int, simulate: bool
) -> list[Path]:
    """Cleanup models.

    Args:
        model_dir (Path): Path to the directory with the models to be cleaned
        versions_to_retain (int): Number of traindata versions to retain the models for.
            If < 0, all versions are retained.
        simulate (bool): Simulate cleanup, files are logged, no files are deleted

    Raises:
        RuntimeError: an error occurred while deleting a file

    Returns:
        list[Path]: List of removed model files
    """
    # Check input
    if versions_to_retain < 0:
        return []
    if not model_dir.exists():
        logger.info(f"Directory doesn't exist: {model_dir!s}")
        return []

    logger.debug(f"clean_models in {model_dir!s}, {versions_to_retain=}, {simulate=}")

    models = model_helper.get_models(model_dir=model_dir)
    traindata_ids = [model["traindata_id"] for model in models]
    traindata_ids = sorted(set(traindata_ids))
    if len(traindata_ids) < versions_to_retain:
        return []

    traindata_ids_to_cleanup = traindata_ids[: len(traindata_ids) - versions_to_retain]
    models_to_cleanup = [
        model["basefilename"]
        for model in models
        if model["traindata_id"] in traindata_ids_to_cleanup
    ]

    # Find all files to remove and remove them
    files_to_remove = []
    for model in models_to_cleanup:
        file_pattern = f"{model}*.*"
        files = list(model_dir.glob(pattern=file_pattern))
        files_to_remove.extend(files)

        for file_to_remove in files:
            logger.info(f"remove model file ({simulate=}): {file_to_remove}")
            if simulate:
                continue

            try:
                file_to_remove.unlink()
            except OSError as ex:  # pragma: no cover
                message = f"ERROR while deleting file {file_to_remove}"
                logger.exception(message)
                raise RuntimeError(message) from ex

    return files_to_remove


def clean_training_data_directories(
    training_dir: Path, versions_to_retain: int, simulate: bool
) -> list[Path]:
    """Cleanup training data directories.

    Args:
        training_dir (Path): Path to the directory with the training data
        versions_to_retain (int): Number of traindata versions to retain the training
            data directories for. If < 0, all versions are retained.
        simulate (bool): Simulate cleanup, directories are logged, no files are deleted

    Raises:
        Exception: an error occcured while deleting a training directory

    Returns:
        list[Path]: List of training directories to be removed
    """
    # Check input
    if versions_to_retain < 0:
        return []
    if not training_dir.exists():
        logger.info(f"Directory doesn't exist: {training_dir!s}")
        return []

    logger.debug(
        f"clean_training_data_directories in {training_dir!s}, {versions_to_retain=}, "
        f"{simulate=}"
    )

    dirnames = [dir for dir in os.listdir(training_dir) if dir.isnumeric()]
    dirnames.sort()
    if len(dirnames) < versions_to_retain:
        return []

    dirnames_to_clean = dirnames[: len(dirnames) - versions_to_retain]
    dirs_to_remove = []
    for dirname in dirnames_to_clean:
        dir = training_dir / dirname
        dirs_to_remove.append(dir)
        logger.info(f"remove training dir ({simulate=}): {dir}")

        if simulate:
            continue

        try:
            shutil.rmtree(dir)
        except Exception as ex:  # pragma: no cover
            message = f"ERROR deleting directory {dir}"
            logger.exception(message)
            raise RuntimeError(message) from ex

    return dirs_to_remove


def clean_predictions(
    output_vector_dir: Path, versions_to_retain: int, simulate: bool
) -> list[Path]:
    """Cleanup predictions.

    If the directory contains files that don't comply with the AiDetectionInfo format,
    is is not cleaned.

    Args:
        output_vector_dir (Path): Path to the directory containing the vector
            predictions
        versions_to_retain (int): Number of traindata versions to retain the
            predictions for. If < 0, all versions are retained.
        simulate (bool): Simulate cleanup, files are logged, no files are deleted

    Raises:
        RuntimeError: ERROR while deleting file

    Returns:
        list[Path]: List of prediction files to be removed
    """
    # Check input
    if versions_to_retain < 0:
        return []
    if not output_vector_dir.exists():
        logger.info(f"Directory doesn't exist: {output_vector_dir!s}")
        return []

    logger.debug(
        f"clean_predictions in {output_vector_dir!s}, "
        f"{versions_to_retain=}, {simulate=}"
    )

    files = output_vector_dir.glob(pattern="*.*")
    ai_detection_infos = []
    traindata_versions = []
    for file in files:
        try:
            ai_detection_info = aidetection_info(path=file)
            ai_detection_infos.append(ai_detection_info)
            traindata_versions.append(ai_detection_info.traindata_version)
        except ValueError as ex:
            logger.error(
                f"Invalid prediction file found, skip cleanup of prediction dir: {ex}"
            )
            return []

    traindata_versions = sorted(set(traindata_versions))
    if len(traindata_versions) < versions_to_retain:
        return []

    versions_to_cleanup = traindata_versions[
        : len(traindata_versions) - versions_to_retain
    ]
    prediction_files_to_cleanup = [
        ai_detection_info.path
        for ai_detection_info in ai_detection_infos
        if ai_detection_info.traindata_version in versions_to_cleanup
    ]

    prediction_files_to_remove = []
    for prediction_path in prediction_files_to_cleanup:
        prediction_files_to_remove.append(prediction_path)
        logger.info(f"remove prediction file ({simulate=}): {prediction_path.name}")
        if simulate:
            continue

        try:
            prediction_path.unlink()
        except Exception as ex:  # pragma: no cover
            message = f"ERROR while deleting file {prediction_path}"
            logger.exception(message)
            raise RuntimeError(message) from ex

    return prediction_files_to_remove


def clean_project_dir(
    model_dir: Path,
    model_versions_to_retain: int,
    training_dir: Path,
    training_versions_to_retain: int,
    output_vector_dir: Path,
    prediction_versions_to_retain: int,
    simulate: bool,
) -> dict:
    """Cleanup project directory.

    Args:
        model_dir (Path): Path to the directory with the models to be cleaned
        model_versions_to_retain (int): Model versions to retain. If <0, all models are
            retained.
        training_dir (Path): Path to the directory with the training data to be cleaned
        training_versions_to_retain (int): Training data versions to retain. If <0, all
            versions are retainded.
        output_vector_dir (Path): Path to the directory with the predictions to be
            cleaned
        prediction_versions_to_retain (int): Prediction versions to retain. If <0, all
            versions are retained.
        simulate (bool): Simulate cleanup, files are logged, no files are deleted

    Returns:
        dict: Dictionary with removed models, training directories and predictions
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
    removed["predictions"] = []
    if output_vector_parent_dir.exists():
        prediction_dirs = os.listdir(output_vector_parent_dir)
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

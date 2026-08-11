"""Module with specific helper functions to get info for gisdata files."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Functions to get base paths of locations where GIS data
# can be stored
# ---------------------------------------------------------


def aidetection_dir() -> Path:
    """AI detection path.

    Returns:
        Path: AI detection path
    """
    return Path("X:/Monitoring/orthoseg")


# ---------------------------------------------------------
# Functions to get info about AI detection files
# ---------------------------------------------------------


class AiDetectionInfo:
    """Information about an ai detection result."""

    def __init__(
        self,
        path: Path,
        subject: str,
        traindata_version: int,
        image_layer: str,
        postprocessing: str,
    ):
        """Init.

        Args:
            path (Path): the path to the file
            subject (str): the detection/segment subject
            traindata_version (int): the version of the data used to train the model
            image_layer (str): the image layer that the detection is based on
            postprocessing (str): the postprocessing done
        """
        self.path = path
        self.subject = subject
        self.traindata_version = traindata_version
        self.image_layer = image_layer
        self.postprocessing = postprocessing


def aidetection_info(path: Path) -> AiDetectionInfo:
    """Get the properties from an ai detection filepath.

    Args:
        path (Path): the filepath to the ai detection file

    Returns:
        AiDetectionInfo: info about the detection.
    """
    # Extract the fields...
    try:
        param_values = path.stem.split("_")
        if len(param_values) < 3:
            logger.warning(
                "No valid path for an ai detection, split('_') should result in at "
                f"least 3 fields: {path}"
            )

        subject = param_values[0]
        model_info = param_values[1]
        model_info_values = model_info.split(".")
        traindata_version = int(model_info_values[0])
        """
        architecture_version = 0
        trainparams_version = 0
        if len(model_info_values) > 1:
            architecture_version = int(model_info_values[1])
            if len(model_info_values) > 2:
                trainparams_version = int(model_info_values[2])
        aimodel_epoch = int(param_values[2])
        """
        # If the third field is an integer, it is a legacy name with the epoch number.
        if param_values[2].isdigit():
            index_image_layer = 3
        else:
            index_image_layer = 2

        image_layer = param_values[index_image_layer]

        index_postprocessing = index_image_layer + 1
        if len(param_values) > index_postprocessing:
            postprocessing = "_".join(param_values[index_postprocessing:])
        else:
            postprocessing = ""

    except Exception as ex:
        raise ValueError(f"Error in get_aidetection_info on {path}") from ex

    return AiDetectionInfo(
        path, subject, traindata_version, image_layer, postprocessing
    )

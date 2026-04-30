"""Helper functions for managing model weights."""

import logging
import re
import tempfile
import urllib.request
from pathlib import Path

WEIGHTS_NOTOP_BASE_URL = (
    "https://github.com/orthoseg/orthoseg_models/releases/download/v0.1.0/"
)

# Listing of available notop weights.
# A dict of architecture -> weights type -> weights versions available.
WEIGHTS_NOTOP_AVAILABLE: dict[str, dict[str, list[int]]] = {
    "mobilenetv2+linknet": {"aerial": [1]},
    "inceptionresnetv2+unet": {"aerial": [1]},
}


logger = logging.getLogger(__name__)


def get_weights_types_for_architecture(architecture: str) -> list[str]:
    """Get the weights types available for the given architecture."""
    return list(WEIGHTS_NOTOP_AVAILABLE.get(architecture, {}).keys())


def get_model_weights_path(
    architecture: str,
    weights_type: str,
    weights_dir: Path | None = None,
) -> Path:
    """Get weights to initialize the model with.

    Check first in the `weights_dir` if a weights file of the following format can be
    found: `{architecture}_{weights_type}_notop.weights.h5`.
    If not found, look further for weights of the given type for the given architecture
    in the `WEIGHTS_NOTOP_AVAILABLE` listing.

    Args:
        architecture (str): the architecture to get the weights for.
        weights_type (str): the type of weights to get.
        weights_dir (Path | None): directory where pretrained weights are cached to
            and read from. If None, the system temp directory is used.

    Raises:
        ValueError: if no weights are available/found for the given architecture and
            weights type.

    Returns:
        Path: the path to the weights file.


    """
    # First check if the weights file can be easily found in the weights dir.
    if weights_dir is not None:
        weights_name = f"{architecture}_{weights_type}_notop.weights.h5"
        weights_path = weights_dir / weights_name
        if weights_path.exists():
            return weights_path

    # Weights not directly found, so look further.
    weights_parts = weights_type.split("-")
    weights_type_base = weights_parts[0]  # e.g. "aerial" from "aerial-v1"
    if re.match(r"^v\d+$", weights_parts[-1]) is not None:
        # If the last part of the weights type is a version number (e.g. "v1"), split
        # base type and version number.
        weights_type_version = int(weights_parts[-1].lstrip("v"))
        weights_type_base = "-".join(weights_parts[:-1])
    else:
        # If no version number specified, use None for now.
        weights_type_version = None
        weights_type_base = weights_type

    weights_versions: list[int] = WEIGHTS_NOTOP_AVAILABLE.get(architecture, {}).get(
        weights_type_base, []
    )

    # There are no weights configured for this situation, so return None.
    if weights_versions is None or len(weights_versions) == 0:
        raise ValueError(
            f"No weights available for {architecture=}, {weights_type=}, only for "
            f"{WEIGHTS_NOTOP_AVAILABLE}"
        )

    if weights_type_version is None:
        weights_type_version = max(weights_versions)

    weights_name = (
        f"{architecture}_{weights_type_base}-v{weights_type_version}_notop.weights.h5"
    )

    if weights_dir is None:
        weights_dir = Path(tempfile.gettempdir())
        tmp_weights_dir = True
    else:
        weights_dir.mkdir(parents=True, exist_ok=True)
        tmp_weights_dir = False

    weights_path = weights_dir / weights_name
    if weights_path.exists():
        return weights_path

    # Download the weights and save them in the weights dir.
    if tmp_weights_dir:
        logger.info(
            "No weights_dir specified, so cache pretrained weights in the "
            f"system temp directory: {weights_dir}"
        )

    weights_url = f"{WEIGHTS_NOTOP_BASE_URL}{weights_name}"
    try:
        logger.info(f"Downloading weights from {weights_url} to {weights_path}")
        urllib.request.urlretrieve(weights_url, weights_path)
    except Exception as ex:
        raise RuntimeError(
            f"Error downloading weights from {weights_url} to {weights_path}: {ex}"
        ) from ex

    return weights_path

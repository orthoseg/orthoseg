"""Module with specific helper functions to manage the configuration of orthoseg."""

import configparser
import json
import logging
import os
import pprint
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from orthoseg.lib.prepare_traindatasets import LabelInfo
from orthoseg.util import config_util
from orthoseg.util.image_util import FileLayerSource, WMSLayerSource

# Get a logger...
logger = logging.getLogger(__name__)

# Define the chars that cannot be used in codes that are use in filenames.
# Remark: '_' cannot be used because '_' is used as devider to parse filenames, and if
# it is used in codes as well the parsing becomes a lot more difficult.
illegal_chars_in_codes = ["_", ",", ".", "?", ":"]

_run_tmp_dir: Path | None = None

config: configparser.ConfigParser
config_paths: list[Path]
config_filepaths_used: list[Path]
config_overrules: Any
config_overrules_path: Path | None
general: Any
model: Any
download: Any
train: Any
predict: Any
postprocess: Any
dirs: Any
files: Any
email: Any
logging_conf: Any
cleanup: Any

layer_config_filepath_used: Path
image_layers: dict


def pformat_config() -> str:
    """Format the config to a string.

    Returns:
        str: the configuration as string
    """
    message = f"Config files used: {pprint.pformat(config_filepaths_used)} \n"
    message += f"Layer config file used: {layer_config_filepath_used} \n"
    message += "Config info listing:\n"
    message += pprint.pformat(config_util.as_dict(config))
    message += "Layer config info listing:\n"
    message += pprint.pformat(image_layers)
    return message


def read_orthoseg_config(config_path: Path, overrules: list[str] = []):
    """Read an orthoseg configuration file.

    Args:
        config_path (Path): path to the configuration file to read.
        overrules (list[str], optional): list of config options that will overrule other
            ways to supply configuration. They should be specified as a list of
            "<section>.<parameter>=<value>" strings. Defaults to [].
    """
    # Set the temporary directory
    _set_tmp_dir()

    # Determine list of config files that should be loaded
    config_paths = config_util.get_config_files(config_path)

    # If there are overrules, write them to a temporary configuration file.
    global config_overrules
    config_overrules = overrules
    global config_overrules_path
    config_overrules_path = None
    if len(config_overrules) > 0:
        config_overrules_path = get_run_tmp_dir() / "config_overrules.ini"

        # Create config parser, add all overrules
        overrules_parser = configparser.ConfigParser()
        for overrule in config_overrules:
            parts = overrule.split("=")
            if len(parts) != 2:
                raise ValueError(f"invalid config overrule found: {overrule}")
            key, value = parts
            parts2 = key.split(".")
            if len(parts2) != 2:
                raise ValueError(f"invalid config overrule found: {overrule}")
            section, parameter = parts2
            if section not in overrules_parser:
                overrules_parser[section] = {}
            overrules_parser[section][parameter] = value

        # Write to temp file and add file to config_paths
        with open(config_overrules_path, "w") as overrules_file:
            overrules_parser.write(overrules_file)
        config_paths.append(config_overrules_path)

    # Load configs
    global config
    config = config_util.read_config_ext(config_paths)

    # Now do orthoseg-specific checks, inits,... on config
    global config_filepaths_used
    config_filepaths_used = config_paths

    # Now set global variables to each section as shortcuts
    global general
    general = config["general"]
    global model
    model = config["model"]
    global download
    download = config["download"]
    global train
    train = config["train"]
    global predict
    predict = config["predict"]
    global postprocess
    postprocess = config["postprocess"]
    global dirs
    dirs = config["dirs"]
    global files
    files = config["files"]
    global logging_conf
    logging_conf = config["logging"]
    global email
    email = config["email"]
    global cleanup
    cleanup = config["cleanup"]

    # Some checks to make sure the config is loaded properly
    segment_subject = general.get("segment_subject")
    if segment_subject is None or segment_subject == "MUST_OVERRIDE":
        raise ValueError(
            "Projectconfig parameter general.segment_subject needs to be overruled to"
            "a proper name in a specific project config file, \nwith config_filepaths "
            f"{config_paths}"
        )
    elif any(
        illegal_character in segment_subject
        for illegal_character in illegal_chars_in_codes
    ):
        raise ValueError(
            f"Projectconfig parameter general.segment_subject ({segment_subject}) "
            f"should not contain any of the following chars: {illegal_chars_in_codes}"
        )

    # If the projects_dir parameter is a relative path, resolve it towards the location
    # of the project config file.
    projects_dir = dirs.getpath("projects_dir")
    if not projects_dir.is_absolute():
        projects_dir_absolute = (config_path.parent / projects_dir).resolve()
        logger.info(
            f"dirs.projects_dir was relative: is resolved to {projects_dir_absolute}"
        )
        dirs["projects_dir"] = projects_dir_absolute.as_posix()

    # Read the layer config
    layer_config_filepath = files.getpath("image_layers_config_filepath")
    global layer_config_filepath_used
    layer_config_filepath_used = layer_config_filepath

    global image_layers
    image_layers = _read_layer_config(layer_config_filepath=layer_config_filepath)


def _set_tmp_dir(dir: str = "orthoseg") -> Path:
    # Check if TMPDIR exists in environment
    tmpdir = os.getenv("TMPDIR")
    tmp_dir = Path(tempfile.gettempdir() if tmpdir is None else tmpdir)

    # Check if the TMPDIR is not already set to a orthoseg directory
    if tmp_dir.name.lower() != dir.lower():
        tmp_dir /= dir
        # Set the TMPDIR in the environment
        os.environ["TMPDIR"] = tmp_dir.as_posix()
        tempfile.tempdir = tmp_dir.as_posix()

    # Create TMPDIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    return tmp_dir


def get_run_tmp_dir() -> Path:
    """Get a temporary directory for this run.

    If no temporary directory exists yet, it is created.

    Returns:
        Path: the path to the temporary directory.
    """
    global _run_tmp_dir

    if _run_tmp_dir is None:
        _run_tmp_dir = Path(tempfile.gettempdir())
        _run_tmp_dir.mkdir(parents=True, exist_ok=True)
        _run_tmp_dir = Path(tempfile.mkdtemp(prefix="run_", dir=_run_tmp_dir))

    return _run_tmp_dir


def remove_run_tmp_dir():
    """Remove temporary run directory, including all files or directories in it."""
    global _run_tmp_dir

    if _run_tmp_dir is not None:
        shutil.rmtree(_run_tmp_dir, ignore_errors=True)
        _run_tmp_dir = None


def get_train_label_infos() -> list[LabelInfo]:
    """Searches and returns LabelInfos that can be used to create a training dataset.

    Returns:
        list[LabelInfo]: List of LabelInfos found.
    """
    train_label_infos = _prepare_train_label_infos(
        labelpolygons_pattern=train.getpath("labelpolygons_pattern"),
        labellocations_pattern=train.getpath("labellocations_pattern"),
        label_datasources=train.getdict("label_datasources", None),
        image_layers=image_layers,
    )
    if train_label_infos is None or len(train_label_infos) == 0:
        raise ValueError(
            "No valid label file config found in train.label_datasources or "
            f"with patterns {train.get('labelpolygons_pattern')} and "
            f"{train.get('labellocations_pattern')}"
        )
    return train_label_infos


def determine_classes():
    """Determine classes.

    Raises:
        Exception: Error reading classes

    Returns:
        any: classes
    """
    try:
        classes = train.getdict("classes")

        # If the burn_value property isn't supplied for the classes, add them
        for class_id, (classname) in enumerate(classes):
            if "burn_value" not in classes[classname]:
                classes[classname]["burn_value"] = class_id
        return classes
    except Exception as ex:
        raise Exception(f"Error reading classes: {train.get('classes')}") from ex


def _read_layer_config(layer_config_filepath: Path) -> dict:
    # Init
    if not layer_config_filepath.exists():
        raise Exception(f"Layer config file not found: {layer_config_filepath}")

    # Read config file...
    layer_config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        converters={
            "list": lambda x: _str2list(x),
            "listint": lambda x: _str2intlist(x),
            "dict": lambda x: None if x is None else json.loads(x),
            "path": lambda x: Path(x),
        },
    )
    layer_config.read(layer_config_filepath)

    # Prepare data
    image_layers: dict[str, dict] = {}
    for image_layer in layer_config.sections():
        # First check if the image_layer code doesn't contain 'illegal' characters
        if any(illegal_char in image_layer for illegal_char in illegal_chars_in_codes):
            raise ValueError(
                f"Section name [{image_layer}] in layer config should not contain any"
                f"of these chars: {illegal_chars_in_codes}, in {layer_config_filepath}"
            )

        # Init layer with all parameters in the section as dict
        image_layers[image_layer] = dict(layer_config[image_layer])
        image_layers[image_layer]["layername"] = image_layer

        # Check if the mandatory layer-level properties are present
        if "projection" not in image_layers[image_layer]:
            raise ValueError(
                f"Image layer [{image_layer}] in layer config needs a 'projection' key!"
            )

        # If the layer source(s) are specified in a json parameter, parse it
        if "layersources" in image_layers[image_layer]:
            image_layers[image_layer]["layersources"] = layer_config[
                image_layer
            ].getdict("layersources")
        else:
            # If not, the layersource is specified in some top-level parameters
            layersource = {}
            layersource_keys = [
                "wms_server_url",
                "wms_version",
                "wms_layernames",
                "wms_layerstyles",
                "bands",
                "wms_username",
                "wms_password",
                "random_sleep",
                "wms_ignore_capabilities_url",
                "path",
                "layername",
            ]
            for key in layersource_keys:
                if key in layer_config[image_layer]:
                    layersource[key] = layer_config[image_layer][key]
            image_layers[image_layer]["layersources"] = [layersource]

        # Convert the layersource dicts to layersource objects
        layersource_objects = []
        for layersource in image_layers[image_layer]["layersources"]:
            layersource_object: WMSLayerSource | FileLayerSource
            try:
                # If not, the layersource should be specified in seperate parameters
                if "wms_server_url" in layersource:
                    layersource_object = WMSLayerSource(
                        wms_server_url=layersource["wms_server_url"],
                        wms_version=layersource.get("wms_version", "1.3.0"),
                        layernames=_str2list(layersource["wms_layernames"]),
                        layerstyles=_str2list(layersource.get("wms_layerstyles")),
                        bands=_str2intlist(layersource.get("bands", None)),
                        username=layersource.get("wms_username", None),
                        password=layersource.get("wms_password", None),
                        random_sleep=int(layersource.get("random_sleep", 0)),
                        wms_ignore_capabilities_url=_str2bool(
                            layersource.get("wms_ignore_capabilities_url", "False")
                        ),
                    )
                elif "path" in layersource:
                    path = Path(layersource["path"])
                    if not path.is_absolute():
                        # Resolve relative path based on layer_config_filepath.parent
                        path = layer_config_filepath.parent / layersource["path"]
                        path = path.resolve()
                    layersource_object = FileLayerSource(
                        path=path,
                        layernames=_str2list(layersource["layername"]),
                        bands=_str2intlist(layersource.get("bands", None)),
                    )
                else:
                    raise ValueError(
                        f"Invalid layersource, should be WMS or file: {layersource}"
                    )
            except Exception as ex:
                raise ValueError(
                    f"Missing parameter in image_layer {image_layer}, layersource "
                    f"{layersource}: {ex}"
                ) from ex
            layersource_objects.append(layersource_object)
        image_layers[image_layer]["layersources"] = layersource_objects

        # Read nb_concurrent calls param
        image_layers[image_layer]["nb_concurrent_calls"] = layer_config[
            image_layer
        ].getint("nb_concurrent_calls", fallback=1)

        # Check if a region of interest is specified as file or bbox
        image_layers[image_layer]["roi_filepath"] = layer_config[image_layer].getpath(
            "roi_filepath", fallback=None
        )
        bbox_tuple = None
        if layer_config.has_option(image_layer, "bbox"):
            bbox_list = layer_config[image_layer].getlist("bbox")
            bbox_tuple = (
                float(bbox_list[0]),
                float(bbox_list[1]),
                float(bbox_list[2]),
                float(bbox_list[3]),
            )
        image_layers[image_layer]["bbox"] = bbox_tuple

        # Check if the grid xmin and xmax are specified
        image_layers[image_layer]["grid_xmin"] = layer_config[image_layer].getfloat(
            "grid_xmin", fallback=0
        )
        image_layers[image_layer]["grid_ymin"] = layer_config[image_layer].getfloat(
            "grid_ymin", fallback=0
        )

        # Check if a image_pixels_ignore_border is specified
        image_layers[image_layer]["image_pixels_ignore_border"] = layer_config[
            image_layer
        ].getint("image_pixels_ignore_border", fallback=0)

        # Convert pixel_x_size and pixel_y_size to float
        image_layers[image_layer]["pixel_x_size"] = layer_config[image_layer].getfloat(
            "pixel_x_size"
        )
        image_layers[image_layer]["pixel_y_size"] = layer_config[image_layer].getfloat(
            "pixel_y_size"
        )

    return image_layers


def _prepare_train_label_infos(
    labelpolygons_pattern: Path,
    labellocations_pattern: Path,
    label_datasources: dict,
    image_layers: dict[str, dict],
) -> list[LabelInfo]:
    # Search for the files based on the file name patterns...
    pattern_label_infos: dict[str, LabelInfo] = {}
    if labelpolygons_pattern is not None or labellocations_pattern is not None:
        pattern_label_infos = {
            info.locations_path.resolve().as_posix(): info
            for info in _search_label_files(
                labelpolygons_pattern, labellocations_pattern
            )
        }

    # Process the label datasources that are explicitly configured in the project.
    label_infos: list[LabelInfo] = []
    datasources_overruled = set()
    if label_datasources is not None:
        for label_key, label_ds in label_datasources.items():
            if label_ds.get("locations_path") is None:
                raise ValueError(f"locations_path not specified for {label_ds}")

            # Backwards compatibility for "data_path"
            if label_ds.get("polygons_path") is None:
                label_ds["polygons_path"] = label_ds.get("data_path")

            # If locations_path was also found via pattern matching, we can reuse info.
            locations_key = Path(label_ds["locations_path"]).resolve().as_posix()
            if locations_key in pattern_label_infos:
                pattern_label_info = pattern_label_infos[locations_key]
                if label_ds.get("polygons_path") is None:
                    label_ds["polygons_path"] = pattern_label_info.polygons_path
                if label_ds.get("image_layer") is None:
                    label_ds["image_layer"] = pattern_label_info.image_layer
                if label_ds.get("pixel_x_size") is None:
                    label_ds["pixel_x_size"] = pattern_label_info.pixel_x_size
                if label_ds.get("pixel_y_size") is None:
                    label_ds["pixel_y_size"] = pattern_label_info.pixel_y_size

                # The explicit information overrules the pattern matched version, so set
                # the pattern-matched LabelInfo as overruled.
                datasources_overruled.add(locations_key)

            if label_ds.get("polygons_path") is None:
                raise ValueError(f"polygons_path not specified for {label_ds}")

            # Add as new LabelInfo
            label_infos.append(
                LabelInfo(
                    locations_path=Path(label_ds["locations_path"]),
                    polygons_path=Path(label_ds["polygons_path"]),
                    image_layer=label_ds["image_layer"],
                    pixel_x_size=label_ds.get("pixel_x_size"),
                    pixel_y_size=label_ds.get("pixel_y_size"),
                )
            )

    # Add all pattern matched LabelInfos that were not overruled by explicit datasources
    for pattern_label_info in pattern_label_infos.values():
        locations_key = pattern_label_info.locations_path.resolve().as_posix()
        if locations_key not in datasources_overruled:
            label_infos.append(pattern_label_info)

    # Check if the configured image_layer exists for all label_infos
    for label_info in pattern_label_infos.values():
        if label_info.image_layer not in image_layers:
            raise ValueError(
                f"invalid image_layer in {label_info}: not in {list(image_layers)}"
            )

    return label_infos


def _search_label_files(
    labelpolygons_pattern: Path, labellocations_pattern: Path
) -> list[LabelInfo]:
    if not labelpolygons_pattern.parent.exists():
        raise ValueError(f"Label dir doesn't exist: {labelpolygons_pattern.parent}")
    if not labellocations_pattern.parent.exists():
        raise ValueError(f"Label dir doesn't exist: {labellocations_pattern.parent}")

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
            raise ValueError(  # pragma: no cover
                f"image_layer token not found in {labellocations_path} using pattern "
                f"{labellocations_pattern}"
            )
        image_layer = tokens["image_layer"]

        # Look for the matching (= same image_layer) data file
        found = False
        for labelpolygons_path in labelpolygons_paths:
            tokens = _unformat(labelpolygons_path.stem, labelpolygons_pattern.stem)
            if "image_layer" not in tokens:
                raise ValueError(  # pragma: no cover
                    f"image_layer token not found in {labelpolygons_path} using "
                    f"pattern {labelpolygons_pattern}"
                )

            if tokens["image_layer"] == image_layer:
                found = True
                break

        if found is False:
            raise ValueError(
                f"no matching polygon data file found for {labellocations_path}"
            )
        label_infos.append(
            LabelInfo(
                locations_path=labellocations_path,
                polygons_path=labelpolygons_path,
                image_layer=image_layer,
            )
        )

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
        raise ValueError(f"pattern {pattern} not found in {string}")


def _str2list(input: str | None):
    if input is None:
        return None
    if isinstance(input, list):
        return input
    return [part.strip() for part in input.split(",")]


def _str2intlist(input: str | None):
    if input is None:
        return None
    if isinstance(input, list):
        return input
    return [int(i.strip()) for i in input.split(",")]


def _str2bool(input: str | None):
    if input is None:
        return None
    if isinstance(input, bool):
        return input
    return input.lower() in ("yes", "true", "false", "1")

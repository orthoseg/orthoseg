"""
Module with specific helper functions to manage the configuration of orthoseg.
"""

import configparser
import json
import logging
from pathlib import Path
import pprint
from typing import List, Optional

import orthoseg.model.model_factory as mf
from orthoseg.util import config_util
from orthoseg.util.ows_util import FileLayerSource, WMSLayerSource

# Get a logger...
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Define the chars that cannot be used in codes that are use in filenames.
# Remark: '_' cannot be used because '_' is used as devider to parse filenames, and if
# it is used in codes as well the parsing becomes a lot more difficult.
illegal_chars_in_codes = ["_", ",", ".", "?", ":"]


def pformat_config():
    message = f"Config files used: {pprint.pformat(config_filepaths_used)} \n"
    message += f"Layer config file used: {layer_config_filepath_used} \n"
    message += "Config info listing:\n"
    message += pprint.pformat(config_util.as_dict(config))
    message += "Layer config info listing:\n"
    message += pprint.pformat(image_layers)
    return message


def read_orthoseg_config(config_path: Path):
    # Determine list of config files that should be loaded
    config_paths = config_util.get_config_files(config_path)  # type: ignore
    # Load them
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
    global logging
    logging = config["logging"]
    global email
    email = config["email"]

    # Some checks to make sure the config is loaded properly
    segment_subject = general.get("segment_subject")
    if segment_subject is None or segment_subject == "MUST_OVERRIDE":
        raise Exception(
            "Projectconfig parameter general.segment_subject needs to be overruled to"
            "a proper name in a specific project config file, \nwith config_filepaths "
            f"{config_paths}"
        )
    elif any(
        illegal_character in segment_subject
        for illegal_character in illegal_chars_in_codes
    ):
        raise Exception(
            f"Projectconfig parameter general.segment_subject ({segment_subject}) "
            f"should not contain any of the following chars: {illegal_chars_in_codes}"
        )

    # If the projects_dir parameter is a relative path, resolve it towards the location
    # of the project config file.
    projects_dir = dirs.getpath("projects_dir")
    if not projects_dir.is_absolute():
        projects_dir_absolute = (config_paths[-1].parent / projects_dir).resolve()
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
    image_layers = {}
    for image_layer in layer_config.sections():
        # First check if the image_layer code doesn't contain 'illegal' characters
        if any(illegal_char in image_layer for illegal_char in illegal_chars_in_codes):
            raise ValueError(
                f"Section name [{image_layer}] in layer config should not contain any"
                f"of these chars: {illegal_chars_in_codes}, in {layer_config_filepath}"
            )

        # Init layer with all parameters in the section as dict
        image_layers[image_layer] = dict(layer_config[image_layer])

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
            layersource_object = None
            try:
                # If not, the layersource should be specified in seperate parameters
                if "wms_server_url" in layersource:
                    layersource_object = WMSLayerSource(
                        wms_server_url=layersource.get("wms_server_url"),
                        wms_version=layersource.get("wms_version", "1.3.0"),
                        layernames=_str2list(
                            layersource["wms_layernames"]
                        ),  # type: ignore
                        layerstyles=_str2list(layersource.get("wms_layerstyles")),
                        bands=_str2intlist(layersource.get("bands", None)),
                        random_sleep=int(layersource.get("random_sleep", 0)),
                        wms_ignore_capabilities_url=_str2bool(
                            layersource.get("wms_ignore_capabilities_url", False)
                        ),  # type: ignore
                    )
                elif "path" in layersource:
                    path = Path(layersource["path"])
                    if not path.is_absolute():
                        # Resolve relative path based on layer_config_filepath.parent
                        path = layer_config_filepath.parent / layersource["path"]
                        path = path.resolve()
                    layersource_object = FileLayerSource(
                        path=path,
                        layernames=layersource["layername"],  # type: ignore
                        bands=_str2intlist(layersource.get("bands", None)),
                    )
            except Exception as ex:
                raise ValueError(
                    f"Missing parameter in image_layer {image_layer}, layersource "
                    f"{layersource}: {ex}"
                ) from ex
            if layersource_object is None:
                raise ValueError(
                    "Invalid layersource, should be WMS or file: " f"{layersource}"
                )
            layersource_objects.append(layersource_object)
        image_layers[image_layer]["layersources"] = layersource_objects

        # Read nb_concurrent calls param
        image_layers[image_layer]["nb_concurrent_calls"] = layer_config[
            image_layer
        ].getint("nb_concurrent_calls", fallback=6)

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


def _str2list(input: Optional[str]):
    if input is None:
        return None
    if isinstance(input, List):
        return input
    return [part.strip() for part in input.split(",")]


def _str2intlist(input: Optional[str]):
    if input is None:
        return None
    if isinstance(input, List):
        return input
    return [int(i.strip()) for i in input.split(",")]


def _str2bool(input: Optional[str]):
    if input is None:
        return None
    if isinstance(input, bool):
        return input
    return input.lower() in ("yes", "true", "false", "1")

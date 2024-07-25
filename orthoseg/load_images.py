"""Script to load images from a WMS server."""

import argparse
import logging
import shlex
import shutil
import sys
import traceback
from pathlib import Path

import pyproj

import orthoseg.model.model_factory as mf
from orthoseg.helpers import config_helper as conf, email_helper
from orthoseg.util import log_util, ows_util

# Get a logger...
logger = logging.getLogger(__name__)


def _load_images_argstr(argstr):
    args = shlex.split(argstr)
    _load_images_args(args)


def _load_images_args(args):
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

    # Interprete arguments
    args = parser.parse_args(args)

    # Run!
    load_images(config_path=Path(args.config), config_overrules=args.config_overrules)


def load_images(
    config_path: Path,
    load_testsample_images: bool = False,
    config_overrules: list[str] = [],
):
    """Load and cache images for a segmentation project.

    Args:
        config_path (Path): Path to the projects config file.
        load_testsample_images (bool, optional): True to only load testsample
            images. Defaults to False.
        config_overrules (list[str], optional): list of config options that will
            overrule other ways to supply configuration. They should be specified in the
            form of "<section>.<parameter>=<value>". Defaults to [].
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
    global logger
    logger = log_util.main_log_init(conf.dirs.getpath("log_dir"), __name__)

    # Log + send email
    message = f"Start load_images for config {config_path.stem}"
    logger.info(message)
    logger.debug(f"Config used: \n{conf.pformat_config()}")
    email_helper.sendmail(message)

    try:
        # Use different setting depending if testsample or all images
        if load_testsample_images:
            output_image_dir = conf.dirs.getpath("predictsample_image_input_dir")

            # Use the same image size as for the training, that is the most
            # convenient to check the quality
            image_pixel_width = conf.train.getint("image_pixel_width")
            image_pixel_height = conf.train.getint("image_pixel_height")
            image_pixel_x_size = conf.train.getfloat("image_pixel_x_size")
            image_pixel_y_size = conf.train.getfloat("image_pixel_y_size")
            image_pixels_overlap = 0
            image_format = ows_util.FORMAT_JPEG

            # To create the testsample, fetch only on every ... images
            nb_images_to_skip = 50

        else:
            output_image_dir = conf.dirs.getpath("predict_image_input_dir")

            # Get the image size for the predict
            image_pixel_width = conf.predict.getint("image_pixel_width")
            image_pixel_height = conf.predict.getint("image_pixel_height")
            image_pixel_x_size = conf.predict.getfloat("image_pixel_x_size")
            image_pixel_y_size = conf.predict.getfloat("image_pixel_y_size")
            image_pixels_overlap = conf.predict.getint("image_pixels_overlap", 0)
            image_format = ows_util.FORMAT_JPEG

            # For the real prediction dataset, no skipping obviously...
            nb_images_to_skip = 0

        # Validate the image size for the model architecture
        input_width_pred = image_pixel_width + 2 * image_pixels_overlap
        input_height_pred = image_pixel_height + 2 * image_pixels_overlap
        mf.check_image_size(
            architecture=conf.model.get("architecture"),
            input_width=input_width_pred,
            input_height=input_height_pred,
        )

        # Get ssl_verify setting
        ssl_verify = conf.general["ssl_verify"]
        # Get the download cron schedule
        download_cron_schedule = conf.download["cron_schedule"]

        # Get the layer info
        predict_layer = conf.predict["image_layer"]
        layersources = conf.image_layers[predict_layer]["layersources"]
        nb_concurrent_calls = conf.image_layers[predict_layer]["nb_concurrent_calls"]
        crs = pyproj.CRS.from_user_input(conf.image_layers[predict_layer]["projection"])
        bbox = conf.image_layers[predict_layer]["bbox"]
        grid_xmin = conf.image_layers[predict_layer]["grid_xmin"]
        grid_ymin = conf.image_layers[predict_layer]["grid_ymin"]
        image_pixels_ignore_border = conf.image_layers[predict_layer][
            "image_pixels_ignore_border"
        ]
        roi_filepath = conf.image_layers[predict_layer]["roi_filepath"]

        # Now we are ready to get the images...
        ows_util.get_images_for_grid(
            layersources=layersources,
            output_image_dir=output_image_dir,
            crs=crs,
            image_gen_bbox=bbox,
            image_gen_roi_filepath=roi_filepath,
            grid_xmin=grid_xmin,
            grid_ymin=grid_ymin,
            image_crs_pixel_x_size=image_pixel_x_size,
            image_crs_pixel_y_size=image_pixel_y_size,
            image_pixel_width=image_pixel_width,
            image_pixel_height=image_pixel_height,
            image_pixels_ignore_border=image_pixels_ignore_border,
            nb_concurrent_calls=nb_concurrent_calls,
            cron_schedule=download_cron_schedule,
            image_format=image_format,
            pixels_overlap=image_pixels_overlap,
            nb_images_to_skip=nb_images_to_skip,
            ssl_verify=ssl_verify,
        )

        # Log and send mail
        message = f"Completed load_images for config {config_path.stem}"
        logger.info(message)
        email_helper.sendmail(message)
    except Exception as ex:
        message = f"ERROR while running load_images for task {config_path.stem}"
        logger.exception(message)
        email_helper.sendmail(
            subject=message, body=f"Exception: {ex}\n\n {traceback.format_exc()}"
        )
        raise Exception(message) from ex
    finally:
        if conf.tmp_dir is not None:
            shutil.rmtree(conf.tmp_dir, ignore_errors=True)


def main():
    """Run load images."""
    try:
        _load_images_args(sys.argv[1:])
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

"""Script to load images to a cache directory."""

import argparse
import logging
import sys
import traceback
from pathlib import Path

import pyproj

import orthoseg.model.model_factory as mf
from orthoseg.helpers import config_helper as conf, email_helper
from orthoseg.util import image_util, log_util

# Get a logger...
logger = logging.getLogger(__name__)


def _load_images_args(args) -> argparse.Namespace:
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
            "Supply any number of config overrules like this: <section>.<key>=<value>"
        ),
    )

    return parser.parse_args(args)


def load_images(
    config_path: Path,
    load_testsample_images: bool = False,
    config_overrules: list[str] | None = None,
):
    """Load and cache images for a segmentation project.

    Args:
        config_path (Path): Path to the projects config file.
        load_testsample_images (bool, optional): True to only load testsample
            images. Defaults to False.
        config_overrules (list[str], optional): list of config options that will
            overrule other ways to supply configuration. They should be specified in the
            form of "<section>.<key>=<value>". Defaults to None.
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
    global logger  # noqa: PLW0603
    logger = log_util.main_log_init(conf.dirs.getpath("log_dir"), __name__)

    # Log + send email
    message = f"Start load_images for {config_path.stem}"
    logger.info(message)
    logger.debug(f"Config used: \n{conf.pformat_config()}")
    email_helper.sendmail(message)

    try:
        # Use different setting depending if testsample or all images
        if load_testsample_images:
            output_image_dir_str = conf.dirs.get("predictsample_image_input_dir")

            # Use the same image size as for the training, that is the most
            # convenient to check the quality
            image_pixel_width = conf.train.getint("image_pixel_width")
            image_pixel_height = conf.train.getint("image_pixel_height")
            image_pixel_x_size = conf.train.getfloat("image_pixel_x_size")
            image_pixel_y_size = conf.train.getfloat("image_pixel_y_size")
            image_pixels_overlap = 0
            image_format = image_util.FORMAT_JPEG

            # To create the testsample, fetch only on every ... images
            nb_images_to_skip = 50

        else:
            output_image_dir_str = conf.dirs.get("predict_image_input_dir")

            # Get the image size for the predict
            image_pixel_width = conf.predict.getint("image_pixel_width")
            image_pixel_height = conf.predict.getint("image_pixel_height")
            image_pixel_x_size = conf.predict.getfloat("image_pixel_x_size")
            image_pixel_y_size = conf.predict.getfloat("image_pixel_y_size")
            image_pixels_overlap = conf.predict.getint("image_pixels_overlap", 0)

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
        ssl_verify = conf.general.get("ssl_verify", True)
        # Get the download cron schedule
        download_cron_schedule = conf.download["cron_schedule"]

        predict_layers = conf.predict.getlist("image_layer")
        for predict_layer in predict_layers:
            if predict_layer not in conf.image_layers:
                raise ValueError(f"{predict_layer=} is not configured in image_layers")

            image_layer_config = conf.image_layers[predict_layer]
            layersources = image_layer_config["layersources"]
            nb_concurrent_calls = image_layer_config["nb_concurrent_calls"]
            crs = pyproj.CRS.from_user_input(image_layer_config["projection"])
            switch_axes = image_layer_config["switch_axes"]
            bbox = image_layer_config["bbox"]
            grid_xmin = image_layer_config["grid_xmin"]
            grid_ymin = image_layer_config["grid_ymin"]
            image_pixels_ignore_border = image_layer_config[
                "image_pixels_ignore_border"
            ]
            roi_filepath = image_layer_config["roi_filepath"]
            image_format = image_layer_config.get(
                "image_format", image_util.FORMAT_JPEG
            )

            # Keep cached images separated per configured layer.
            layer_output_image_dir = Path(
                output_image_dir_str.format(predict_image_layer=predict_layer)
            )

            image_util.load_images_to_cache(
                layersources=layersources,
                output_image_dir=layer_output_image_dir,
                crs=crs,
                switch_axes=switch_axes,
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
        message = f"Completed load_images for {config_path.stem}"
        logger.info(message)
        email_helper.sendmail(message)
    except Exception as ex:
        message = f"ERROR in load_images for {config_path.stem}"
        logger.exception(message)
        email_helper.sendmail(
            subject=message, body=f"Exception: {ex}\n\n {traceback.format_exc()}"
        )
        raise RuntimeError(f"{message}: {ex}") from ex
    finally:
        conf.remove_run_tmp_dir()


def main():
    """Run load images."""
    try:
        # Interprete arguments
        args = _load_images_args(sys.argv[1:])

        # Run!
        load_images(
            config_path=Path(args.config), config_overrules=args.config_overrules
        )
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


# If the script is ran directly...
if __name__ == "__main__":
    main()

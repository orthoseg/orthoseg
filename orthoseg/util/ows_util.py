# -*- coding: utf-8 -*-
"""
Module with generic usable utility functions to make some tasks using OWS services
easier.
"""

import concurrent.futures as futures
import datetime
import itertools as it
import logging
import math
from pathlib import Path
import random
import time
from typing import List, Optional, Tuple, Union
import urllib3
import warnings

import numpy as np
import geofileops as gfo
import geofileops.util.grid_util
import owslib
import owslib.wms
import owslib.util
import pycron
import pyproj
import rasterio as rio
import rasterio.errors as rio_errors
from rasterio import profiles as rio_profiles
from rasterio import transform as rio_transform
from rasterio import windows as rio_windows
import geopandas as gpd
import shapely.geometry as sh_geom

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
FORMAT_GEOTIFF = "image/geotiff"
FORMAT_GEOTIFF_EXT = ".tif"
FORMAT_GEOTIFF_EXT_WORLD = ".tfw"

FORMAT_TIFF = "image/tiff"
FORMAT_TIFF_EXT = ".tif"
FORMAT_TIFF_EXT_WORLD = ".tfw"

FORMAT_JPEG = "image/jpeg"
FORMAT_JPEG_EXT = ".jpg"
FORMAT_JPEG_EXT_WORLD = ".jgw"

FORMAT_PNG = "image/png"
FORMAT_PNG_EXT = ".png"
FORMAT_PNG_EXT_WORLD = ".pgw"

# Get a logger...
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def get_images_for_grid(
    layersources: List[dict],
    output_image_dir: Path,
    crs: pyproj.CRS,
    image_gen_bounds: Optional[Tuple[float, float, float, float]] = None,
    image_gen_roi_filepath: Optional[Path] = None,
    grid_xmin: float = 0.0,
    grid_ymin: float = 0.0,
    image_crs_pixel_x_size: float = 0.25,
    image_crs_pixel_y_size: float = 0.25,
    image_pixel_width: int = 1024,
    image_pixel_height: int = 1024,
    image_pixels_ignore_border: int = 0,
    nb_concurrent_calls: int = 1,
    cron_schedule: Optional[str] = None,
    image_format: str = FORMAT_GEOTIFF,
    image_format_save: Optional[str] = None,
    tiff_compress: str = "lzw",
    transparent: bool = False,
    pixels_overlap: int = 0,
    column_start: int = 0,
    nb_images_to_skip: int = 0,
    max_nb_images: int = -1,
    ssl_verify: Union[bool, str] = True,
    force: bool = False,
):
    """
    Loads all images in a grid from a WMS service.

    Args:
        layersources (List[dict]): Layer sources to get images from. Multiple
            sources can be specified to create a combined image, eg. use band
            1 of a WMS service with band 2 and 3 of another one.
        output_image_dir (Path): Directory to save the images to.
        crs (pyproj.CRS): The crs of the source and destination images.
        image_gen_bounds (Tuple[float, float, float, float], optional): Bounds
            of the roi to request/save images for. Defaults to None.
        image_gen_roi_filepath (Optional[Path], optional): File with the roi
            where images should be requested/saved for. Defaults to None.
        grid_xmin (float, optional): xmin for the grid to be used.
            Defaults to 0.0.
        grid_ymin (float, optional): ymin for the grid to be used.
            Defaults to 0.0.
        image_crs_pixel_x_size (float, optional): [description]. Defaults to 0.25.
        image_crs_pixel_y_size (float, optional): [description]. Defaults to 0.25.
        image_pixel_width (int, optional): [description]. Defaults to 1024.
        image_pixel_height (int, optional): [description]. Defaults to 1024.
        image_pixels_ignore_border (int, optional): [description]. Defaults to 0.
        nb_concurrent_calls (int, optional): Number of images to treat in
            parallel. Will increase the load on the WMS server! Defaults to 1.
        cron_schedule (str, optional): [description]. Defaults to None.
        image_format (str, optional): The image format to get. Defaults to
            FORMAT_GEOTIFF.
        image_format_save (str, optional): The image format to save to.
            Defaults to None.
        tiff_compress (str, optional): [description]. Defaults to 'lzw'.
        transparent (bool, optional): [description]. Defaults to False.
        pixels_overlap (int, optional): [description]. Defaults to 0.
        column_start (int, optional): [description]. Defaults to 0.
        nb_images_to_skip (int, optional): [description]. Defaults to 0.
        max_nb_images (int, optional): [description]. Defaults to -1.
        ssl_verify (bool or str, optional): True to use the default
            certificate bundle as installed on your system. False disables
            certificate validation (NOT recommended!). If a path to a
            certificate bundle file (.pem) is passed, this will be used.
            In corporate networks using a proxy server this is often needed
            to evade CERTIFICATE_VERIFY_FAILED errors. Defaults to True.
        force (bool, optional): [description]. Defaults to False.

    Raises:
        Exception: [description]
    """

    # Init
    if image_format_save is None:
        image_format_save = image_format

    # Interprete ssl_verify
    auth = None
    if ssl_verify is not None:
        # If it is a string, make sure it isn't actually a bool
        if isinstance(ssl_verify, str):
            if ssl_verify.lower() == "true":
                ssl_verify = True
            elif ssl_verify.lower() == "false":
                ssl_verify = False

        assert isinstance(ssl_verify, bool)
        auth = owslib.util.Authentication(verify=ssl_verify)
        if ssl_verify is False:
            urllib3.disable_warnings()
            logger.warn("SSL VERIFICATION IS TURNED OFF!!!")

    crs_width = math.fabs(
        image_pixel_width * image_crs_pixel_x_size
    )  # tile width in units of crs => 500 m
    crs_height = math.fabs(
        image_pixel_height * image_crs_pixel_y_size
    )  # tile height in units of crs => 500 m
    if cron_schedule is not None and cron_schedule != "":
        logger.info(
            "A cron_schedule was specified, so the download will only proceed in the "
            f"specified time range: {cron_schedule}"
        )

    # Read the region of interest file if provided
    roi_gdf = None
    if image_gen_roi_filepath is not None:
        # Read roi
        logger.info(f"Read + optimize region of interest from {image_gen_roi_filepath}")
        roi_gdf = gpd.read_file(str(image_gen_roi_filepath))

        # If the generate_window wasn't specified, calculate the bounds
        # based on the roi (but make sure they fit the grid!)
        if image_gen_bounds is None:
            roi_bounds = roi_gdf.geometry.total_bounds
            image_gen_bounds = (
                roi_bounds[0] - ((roi_bounds[0] - grid_xmin) % crs_width),
                roi_bounds[1] - ((roi_bounds[1] - grid_ymin) % crs_height),
                roi_bounds[2] + (grid_xmin - ((roi_bounds[2] - grid_xmin) % crs_width)),
                roi_bounds[3]
                + (grid_ymin - ((roi_bounds[3] - grid_ymin) % crs_height)),
            )
            logger.debug(
                f"roi_bounds: {roi_bounds}, image_gen_bounds: {image_gen_bounds}"
            )

    # If there is still no image_gen_bounds, stop.
    if image_gen_bounds is None:
        raise Exception(
            "Either image_gen_bounds or an image_gen_roi_filepath should be specified."
        )

    # Check if the image_gen_bounds are compatible with the grid...
    if (image_gen_bounds[0] - grid_xmin) % crs_width != 0:
        xmin_new = image_gen_bounds[0] - ((image_gen_bounds[0] - grid_xmin) % crs_width)
        logger.warning(
            f"xmin {image_gen_bounds[0]} incompatible with grid, {xmin_new} used"
        )
        image_gen_bounds = (
            xmin_new,
            image_gen_bounds[1],
            image_gen_bounds[2],
            image_gen_bounds[3],
        )
    if (image_gen_bounds[1] - grid_ymin) % crs_height != 0:
        ymin_new = image_gen_bounds[1] - (
            (image_gen_bounds[1] - grid_ymin) % crs_height
        )
        logger.warning(
            f"ymin {image_gen_bounds[1]} incompatible with grid, {ymin_new} used"
        )
        image_gen_bounds = (
            image_gen_bounds[0],
            ymin_new,
            image_gen_bounds[2],
            image_gen_bounds[3],
        )
    if (image_gen_bounds[2] - grid_xmin) % crs_width != 0:
        xmax_new = (
            image_gen_bounds[2]
            + crs_width
            - ((image_gen_bounds[2] - grid_xmin) % crs_width)
        )
        logger.warning(
            f"xmax {image_gen_bounds[2]} incompatible with grid, {xmax_new} used"
        )
        image_gen_bounds = (
            image_gen_bounds[0],
            image_gen_bounds[1],
            xmax_new,
            image_gen_bounds[3],
        )
    if (image_gen_bounds[3] - grid_ymin) % crs_height != 0:
        ymax_new = (
            image_gen_bounds[3]
            + crs_height
            - ((image_gen_bounds[3] - grid_ymin) % crs_height)
        )
        logger.warning(
            f"ymax {image_gen_bounds[3]} incompatible with grid, {ymax_new} used"
        )
        image_gen_bounds = (
            image_gen_bounds[0],
            image_gen_bounds[1],
            image_gen_bounds[2],
            ymax_new,
        )

    # Create the output dir if it doesn't exist yet
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # Write the cache image grid to file
    grid_path = output_image_dir / "imagecache_grid.gpkg"
    if not grid_path.exists():
        grid_for_roi_gdf = geofileops.util.grid_util.create_grid3(
            image_gen_bounds, width=crs_width, height=crs_height, crs=crs
        )
        gfo.to_file(grid_for_roi_gdf, grid_path)

    # Calculate width and height...
    dx = math.fabs(
        image_gen_bounds[0] - image_gen_bounds[2]
    )  # area width in units of crs
    dy = math.fabs(
        image_gen_bounds[1] - image_gen_bounds[3]
    )  # area height in units of crs
    cols = int(math.ceil(dx / crs_width)) + 1
    rows = int(math.ceil(dy / crs_height)) + 1

    # If an roi is defined, split it using a grid as large objects are small
    if roi_gdf is not None:
        # TODO: support creating a grid in latlon????
        if crs.is_projected:

            # Create grid
            grid_for_roi_gdf = geofileops.util.grid_util.create_grid3(
                image_gen_bounds, width=crs_width, height=crs_height, crs=crs
            )

            # Create intersection layer between grid and roi
            roi_gdf = gpd.overlay(grid_for_roi_gdf, roi_gdf, how="intersection")

            # Explode possible multipolygons to polygons
            # roi_gdf.reset_index(drop=True, inplace=True)
            # assert to evade pyLance warning
            assert isinstance(roi_gdf, gpd.GeoDataFrame)
            roi_gdf = roi_gdf.explode(ignore_index=True)
            # roi_gdf.reset_index(drop=True, inplace=True)

            # Write to file
            grid_for_roi_path = output_image_dir / "grid_for_roi.gpkg"
            if grid_for_roi_path.exists() is False:
                gfo.to_file(grid_for_roi_gdf, grid_for_roi_path)
            assert isinstance(roi_gdf, gpd.GeoDataFrame)
            gridded_roi_path = output_image_dir / "gridded_roi.gpkg"
            if gridded_roi_path.exists() is False:
                gfo.to_file(roi_gdf, gridded_roi_path)

    # Inits to start getting images
    layersources_prepared = []
    for layersource in layersources:
        wms_service = owslib.wms.WebMapService(
            url=layersource["wms_server_url"],
            version=layersource["wms_version"],
            username=layersource["wms_username"],
            password=layersource["wms_password"],
            auth=auth,
        )
        if layersource["wms_ignore_capabilities_url"]:
            # If the wms url in capabilities should be ignored,
            # overwrite with original url
            nb = len(wms_service.getOperationByName("GetMap").methods)
            for method_id in range(nb):
                wms_service.getOperationByName("GetMap").methods[method_id][
                    "url"
                ] = layersource["wms_server_url"]
        layersources_prepared.append(
            LayerSource(
                wms_service=wms_service,
                layernames=layersource["layernames"],
                layerstyles=layersource["layerstyles"],
                bands=layersource["bands"],
                random_sleep=layersource["random_sleep"],
            )
        )

    with futures.ThreadPoolExecutor(nb_concurrent_calls) as pool:

        # Loop through all columns and get the images...
        logger.info("Start loading images")
        start_time = None
        start_time_lastprogress = None
        nb_todo = cols * rows
        nb_processed = 0
        nb_processed_lastprogress = 0
        nb_downloaded = 0
        nb_ignore_in_progress = 0
        bbox_list = []
        size_list = []
        output_filename_list = []
        output_dir_list = []
        for col in range(column_start, cols):

            image_xmin = col * crs_width + image_gen_bounds[0]
            image_xmax = (col + 1) * crs_width + image_gen_bounds[0]

            # If overlapping images are wanted... increase image bbox
            if pixels_overlap:
                image_xmin = image_xmin - (pixels_overlap * image_crs_pixel_x_size)
                image_xmax = image_xmax + (pixels_overlap * image_crs_pixel_x_size)

            # Put all the images of this column in a dir
            if crs.is_projected:
                output_dir = output_image_dir / f"{image_xmin:06.0f}"
            else:
                output_dir = output_image_dir / f"{image_xmin:09.4f}"
            if not output_dir.exists():
                output_dir.mkdir()

            logger.debug(
                f"load_images to {output_image_dir.parent.name}/{output_image_dir.name}"
                f", column {col} ({output_dir.name})"
            )
            for row in range(0, rows):

                # If a cron_schedule is specified, check if we should be running
                if cron_schedule is not None and cron_schedule != "":
                    # Sleep till the schedule becomes active
                    first_time = True
                    while not pycron.is_now(cron_schedule):
                        # The first time, log message that we are going to sleep...
                        if first_time is True:
                            logger.info(
                                f"Time schedule specified: sleep: {cron_schedule}"
                            )
                            first_time = False
                        time.sleep(60)

                nb_processed += 1

                # Calculate y bounds
                image_ymin = row * crs_height + image_gen_bounds[1]
                image_ymax = (row + 1) * crs_height + image_gen_bounds[1]

                # If overlapping images are wanted... increase image bbox
                if pixels_overlap:
                    image_ymin = image_ymin - (pixels_overlap * image_crs_pixel_y_size)
                    image_ymax = image_ymax + (pixels_overlap * image_crs_pixel_y_size)

                # Create output filename
                output_filename = _create_filename(
                    crs=crs,
                    bbox=(image_xmin, image_ymin, image_xmax, image_ymax),
                    size=(
                        image_pixel_width + 2 * pixels_overlap,
                        image_pixel_height + 2 * pixels_overlap,
                    ),
                    image_format=image_format,
                    layername=None,
                )
                output_filepath = output_dir / output_filename

                # Do some checks to know if the image needs to be downloaded
                image_to_be_skipped = False
                if nb_images_to_skip > 0 and (nb_processed % nb_images_to_skip) != 0:
                    # If we need to skip images, do so...
                    nb_ignore_in_progress += 1
                    image_to_be_skipped = True
                elif (
                    not force
                    and output_filepath.exists()
                    and output_filepath.stat().st_size > 0
                ):
                    # Image exists already
                    nb_ignore_in_progress += 1
                    image_to_be_skipped = True
                    logger.debug("    -> image exists already, so skip")
                elif roi_gdf is not None:
                    # If roi was provided, check first if the current image overlaps
                    image_shape = sh_geom.box(
                        image_xmin, image_ymin, image_xmax, image_ymax
                    )

                    possible_match_indexes = list(
                        roi_gdf.geometry.sindex.query(image_shape)
                    )
                    possible_matches_gdf = roi_gdf.iloc[possible_match_indexes]
                    precise_matches_gdf = possible_matches_gdf.loc[
                        possible_matches_gdf.intersects(image_shape)
                    ]

                    if len(precise_matches_gdf) == 0:
                        nb_ignore_in_progress += 1
                        logger.debug("    -> image doesn't overlap with roi, so skip")
                        image_to_be_skipped = True

                # If the image doesn't need to be skipped... append the image
                # info to the batch arrays so they can be treated in
                # bulk if the batch size is reached
                if image_to_be_skipped is False:
                    bbox_list.append((image_xmin, image_ymin, image_xmax, image_ymax))
                    size_list.append(
                        (
                            image_pixel_width + 2 * pixels_overlap,
                            image_pixel_height + 2 * pixels_overlap,
                        )
                    )
                    output_filename_list.append(output_filename)
                    output_dir_list.append(output_dir)

                # If the batch size is reached or we are at the last images
                nb_images_in_batch = len(bbox_list)
                if nb_images_in_batch == nb_concurrent_calls or nb_processed == (
                    nb_todo - 1
                ):

                    # Now we are getting to start fetching images... init start_time
                    if start_time is None or start_time_lastprogress is None:
                        start_time = datetime.datetime.now()
                        nb_processed_lastprogress = nb_processed

                    # Exec in parallel
                    read_results = pool.map(
                        getmap_to_file,  # Function
                        it.repeat(layersources_prepared),
                        output_dir_list,
                        it.repeat(crs),
                        bbox_list,
                        size_list,
                        it.repeat(image_format),
                        it.repeat(image_format_save),
                        output_filename_list,
                        it.repeat(transparent),
                        it.repeat(tiff_compress),
                        it.repeat(image_pixels_ignore_border),
                        it.repeat(force),
                    )

                    for _ in read_results:
                        nb_downloaded += 1

                    # Progress
                    logger.debug(
                        f"Process image {nb_processed} out of {cols*rows}: "
                        f"{nb_processed/(cols*rows):.2f} %"
                    )

                    # Log the progress and prediction speed
                    time_between_progress_s = 60
                    time_passed_s = (
                        datetime.datetime.now() - start_time
                    ).total_seconds()
                    if start_time_lastprogress is None:
                        time_passed_lastprogress_s = time_between_progress_s
                    else:
                        time_passed_lastprogress_s = (
                            datetime.datetime.now() - start_time_lastprogress
                        ).total_seconds()
                    if (
                        time_passed_s > 0
                        and time_passed_lastprogress_s >= time_between_progress_s
                    ):
                        nb_per_hour = (
                            (nb_processed - nb_ignore_in_progress) / time_passed_s
                        ) * 3600
                        nb_per_hour_lastprogress = (
                            nb_processed_lastprogress
                            - nb_processed / time_passed_lastprogress_s
                        ) * 3600
                        hours_to_go = (int)((nb_todo - nb_processed) / nb_per_hour)
                        min_to_go = (int)(
                            (((nb_todo - nb_processed) / nb_per_hour) % 1) * 60
                        )
                        progress_message = (
                            f"load_images to {output_image_dir.parent.name}/"
                            f"{output_image_dir.name}, {hours_to_go:3d}:{min_to_go:2d} "
                            f"left for {nb_todo-nb_processed} images at "
                            f"{nb_per_hour:0.0f}/h ({nb_per_hour_lastprogress:0.0f}/h "
                            f"last batch), with {nb_ignore_in_progress} skipped"
                        )
                        logger.info(progress_message)

                        start_time_lastprogress = datetime.datetime.now()
                        nb_processed_lastprogress = nb_processed

                    # Reset variable for next batch
                    bbox_list = []
                    size_list = []
                    output_filename_list = []
                    output_dir_list = []

                    if max_nb_images > -1 and nb_downloaded >= max_nb_images:
                        return


class LayerSource:
    def __init__(
        self,
        wms_service: Union[
            owslib.wms.wms111.WebMapService_1_1_1, owslib.wms.wms130.WebMapService_1_3_0
        ],
        layernames: List[str],
        layerstyles: Optional[List[str]] = None,
        bands: Optional[List[int]] = None,
        random_sleep: int = 0,
    ):
        self.wms_service = wms_service
        self.layernames = layernames
        self.layerstyles = layerstyles
        self.bands = bands
        self.random_sleep = random_sleep


def getmap_to_file(
    layersources: List[LayerSource],
    output_dir: Path,
    crs: Union[str, pyproj.CRS],
    bbox: Tuple[float, float, float, float],
    size: Tuple[int, int],
    image_format: str = FORMAT_GEOTIFF,
    image_format_save: Optional[str] = None,
    output_filename: Optional[str] = None,
    transparent: bool = False,
    tiff_compress: str = "lzw",
    image_pixels_ignore_border: int = 0,
    force: bool = False,
    layername_in_filename: bool = False,
) -> Optional[Path]:
    """


    Args:
        layersources (List[dict]): Layer sources to get images from. Multiple
            sources can be specified to create a combined image, eg. use band
            1 of a WMS service with band 2 and 3 of another one.
        output_image_dir (Path): Directory to save the images to.
        crs (pyproj.CRS): The crs of the source and destination images.
        bbox (Tuple[float, float, float, float]): Bbox of the image to get.
        size (Tuple[int, int]): The image width and height.
        image_format (str, optional): [description]. Defaults to FORMAT_GEOTIFF.
        image_format_save (str, optional): [description]. Defaults to None.
        output_filename (str, optional): [description]. Defaults to None.
        transparent (bool, optional): [description]. Defaults to False.
        tiff_compress (str, optional): [description]. Defaults to 'lzw'.
        image_pixels_ignore_border (int, optional): [description]. Defaults to 0.
        force (bool, optional): [description]. Defaults to False.
        layername_in_filename (bool, optional): [description]. Defaults to False.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        Optional[Path]: [description]
    """

    # Init
    # If no seperate save format is specified, use the standard image_format
    if image_format_save is None:
        image_format_save = image_format

    # If crs is specified as str, convert to CRS
    if isinstance(crs, str):
        crs = pyproj.CRS(crs)

    # If there isn't a filename supplied, create one...
    if output_filename is None:
        layername = None
        if layername_in_filename:
            for layersource in layersources:
                if layername is None:
                    layername = "_".join(layersource.layernames)
                else:
                    layername += f"_{'_'.join(layersource.layernames)}"

        output_filename = _create_filename(
            crs=crs,
            bbox=bbox,
            size=size,
            image_format=image_format_save,
            layername=layername,
        )

    # Create full output filepath
    output_filepath = output_dir / output_filename

    # If force is false and file exists already, stop...
    if force is False and output_filepath.exists():
        if output_filepath.stat().st_size > 0:
            logger.debug(f"File already exists, skip: {output_filepath}")
            return None
        else:
            output_filepath.unlink()

    logger.debug(f"Get image to {output_filepath}")

    if not output_dir.exists():
        output_dir.mkdir()

    # Get image(s), read the band to keep and save
    # Some hacks for special cases...
    bbox_for_getmap = bbox
    size_for_getmap = size
    # Dirty hack to ask a bigger picture, and then remove the border again!
    if image_pixels_ignore_border > 0:
        x_pixsize = (bbox[2] - bbox[0]) / size[0]
        y_pixsize = (bbox[3] - bbox[1]) / size[1]
        bbox_for_getmap = (
            bbox[0] - x_pixsize * image_pixels_ignore_border,
            bbox[1] - y_pixsize * image_pixels_ignore_border,
            bbox[2] + x_pixsize * image_pixels_ignore_border,
            bbox[3] + y_pixsize * image_pixels_ignore_border,
        )
        size_for_getmap = (
            size[0] + 2 * image_pixels_ignore_border,
            size[1] + 2 * image_pixels_ignore_border,
        )

    # For coordinate systems with switched axis (y, x or lon, lat), switch x and y
    if _has_switched_axes(crs):
        bbox_for_getmap = (
            bbox_for_getmap[1],
            bbox_for_getmap[0],
            bbox_for_getmap[3],
            bbox_for_getmap[2],
        )

    image_data_output = None
    image_profile_output = None
    response = None
    for layersource in layersources:

        # Get image from server, and retry up to 10 times...
        nb_retries = 0
        time_sleep = 5
        image_retrieved = False
        while image_retrieved is False:

            try:
                logger.debug(f"Start call GetMap for bbox {bbox}")
                response = layersource.wms_service.getmap(
                    layers=layersource.layernames,
                    styles=layersource.layerstyles,
                    srs=f"epsg:{crs.to_epsg()}",
                    bbox=bbox_for_getmap,
                    size=size_for_getmap,
                    format=image_format,
                    transparent=transparent,
                )
                logger.debug(f"Finished doing request {response.geturl()}")

                # If a random sleep was specified... apply it
                if layersource.random_sleep > 0:
                    time.sleep(random.uniform(0, layersource.random_sleep))

                # Image was retrieved... so stop loop
                image_retrieved = True
            except Exception as ex:
                if isinstance(ex, owslib.util.ServiceException):
                    if "Error rendering coverage on the fast path" in str(ex):
                        logger.error(
                            f"Request for bbox {bbox_for_getmap} gave a non-blocking "
                            f"exception, SKIP and proceed: {ex}"
                        )
                        return
                    elif "java.lang.OutOfMemoryError: Java heap space" in str(ex):
                        logger.debug(
                            f"Request for bbox {bbox_for_getmap} gave a non-blocking "
                            f"exception, try again in {time_sleep} s: {ex}"
                        )
                    else:
                        raise Exception(
                            f"WMS Service gave error for bbox {bbox_for_getmap}: {ex}"
                        ) from ex

                # If the exception isn't handled yet, retry 10 times...
                if nb_retries < 10:
                    time.sleep(time_sleep)

                    # Increase sleep time every time.
                    time_sleep += 5
                    nb_retries += 1
                    continue
                else:
                    message = (
                        "Retried 10 times and didn't work, with layers: "
                        f"{layersource.layernames}, styles: {layersource.layernames}, "
                        f"for bbox: {bbox_for_getmap}"
                    )
                    logger.exception(message)
                    raise Exception(message) from ex

        # If the response is None, error
        if response is None:
            raise Exception("No valid response retrieved...")

        # Write image to temp file...
        # If all bands need to be kept, just save to output
        with rio.MemoryFile(response.read()) as memfile:

            # Because the image returned by WMS doesn't contain georeferencing
            # info, suppress NotGeoreferencedWarning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=rio_errors.NotGeoreferencedWarning
                )

                with memfile.open() as image_ds:
                    image_profile_curr = image_ds.profile

                    # Read the data we need from the memoryfile
                    if layersource.bands is None:
                        # If no specific bands specified, read them all...
                        if image_data_output is None:
                            image_data_output = image_ds.read()
                        else:
                            image_data_output = np.append(
                                image_data_output, image_ds.read(), axis=0
                            )
                    elif len(layersource.bands) == 1 and layersource.bands[0] == -1:
                        # If 1 band, -1 specified: dirty hack to use greyscale version
                        # of rgb image
                        image_data_tmp = image_ds.read()
                        image_data_grey = np.mean(image_data_tmp, axis=0).astype(
                            image_data_tmp.dtype
                        )
                        new_shape = (
                            1,
                            image_data_grey.shape[0],
                            image_data_grey.shape[1],
                        )
                        image_data_grey = np.reshape(image_data_grey, new_shape)
                        if image_data_output is None:
                            image_data_output = image_data_grey
                        else:
                            image_data_output = np.append(
                                image_data_output, image_data_grey, axis=0
                            )
                    else:
                        # If bands specified, only read the bands to keep...
                        for band in layersource.bands:
                            # Read the band needed + reshape
                            # Remark: rasterio uses 1-based indexing instead of 0-based
                            image_data_curr = image_ds.read(band + 1)
                            new_shape = (
                                1,
                                image_data_curr.shape[0],
                                image_data_curr.shape[1],
                            )
                            image_data_curr = np.reshape(image_data_curr, new_shape)

                            # Set or append to image_data_output
                            if image_data_output is None:
                                image_data_output = image_data_curr
                            else:
                                image_data_output = np.append(
                                    image_data_output, image_data_curr, axis=0
                                )

                    # Set output profile (# of bands will be corrected later if needed)
                    if image_profile_output is None:
                        image_profile_output = image_profile_curr

    # Write (temporary) output file
    # evade pyLance warning
    assert image_profile_output is not None
    assert isinstance(image_data_output, np.ndarray)

    # Set the number of bands to write correctly...
    if (
        image_format_save in [FORMAT_JPEG, FORMAT_PNG]
        and image_data_output.shape[0] == 2
    ):
        zero_band = np.zeros(
            shape=(1, image_data_output.shape[1], image_data_output.shape[2]),
            dtype=image_data_output.dtype,
        )
        image_data_output = np.append(image_data_output, zero_band, axis=0)

    assert isinstance(image_data_output, np.ndarray)
    image_profile_output.update(count=image_data_output.shape[0])
    image_profile_output = _get_cleaned_write_profile(image_profile_output)

    # Because the (temporary) output file doesn't contain coordinates (yet),
    # suppress NotGeoreferencedWarning while writing
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rio_errors.NotGeoreferencedWarning)

        with rio.open(str(output_filepath), "w", **image_profile_output) as image_file:
            image_file.write(image_data_output)

    # If an aux.xml file was written, remove it again...
    output_aux_path = output_filepath.parent / f"{output_filepath.name}.aux.xml"
    if output_aux_path.exists() is True:
        output_aux_path.unlink()

    # Make the output image compliant with image_format_save

    # If geotiff is asked, check if the the coordinates are embedded...
    if image_format_save == FORMAT_GEOTIFF:
        # Read output image to check if coördinates are there
        with rio.open(str(output_filepath)) as image_ds:
            image_profile_orig = image_ds.profile
            image_transform_affine = image_ds.transform

            if image_pixels_ignore_border == 0:
                image_data_output = image_ds.read()
            else:
                image_data_output = image_ds.read(
                    window=rio_windows.Window(
                        col_off=image_pixels_ignore_border,  # type: ignore
                        row_off=image_pixels_ignore_border,  # type: ignore
                        width=size[0],  # type: ignore
                        height=size[1],  # type: ignore
                    )
                )

        logger.debug(f"original image_profile: {image_profile_orig}")

        # If coordinates are not embedded add them, if image_pixels_ignore_border
        # change them
        if (
            image_transform_affine[2] == 0 and image_transform_affine[5] == 0
        ) or image_pixels_ignore_border > 0:
            logger.debug(
                f"Coordinates not in image, driver: {image_profile_orig['driver']}"
            )

            # If profile format is not gtiff, create new profile
            if image_profile_orig["driver"] != "GTiff":
                image_profile_gtiff = rio_profiles.DefaultGTiffProfile.defaults

                # Copy appropriate info from source file
                image_profile_gtiff.update(
                    count=image_profile_orig["count"],
                    width=size[0],
                    height=size[1],
                    nodata=image_profile_orig["nodata"],
                    dtype=image_profile_orig["dtype"],
                )
                image_profile = image_profile_gtiff
            else:
                image_profile = image_profile_orig

            # Set the asked compression
            image_profile.update(compress=tiff_compress)

            # For some coordinate systems apparently the axis ordered is wrong in LibOWS
            crs_pixel_x_size = (bbox[2] - bbox[0]) / size[0]
            crs_pixel_y_size = (bbox[1] - bbox[3]) / size[1]

            logger.debug(
                "Coordinates to put in geotiff:\n"
                + f"    - x-part of pixel width, W-E: {crs_pixel_x_size}\n"
                + "    - y-part of pixel width, W-E (0 if image is exactly N up): 0\n"
                + f"    - top-left x: {bbox[0]}\n"
                + "    - x-part of pixel height, N-S (0 if image is exactly N up): \n"
                + f"    - y-part of pixel height, N-S: {crs_pixel_y_size}\n"
                + f"    - top-left y: {bbox[3]}"
            )

            # Add transform and crs to the profile
            image_profile.update(
                transform=rio_transform.Affine(
                    crs_pixel_x_size, 0, bbox[0], 0, crs_pixel_y_size, bbox[3]
                ),
                crs=crs,
            )

            # Delete output file, and write again
            output_filepath.unlink()
            with rio.open(str(output_filepath), "w", **image_profile) as image_file:
                image_file.write(image_data_output)

    else:
        # For file formats that doesn't support coordinates, we add a worldfile
        crs_pixel_x_size = (bbox[2] - bbox[0]) / size[0]
        crs_pixel_y_size = (bbox[1] - bbox[3]) / size[1]

        path_noext = output_filepath.parent / output_filepath.stem
        ext_world = _get_world_ext_for_image_format(image_format_save)
        output_worldfile_filepath = Path(str(path_noext) + ext_world)

        with output_worldfile_filepath.open("w") as wld_file:
            wld_file.write(f"{crs_pixel_x_size}")
            wld_file.write("\n0.000")
            wld_file.write("\n0.000")
            wld_file.write(f"\n{crs_pixel_y_size}")
            wld_file.write(f"\n{bbox[0]}")
            wld_file.write(f"\n{bbox[3]}")

        # If the image format to save is different, or if a border needs to be ignored
        if image_format != image_format_save or image_pixels_ignore_border > 0:

            # Read image
            with rio.open(str(output_filepath)) as image_ds:
                image_profile_orig = image_ds.profile
                image_transform_affine = image_ds.transform

                # If border needs to be ignored, only read data we are interested in
                if image_pixels_ignore_border == 0:
                    image_data_output = image_ds.read()
                else:
                    image_data_output = image_ds.read(
                        window=rio_windows.Window(
                            col_off=image_pixels_ignore_border,  # type: ignore
                            row_off=image_pixels_ignore_border,  # type: ignore
                            width=size[0],  # type: ignore
                            height=size[1],  # type: ignore
                        )
                    )

            # If same save format, reuse profile
            if image_format == image_format_save:
                image_profile_curr = image_profile_orig
                if image_pixels_ignore_border != 0:
                    image_profile_curr.update(width=size[0], height=size[1])
            else:
                if image_format_save == FORMAT_TIFF:
                    driver = "GTiff"
                    compress = tiff_compress
                else:
                    raise Exception(
                        f"Unsupported image_format_save: {image_format_save}"
                    )
                image_profile_curr = rio_profiles.Profile(
                    width=size[0],
                    height=size[1],
                    count=image_profile_orig["count"],
                    nodata=image_profile_orig["nodata"],
                    dtype=image_profile_orig["dtype"],
                    compress=compress,
                    driver=driver,
                )

            # Delete output file, and write again
            output_filepath.unlink()
            image_profile_curr = _get_cleaned_write_profile(image_profile_curr)
            with rio.open(
                str(output_filepath), "w", **image_profile_curr
            ) as image_file:
                image_file.write(image_data_output)

    return output_filepath


def _create_filename(
    crs: pyproj.CRS, bbox, size, image_format: str, layername: Optional[str] = None
):

    # Get image extension based on format
    image_ext = _get_ext_for_image_format(image_format)

    # Use different file names for projected vs geographic crs
    if crs.is_projected:
        output_filename = (
            f"{bbox[0]:06.0f}_{bbox[1]:06.0f}_{bbox[2]:06.0f}_{bbox[3]:06.0f}_"
            f"{size[0]}_{size[1]}"
        )
    else:
        output_filename = (
            f"{bbox[0]:09.4f}_{bbox[1]:09.4f}_{bbox[2]:09.4f}_{bbox[3]:09.4f}_"
            f"{size[0]}_{size[1]}"
        )

    # Add layername if it is not None
    if layername is not None:
        output_filename += "_" + layername

    # Add extension
    output_filename += image_ext

    return output_filename


def _has_switched_axes(crs: pyproj.CRS):
    if len(crs.axis_info) < 2:
        logger.warning(
            f"has_switched_axes False: len(crs_31370.axis_info) < 2 for {crs}"
        )

    has_switched_axes_options = [
        {"abbrev": "x", "direction": "east", "has_switched_axes": False},
        {"abbrev": "y", "direction": "north", "has_switched_axes": False},
        {"abbrev": "x", "direction": "north", "has_switched_axes": True},
        {"abbrev": "y", "direction": "east", "has_switched_axes": True},
        {"abbrev": "lat", "direction": "north", "has_switched_axes": False},
        {"abbrev": "lon", "direction": "east", "has_switched_axes": False},
        {"abbrev": "lat", "direction": "east", "has_switched_axes": True},
        {"abbrev": "lon", "direction": "north", "has_switched_axes": True},
    ]
    for axis_info in crs.axis_info:
        for option in has_switched_axes_options:
            if (
                axis_info.abbrev.lower() == option["abbrev"]
                and axis_info.direction.lower() == option["direction"]
            ):
                return option["has_switched_axes"]

    logger.warning(f"has_switched_axes option not found, so assume False for {crs}")
    return False


def _get_ext_for_image_format(image_format: str) -> str:
    # Choose image extension based on format
    if image_format == FORMAT_GEOTIFF:
        return FORMAT_GEOTIFF_EXT
    elif image_format == FORMAT_TIFF:
        return FORMAT_TIFF_EXT
    elif image_format == FORMAT_JPEG:
        return FORMAT_JPEG_EXT
    elif image_format == FORMAT_PNG:
        return FORMAT_PNG_EXT
    else:
        raise Exception(
            f"get_ext_for_image_format for image format {image_format} not implemented"
        )


def _get_world_ext_for_image_format(image_format: str) -> str:
    # Choose image extension based on format
    if image_format == FORMAT_GEOTIFF:
        return FORMAT_GEOTIFF_EXT_WORLD
    elif image_format == FORMAT_TIFF:
        return FORMAT_TIFF_EXT_WORLD
    elif image_format == FORMAT_JPEG:
        return FORMAT_JPEG_EXT_WORLD
    elif image_format == FORMAT_PNG:
        return FORMAT_PNG_EXT_WORLD
    else:
        raise Exception(
            f"get_world_ext_for_image_format for format {image_format} not implemented"
        )


def _get_cleaned_write_profile(
    profile: rio_profiles.Profile,
) -> Union[dict, rio_profiles.Profile]:

    # Depending on the driver, different profile keys are supported
    if profile.get("driver") == "JPEG":
        # Don't copy profile keys to cleaned version that are not supported for JPEG
        profile_cleaned = {}
        for profile_key in profile:
            if profile_key not in [
                "blockxsize",
                "blockysize",
                "compress",
                "interleave",
                "photometric",
                "tiled",
            ]:
                profile_cleaned[profile_key] = profile[profile_key]
    elif profile.get("driver") == "PNG":
        # Don't copy profile keys to cleaned version that are not supported for JPEG
        profile_cleaned = {}
        for profile_key in profile:
            if profile_key not in ["blockxsize", "blockysize", "interleave", "tiled"]:
                profile_cleaned[profile_key] = profile[profile_key]
    else:
        profile_cleaned = profile.copy()

    return profile_cleaned

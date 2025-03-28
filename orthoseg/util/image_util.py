"""Module with generic usable utility functions to load images."""

import logging
import math
import random
import time
import warnings
from concurrent import futures
from pathlib import Path
from typing import Any

import geofileops as gfo
import geopandas as gpd
import numpy as np
import owslib
import owslib.util
import owslib.wms
import pycron
import pygeoops
import pyproj
import rasterio as rio
import rasterio.enums
import rasterio.errors as rio_errors
import urllib3
from rasterio import (
    profiles as rio_profiles,
    transform as rio_transform,
    windows as rio_windows,
)
from rasterio._err import CPLE_AppDefinedError
from shapely.geometry import box

from . import progress_util

FORMAT_GEOTIFF = "image/geotiff"
FORMAT_GEOTIFF_DRIVER = "Gtiff"
FORMAT_GEOTIFF_EXT = ".tif"
FORMAT_GEOTIFF_EXT_WORLD = ".tfw"

FORMAT_TIFF = "image/tiff"
FORMAT_TIFF_DRIVER = "Gtiff"
FORMAT_TIFF_EXT = ".tif"
FORMAT_TIFF_EXT_WORLD = ".tfw"

FORMAT_JPEG = "image/jpeg"
FORMAT_JPEG_DRIVER = "JPEG"
FORMAT_JPEG_EXT = ".jpg"
FORMAT_JPEG_EXT_WORLD = ".jgw"

FORMAT_PNG = "image/png"
FORMAT_PNG_DRIVER = "PNG"
FORMAT_PNG_EXT = ".png"
FORMAT_PNG_EXT_WORLD = ".pgw"

# Get a logger...
logger = logging.getLogger(__name__)


class WMSLayerSource:
    """Properties of a WMS layer source."""

    def __init__(
        self,
        wms_server_url: str,
        layernames: list[str],
        layerstyles: list[str] | None = None,
        bands: list[int] | None = None,
        username: str | None = None,
        password: str | None = None,
        wms_version: str = "1.3.0",
        wms_ignore_capabilities_url: bool = False,
        random_sleep: int = 0,
        wms_service: owslib.wms.wms111.WebMapService_1_1_1
        | owslib.wms.wms130.WebMapService_1_3_0
        | None = None,
    ):
        """Constructor of WMSLayerSource.

        Args:
            wms_server_url (str): _description_
            layernames (list[str]): _description_
            layerstyles (Optional[list[str]], optional): _description_.
                Defaults to None.
            bands (Optional[list[int]], optional): _description_. Defaults to None.
            username (str, optional): username to logon with. Defaults to None.
            password (str, optional): password to logon with. Defaults to None.
            wms_version (str, optional): _description_. Defaults to "1.3.0".
            wms_ignore_capabilities_url (bool, optional): _description_.
                Defaults to False.
            random_sleep (int, optional): _description_. Defaults to 0.
            wms_service: The WMS service. Defaults to None.
        """
        self.wms_server_url = wms_server_url
        self.wms_version = wms_version
        self.wms_ignore_capabilities_url = wms_ignore_capabilities_url
        self.layernames = layernames
        self.layerstyles = layerstyles
        self.bands = bands
        self.username = username
        self.password = password
        self.random_sleep = random_sleep
        self.wms_service = wms_service


class FileLayerSource:
    """Properties of a file layer source."""

    def __init__(
        self,
        path: str | Path,
        layernames: list[str],
        bands: list[int] | None = None,
    ):
        """Contructor for FileLayerSource.

        Args:
            path (Union[str, Path]): Path to the layer.
            layernames (list[str]): list of layer names.
            bands (Optional[list[int]], optional): list of bands. Defaults to None.
        """
        self.path = Path(path)
        self.layernames = layernames
        self.bands = bands


def get_images_for_grid(
    output_image_dir: Path,
    crs: pyproj.CRS,
    image_gen_bbox: tuple[float, float, float, float] | None = None,
    image_gen_roi_filepath: Path | None = None,
    grid_xmin: float = 0.0,
    grid_ymin: float = 0.0,
    image_crs_pixel_x_size: float = 0.25,
    image_crs_pixel_y_size: float = 0.25,
    image_pixel_width: int = 1024,
    image_pixel_height: int = 1024,
    image_format: str = FORMAT_GEOTIFF,
    pixels_overlap: int = 0,
) -> gpd.GeoDataFrame:
    """Get a list of all images in the grid specified.

    Args:
        output_image_dir (Path): Directory to save the images to.
        crs (Union[str, pyproj.CRS]): The crs of the source and destination images.
        image_gen_bbox (tuple[float, float, float, float], optional): bbox of the roi to
            request/save images for. Defaults to None.
        image_gen_roi_filepath (Optional[Path], optional): File with the roi
            where images should be requested/saved for. Defaults to None.
        grid_xmin (float, optional): xmin for the grid to be used.
            Defaults to 0.0.
        grid_ymin (float, optional): ymin for the grid to be used.
            Defaults to 0.0.
        image_crs_pixel_x_size (float, optional): Pixel size of the image tiles to
            create in the `crs` specified. Defaults to 0.25.
        image_crs_pixel_y_size (float, optional): Pixel size of the image tiles to
            create in the `crs` specified. Defaults to 0.25.
        image_pixel_width (int, optional): Width of the tiles to create in number of
            pixels. Defaults to 1024.
        image_pixel_height (int, optional): Height of the tiles to create in number of
            pixels. Defaults to 1024.
        image_format (str, optional): The image format to save to. Defaults to
            FORMAT_GEOTIFF.
        pixels_overlap (int, optional): The number of pixels the tiles should be
            enlarged in all directions to create overlapping tiles.
            Defaults to 0.
    """
    # Tile size in units of crs
    crs_width = math.fabs(image_pixel_width * image_crs_pixel_x_size)
    crs_height = math.fabs(image_pixel_height * image_crs_pixel_y_size)

    # Read the region of interest file if provided
    grid_bbox = image_gen_bbox
    if grid_bbox is None and image_gen_roi_filepath is not None:
        # Read roi from file
        grid_bbox = gfo.get_layerinfo(image_gen_roi_filepath).total_bounds

    # If there is still no grid_bbox, stop.
    if grid_bbox is None:
        raise ValueError(
            "At least one of image_gen_bbox or image_gen_roi_filepath should be "
            "specified."
        )

    # Make grid_bounds compatible with the grid
    grid_bbox = _align_bbox_to_grid(
        bbox=grid_bbox,
        grid_xmin=grid_xmin,
        grid_ymin=grid_ymin,
        pixel_size_x=crs_width,
        pixel_size_y=crs_height,
        log_level=logging.WARNING,
    )

    # Create a grid of the images that need to be downloaded
    tiles_to_download_gdf = gpd.GeoDataFrame(
        geometry=pygeoops.create_grid3(grid_bbox, width=crs_width, height=crs_height),
        crs=crs,
    )

    # Loop through all tiles to apply pixels_overlap and add the output path
    for tile in tiles_to_download_gdf.geometry.bounds.itertuples():
        _, tile_xmin, tile_ymin, tile_xmax, tile_ymax = tile
        tile_pixel_width = image_pixel_width
        tile_pixel_height = image_pixel_height

        # If overlapping images are wanted... increase image bbox
        if pixels_overlap:
            tile_xmin -= pixels_overlap * image_crs_pixel_x_size
            tile_xmax += pixels_overlap * image_crs_pixel_x_size
            tile_ymin -= pixels_overlap * image_crs_pixel_y_size
            tile_ymax += pixels_overlap * image_crs_pixel_y_size
            tile_pixel_width += 2 * pixels_overlap
            tile_pixel_height += 2 * pixels_overlap

        tiles_to_download_gdf.at[tile.Index, "geometry"] = box(
            tile_xmin, tile_ymin, tile_xmax, tile_ymax
        )

        # Create output filepath
        # Put images in a subdirectory based on the x-coordinate of the tile to avoid
        # one directory with too many files
        if crs.is_projected:
            output_dir = output_image_dir / f"{tile_xmin:06.0f}"
        else:
            output_dir = output_image_dir / f"{tile_xmin:09.4f}"
        output_filename = create_filename(
            crs=crs,
            bbox=(tile_xmin, tile_ymin, tile_xmax, tile_ymax),
            size=(tile_pixel_width, tile_pixel_height),
            image_format=image_format,
            layername=None,
        )
        output_filepath = output_dir / output_filename

        # Add the extra info to the gdf
        tiles_to_download_gdf.at[tile.Index, "path"] = output_filepath
        tiles_to_download_gdf.at[tile.Index, "pixel_width"] = tile_pixel_width
        tiles_to_download_gdf.at[tile.Index, "pixel_height"] = tile_pixel_height

    # Write the tiles to download to file for reference
    tiles_path = output_image_dir / "tiles_to_download.gpkg"
    if image_gen_roi_filepath is not None:
        # If an roi is specified, filter the tiles to download with it
        tiles_all_tmp_path = output_image_dir / "tiles_all_tmp.gpkg"
        gfo.to_file(tiles_to_download_gdf, tiles_all_tmp_path)
        gfo.export_by_location(
            input_to_select_from_path=tiles_all_tmp_path,
            input_to_compare_with_path=image_gen_roi_filepath,
            output_path=tiles_path,
            spatial_relations_query="intersects is True",
            force=True,
        )
        gfo.remove(tiles_all_tmp_path)

        # Read the filtered tiles to download to return them
        tiles_to_download_gdf = gfo.read_file(tiles_path)
        tiles_to_download_gdf["path"] = tiles_to_download_gdf["path"].apply(Path)

    else:
        # No roi specified, so just write the tiles to download
        gfo.to_file(tiles_to_download_gdf, tiles_path)

    return tiles_to_download_gdf


def load_images_to_cache(
    layersources: list[FileLayerSource | WMSLayerSource],
    output_image_dir: Path,
    crs: str | pyproj.CRS,
    image_gen_bbox: tuple[float, float, float, float] | None = None,
    image_gen_roi_filepath: Path | None = None,
    grid_xmin: float = 0.0,
    grid_ymin: float = 0.0,
    image_crs_pixel_x_size: float = 0.25,
    image_crs_pixel_y_size: float = 0.25,
    image_pixel_width: int = 1024,
    image_pixel_height: int = 1024,
    image_pixels_ignore_border: int = 0,
    nb_concurrent_calls: int = 1,
    cron_schedule: str | None = None,
    image_format: str = FORMAT_GEOTIFF,
    image_format_save: str | None = None,
    tiff_compress: str = "lzw",
    transparent: bool = False,
    pixels_overlap: int = 0,
    nb_images_to_skip: int = 0,
    max_nb_images: int = -1,
    ssl_verify: bool | str = True,
    force: bool = False,
):
    """Loads all images in a grid from a layer source to a cache directory.

    Args:
        layersources (list[dict]): Layer sources to get images from. Multiple
            sources can be specified to create a combined image, eg. use band
            1 of a WMS service with band 2 and 3 of another one.
        output_image_dir (Path): Directory to save the images to.
        crs (Union[str, pyproj.CRS]): The crs of the source and destination images.
        image_gen_bbox (tuple[float, float, float, float], optional): bbox of the roi to
            request/save images for. Defaults to None.
        image_gen_roi_filepath (Optional[Path], optional): File with the roi
            where images should be requested/saved for. Defaults to None.
        grid_xmin (float, optional): xmin for the grid to be used.
            Defaults to 0.0.
        grid_ymin (float, optional): ymin for the grid to be used.
            Defaults to 0.0.
        image_crs_pixel_x_size (float, optional): Pixel size of the image tiles to
            create in the `crs` specified. Defaults to 0.25.
        image_crs_pixel_y_size (float, optional): Pixel size of the image tiles to
            create in the `crs` specified. Defaults to 0.25.
        image_pixel_width (int, optional): Width of the tiles to create in number of
            pixels. Defaults to 1024.
        image_pixel_height (int, optional): Height of the tiles to create in number of
            pixels. Defaults to 1024.
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
        pixels_overlap (int, optional): The number of pixels the tiles should be
            enlarged in all directions to create overlapping tiles.
            Defaults to 0.
        nb_images_to_skip (int, optional): [description]. Defaults to 0.
        max_nb_images (int, optional): [description]. Defaults to -1.
        ssl_verify (bool or str, optional): True to use the default
            certificate bundle as installed on your system. False disables
            certificate validation (NOT recommended!). If a path to a
            certificate bundle file (.pem) is passed, this will be used.
            In corporate networks using a proxy server this is often needed
            to evade CERTIFICATE_VERIFY_FAILED errors. Defaults to True.
        force (bool, optional): [description]. Defaults to False.
    """
    # Init
    if image_format_save is None:
        image_format_save = image_format
    if not isinstance(crs, pyproj.CRS):
        crs = pyproj.CRS(crs)

    if cron_schedule is not None and cron_schedule != "":
        logger.info(
            "A cron_schedule was specified, so the download will only proceed in the "
            f"specified time range: {cron_schedule}"
        )

    # Create the output dir if it doesn't exist yet
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # Write the tiles to download to file for reference
    tiles_to_download_gdf = get_images_for_grid(
        output_image_dir=output_image_dir,
        crs=crs,
        image_gen_bbox=image_gen_bbox,
        image_gen_roi_filepath=image_gen_roi_filepath,
        grid_xmin=grid_xmin,
        grid_ymin=grid_ymin,
        image_crs_pixel_x_size=image_crs_pixel_x_size,
        image_crs_pixel_y_size=image_crs_pixel_y_size,
        image_pixel_width=image_pixel_width,
        image_pixel_height=image_pixel_height,
        image_format=image_format,
        pixels_overlap=pixels_overlap,
    )

    with futures.ProcessPoolExecutor(nb_concurrent_calls) as pool:
        # Loop through all columns and get the images...
        has_switched_axes = _has_switched_axes(crs)
        nb_total = len(tiles_to_download_gdf)
        nb_processed = 0
        nb_downloaded = 0
        download_queue = {}
        logger.info(f"Start loading {nb_total} images")
        progress = None
        for tile in tiles_to_download_gdf.geometry.bounds.itertuples():
            _, tile_xmin, tile_ymin, tile_xmax, tile_ymax = tile
            tile_pixel_width = image_pixel_width
            tile_pixel_height = image_pixel_height

            # Init progress
            if progress is None:
                message = (
                    f"load_images to {output_image_dir.parent.name}/"
                    f"{output_image_dir.name}"
                )
                progress = progress_util.ProgressLogger(
                    message=message,
                    nb_steps_total=nb_total,
                    nb_steps_done=0,
                )

            nb_processed += 1
            output_filepath = tiles_to_download_gdf.at[tile.Index, "path"]
            output_dir = output_filepath.parent
            output_filename = output_filepath.name

            # Do some checks to know if the image needs to be downloaded
            if nb_images_to_skip > 0 and (nb_processed % nb_images_to_skip) != 0:
                # If we need to skip images, do so...
                progress.step()
                continue
            elif (
                not force
                and output_filepath.exists()
                and output_filepath.stat().st_size > 0
            ):
                # Image exists already
                progress.step()
                logger.debug("    -> image exists already, so skip")
                continue

            # If a cron_schedule is specified, check if we should be running
            if cron_schedule is not None and cron_schedule not in ["", "* * * * *"]:
                # Sleep till the schedule becomes active
                first_cron_check = True
                while not pycron.is_now(cron_schedule):
                    # The first time, log message that we are going to sleep...
                    if first_cron_check is True:
                        logger.info(f"Time schedule specified: sleep: {cron_schedule}")
                        first_cron_check = False
                    time.sleep(60)

            # Submit the image to be downloaded
            output_dir.mkdir(parents=True, exist_ok=True)
            if pixels_overlap:
                tile_pixel_width += 2 * pixels_overlap
                tile_pixel_height += 2 * pixels_overlap

            future = pool.submit(
                load_image_to_file,  # Function
                layersources=layersources,
                output_dir=output_dir,
                crs=crs,
                bbox=(tile_xmin, tile_ymin, tile_xmax, tile_ymax),
                size=(tile_pixel_width, tile_pixel_height),
                ssl_verify=ssl_verify,
                image_format=image_format,
                image_format_save=image_format_save,
                output_filename=output_filename,
                transparent=transparent,
                tiff_compress=tiff_compress,
                image_pixels_ignore_border=image_pixels_ignore_border,
                has_switched_axes=has_switched_axes,
                force=force,
                on_outside_layer_bounds="return",
            )
            download_queue[future] = output_filename

            # Process finished downloads till queue is of acceptable size
            while True:
                # Process downloads that are ready
                futures_done = []
                for future in download_queue:
                    if not future.done():
                        continue

                    # Fetch result: will throw exception if something went wrong
                    _ = future.result()
                    nb_downloaded += 1
                    futures_done.append(future)

                    # Log the progress and download speed
                    progress.step()

                # Remove futures that are done
                for future in futures_done:
                    del download_queue[future]
                futures_done = []

                # If all image tiles have been processed or if the max number of
                # images to download is reached...
                if (
                    nb_processed >= nb_total
                    or max_nb_images > -1
                    and nb_downloaded >= max_nb_images
                ):
                    if len(download_queue) == 0:
                        return
                else:
                    # Not all tiles have been processed yet, and the queue isn't too
                    # full, so process some more
                    if len(download_queue) < nb_concurrent_calls * 2:
                        break

                # Sleep a bit before checking again if there are downloads ready
                time.sleep(0.1)


def _align_bbox_to_grid(
    bbox: tuple[float, float, float, float],
    grid_xmin: float,
    grid_ymin: float,
    pixel_size_x: float,
    pixel_size_y: float,
    log_level: int = logging.INFO,
) -> tuple[float, float, float, float]:
    """Align a bounding box to the grid specified.

    Args:
        bbox (tuple[float, float, float, float]): the bounding box
        grid_xmin (float): xmin of the grid to align to.
        grid_ymin (float): ymin of the grid to align to.
        pixel_size_x (float): pixel size for x.
        pixel_size_y (float): pixel size for y.
        log_level (int, optional): the log level to use. Defaults to logging.INFO.

    Returns:
        tuple[float, float, float, float]: the aligned bbox.
    """
    # Make bounds compatible with the grid
    bbox_tmp = list(bbox)
    if (bbox_tmp[0] - grid_xmin) % pixel_size_x != 0:
        xmin_new = bbox_tmp[0] - ((bbox_tmp[0] - grid_xmin) % pixel_size_x)
        msg = f"xmin {bbox_tmp[0]} incompatible with grid, {xmin_new} used"
        logger.log(level=log_level, msg=msg)
        bbox_tmp[0] = xmin_new
    if (bbox_tmp[1] - grid_ymin) % pixel_size_y != 0:
        ymin_new = bbox_tmp[1] - ((bbox_tmp[1] - grid_ymin) % pixel_size_y)
        msg = f"ymin {bbox_tmp[1]} incompatible with grid, {ymin_new} used"
        logger.log(level=log_level, msg=msg)
        bbox_tmp[1] = ymin_new
    if (bbox_tmp[2] - grid_xmin) % pixel_size_x != 0:
        xmax_new = (
            bbox_tmp[2] + pixel_size_x - ((bbox_tmp[2] - grid_xmin) % pixel_size_x)
        )
        msg = f"xmax {bbox_tmp[2]} incompatible with grid, {xmax_new} used"
        logger.log(level=log_level, msg=msg)
        bbox_tmp[2] = xmax_new
    if (bbox_tmp[3] - grid_ymin) % pixel_size_y != 0:
        ymax_new = (
            bbox_tmp[3] + pixel_size_y - ((bbox_tmp[3] - grid_ymin) % pixel_size_y)
        )
        msg = f"ymax {bbox_tmp[3]} incompatible with grid, {ymax_new} used"
        logger.log(level=log_level, msg=msg)
        bbox_tmp[3] = ymax_new

    return (bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[3])


def _interprete_ssl_verify(ssl_verify: bool | str | None):
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
            logger.warning("SSL VERIFICATION IS TURNED OFF!!!")

    return auth


def load_image_to_file(
    layersources: WMSLayerSource | FileLayerSource | list,
    output_dir: Path,
    crs: str | pyproj.CRS,
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    ssl_verify: bool | str = True,
    image_format: str = FORMAT_GEOTIFF,
    image_format_save: str | None = None,
    output_filename: str | None = None,
    transparent: bool = False,
    tiff_compress: str = "lzw",
    image_pixels_ignore_border: int = 0,
    force: bool = False,
    layername_in_filename: bool = False,
    has_switched_axes: bool | None = None,
    on_outside_layer_bounds: str | None = "raise",
) -> Path | None:
    """Loads an image from a layer source and saves it to a file.

    Args:
        layersources (WMSLayerSource, FileLayerSource, List): Layer source(s) to get
            images from. Multiple sources can be specified to create a combined image,
            eg. use band 1 of a layersource with band 2 and 3 of another one.
        output_dir (Path): Directory to save the images to.
        crs (pyproj.CRS): The crs of the source and destination images.
        bbox (tuple[float, float, float, float]): Bbox of the image to get.
        size (tuple[int, int]): The image width and height.
        ssl_verify (bool or str, optional): True to use the default
            certificate bundle as installed on your system. False disables
            certificate validation (NOT recommended!). If a path to a
            certificate bundle file (.pem) is passed, this will be used.
            In corporate networks using a proxy server this is often needed
            to evade CERTIFICATE_VERIFY_FAILED errors. Defaults to True.
        image_format (str, optional): [description]. Defaults to FORMAT_GEOTIFF.
        image_format_save (str, optional): [description]. Defaults to None.
        output_filename (str, optional): [description]. Defaults to None.
        transparent (bool, optional): [description]. Defaults to False.
        tiff_compress (str, optional): [description]. Defaults to 'lzw'.
        image_pixels_ignore_border (int, optional): [description]. Defaults to 0.
        force (bool, optional): [description]. Defaults to False.
        layername_in_filename (bool, optional): [description]. Defaults to False.
        has_switched_axes (bool, optional): True if x and y axes should be switched to
            in the WMS GetMap request. If None, an effort is made to determine it
            automatically based on the crs. Defaults to None.
        on_outside_layer_bounds (str, optional): What to do if the bbox asked is outside
            the layer bounds. Defaults to "raise". Options:
            - "raise": raise an error.
            - "return": don't save a file and return None.

    Returns:
        Optional[Path]: The path the file is created at if created succesfully. None if
            the file is not created.
    """
    # Init
    if on_outside_layer_bounds not in ["raise", "return"]:
        raise ValueError(f"Invalid value for {on_outside_layer_bounds=}")

    # If no separate save format is specified, use the standard image_format
    if image_format_save is None:
        image_format_save = image_format

    # Convert input parameters if relevant
    if isinstance(crs, str):
        crs = pyproj.CRS(crs)
    if not isinstance(layersources, list):
        layersources = [layersources]

    # If there isn't a filename supplied, create one...
    if output_filename is None:
        layername = None
        if layername_in_filename:
            for layersource in layersources:
                if layername is None:
                    layername = "_".join(layersource.layernames)
                else:
                    layername += f"_{'_'.join(layersource.layernames)}"

        output_filename = create_filename(
            crs=crs,
            bbox=bbox,
            size=size,
            image_format=image_format_save,
            layername=layername,
        )

    # Create full output filepath
    output_filepath = output_dir / output_filename

    # If force is false and file exists already, stop...
    if not force and output_filepath.exists():
        if output_filepath.stat().st_size > 0:
            logger.debug(f"File already exists, skip: {output_filepath}")
            return output_filepath
        else:
            try:
                output_filepath.unlink()
            except Exception as ex:  # pragma: no cover
                logger.warning(f"Error removing file {output_filepath}: {ex}")

    logger.debug(f"Get image to {output_filepath}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = load_image(
            layersources=layersources,
            crs=crs,
            bbox=bbox,
            size=size,
            ssl_verify=ssl_verify,
            image_format=image_format,
            transparent=transparent,
            image_pixels_ignore_border=image_pixels_ignore_border,
            has_switched_axes=has_switched_axes,
        )
    except RuntimeError as ex:  # pragma: no cover
        if str(ex).startswith("Bbox outside layer bounds"):
            if on_outside_layer_bounds == "return":
                return None
        raise ex

    # Write (temporary) output file
    image_data_output, image_profile_output = image

    # Set correct output driver in profile
    image_profile_output["driver"] = _get_driver_for_image_format(image_format_save)

    # Prepare output bands and set them correctly in profile
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
    image_profile_output["count"] = image_data_output.shape[0]
    image_profile_output = _get_cleaned_write_profile(image_profile_output)

    # Because the (temporary) output file doesn't contain coordinates (yet),
    # suppress NotGeoreferencedWarning while writing
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rio_errors.NotGeoreferencedWarning)

        try:
            with rio.open(
                str(output_filepath), "w", **image_profile_output
            ) as image_file:
                image_file.write(image_data_output)
        except CPLE_AppDefinedError as ex:  # pragma: no cover
            errmsg = ex.errmsg.strip()
            if errmsg.startswith("Deleting ") and errmsg.endswith(
                (" failed: No such file or directory", " failed: Permission denied")
            ):
                # Occasionally this error occurs, not sure why: ignore it.
                logger.debug(f"Ignore error: {ex}")
            else:
                raise ex

    # If an aux.xml file was written, remove it again...
    output_aux_path = output_filepath.parent / f"{output_filepath.name}.aux.xml"
    try:
        output_aux_path.unlink(missing_ok=True)
    except Exception as ex:  # pragma: no cover
        # Occasionally the .aux.xml file is locked, not sure why: ignore it.
        logger.debug(f"Ignore error: {ex}")

    # Make the output image compliant with image_format_save

    # If geotiff is asked, check if the the coordinates are embedded...
    if image_format_save == FORMAT_GEOTIFF:
        # Read output image to check if coördinates are there
        with rio.open(str(output_filepath)) as image_file:
            image_profile_orig = image_file.profile
            image_transform_affine = image_file.transform

            if image_pixels_ignore_border == 0:
                image_data_output = image_file.read()
            else:
                image_data_output = image_file.read(
                    window=rio_windows.Window(
                        col_off=image_pixels_ignore_border,
                        row_off=image_pixels_ignore_border,
                        width=size[0],
                        height=size[1],
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
                f"    - x-part of pixel width, W-E: {crs_pixel_x_size}\n"
                "    - y-part of pixel width, W-E (0 if image is exactly N up): 0\n"
                f"    - top-left x: {bbox[0]}\n"
                "    - x-part of pixel height, N-S (0 if image is exactly N up): \n"
                f"    - y-part of pixel height, N-S: {crs_pixel_y_size}\n"
                f"    - top-left y: {bbox[3]}"
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
            with rio.open(str(output_filepath)) as image_file:
                image_profile_orig = image_file.profile
                image_transform_affine = image_file.transform

                # If border needs to be ignored, only read data we are interested in
                if image_pixels_ignore_border == 0:
                    image_data_output = image_file.read()
                else:
                    image_data_output = image_file.read(
                        window=rio_windows.Window(
                            col_off=image_pixels_ignore_border,
                            row_off=image_pixels_ignore_border,
                            width=size[0],
                            height=size[1],
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


def load_image(
    layersources: WMSLayerSource | FileLayerSource | list,
    crs: str | pyproj.CRS,
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    ssl_verify: bool | str = True,
    image_format: str = FORMAT_GEOTIFF,
    transparent: bool = False,
    image_pixels_ignore_border: int = 0,
    has_switched_axes: bool | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Loads an image from a layer source and saves it to a file.

    Args:
        layersources (WMSLayerSource, FileLayerSource, List): Layer source(s) to get
            images from. Multiple sources can be specified to create a combined image,
            eg. use band 1 of a layersource with band 2 and 3 of another one.
        output_dir (Path): Directory to save the images to.
        crs (pyproj.CRS): The crs of the source and destination images.
        bbox (tuple[float, float, float, float]): Bbox of the image to get.
        size (tuple[int, int]): The image width and height.
        ssl_verify (bool or str, optional): True to use the default
            certificate bundle as installed on your system. False disables
            certificate validation (NOT recommended!). If a path to a
            certificate bundle file (.pem) is passed, this will be used.
            In corporate networks using a proxy server this is often needed
            to evade CERTIFICATE_VERIFY_FAILED errors. Defaults to True.
        image_format (str, optional): [description]. Defaults to FORMAT_GEOTIFF.
        image_format_save (str, optional): [description]. Defaults to None.
        output_filename (str, optional): [description]. Defaults to None.
        transparent (bool, optional): [description]. Defaults to False.
        tiff_compress (str, optional): [description]. Defaults to 'lzw'.
        image_pixels_ignore_border (int, optional): [description]. Defaults to 0.
        force (bool, optional): [description]. Defaults to False.
        layername_in_filename (bool, optional): [description]. Defaults to False.
        has_switched_axes (bool, optional): True if x and y axes should be switched to
            in the WMS GetMap request. If None, an effort is made to determine it
            automatically based on the crs. Defaults to None.

    Raises:
        RuntimeError: If the image can't be retrieved.

    Returns:
        tuple(ndarray, dict): a tuple with an array with the image data and a dict with
            the profile information of the file (transform,...).
    """
    # Init
    if isinstance(crs, str):
        crs = pyproj.CRS.from_user_input(crs)
    if not isinstance(layersources, list):
        layersources = [layersources]

    # Get image(s), read the band to keep and save
    # Some hacks for special cases...
    bbox_local = bbox
    size_to_use = size
    x_pixsize = (bbox[2] - bbox[0]) / size[0]
    y_pixsize = (bbox[3] - bbox[1]) / size[1]

    # Dirty hack to ask a bigger picture, and then remove the border again!
    if image_pixels_ignore_border > 0:
        bbox_local = (
            bbox[0] - x_pixsize * image_pixels_ignore_border,
            bbox[1] - y_pixsize * image_pixels_ignore_border,
            bbox[2] + x_pixsize * image_pixels_ignore_border,
            bbox[3] + y_pixsize * image_pixels_ignore_border,
        )
        size_to_use = (
            size[0] + 2 * image_pixels_ignore_border,
            size[1] + 2 * image_pixels_ignore_border,
        )

    # For coordinate systems with switched axis (y, x or lon, lat), switch x and y
    if has_switched_axes is None:
        has_switched_axes = _has_switched_axes(crs)
    if has_switched_axes:
        bbox_local = (bbox_local[1], bbox_local[0], bbox_local[3], bbox_local[2])

    image_data_output = None
    image_profile_output: dict[str, Any] | None = None
    response = None
    for layersource in layersources:
        window = None
        memfile = None
        image_file = None
        rio_read_kwargs = {}
        try:
            # If it is a WMS layer source
            if isinstance(layersource, WMSLayerSource):
                # Initialize WMS if this hasn't been done yet
                if layersource.wms_service is None:
                    auth = _interprete_ssl_verify(ssl_verify=ssl_verify)
                    wms_service = owslib.wms.WebMapService(
                        url=layersource.wms_server_url,
                        version=layersource.wms_version,
                        username=layersource.username,
                        password=layersource.password,
                        auth=auth,
                    )
                    if layersource.wms_ignore_capabilities_url:
                        # If the wms url in capabilities should be ignored,
                        # overwrite with original url
                        nb = len(wms_service.getOperationByName("GetMap").methods)
                        for method_id in range(nb):
                            wms_service.getOperationByName("GetMap").methods[method_id][
                                "url"
                            ] = layersource.wms_server_url
                    layersource.wms_service = wms_service

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
                            bbox=bbox_local,
                            size=size_to_use,
                            format=image_format,
                            transparent=transparent,
                        )
                        logger.debug(f"Finished doing request {response.geturl()}")

                        # If a random sleep was specified... apply it
                        if layersource.random_sleep > 0:
                            time.sleep(random.uniform(0, layersource.random_sleep))

                        # Image was retrieved... so stop loop
                        image_retrieved = True
                    except Exception as ex:  # pragma: no cover
                        if isinstance(ex, owslib.util.ServiceException):
                            if "Error rendering coverage on the fast path" in str(ex):
                                message = f"Bbox outside layer bounds? {bbox_local}"
                                raise RuntimeError(message) from ex
                            elif "java.lang.OutOfMemoryError: Java heap" in str(ex):
                                logger.debug(
                                    f"Request for bbox {bbox_local} gave an "
                                    f"exception, try again in {time_sleep} s: {ex}"
                                )
                            else:
                                message = f"WMS error for bbox {bbox_local}: {ex}"
                                raise RuntimeError(message) from ex

                        # If the exception isn't handled yet, retry 10 times...
                        if nb_retries < 10:
                            time.sleep(time_sleep)

                            # Increase sleep time every
                            # time.
                            time_sleep += 5
                            nb_retries += 1
                            continue
                        else:
                            message = (
                                "Retried 10 times and didn't work, with "
                                f"layers: {layersource.layernames}, "
                                f"styles: {layersource.layerstyles}, "
                                f"for bbox: {bbox_local}"
                            )
                            raise RuntimeError(message) from ex

                # If the response is None, error
                if response is None:
                    raise RuntimeError("No valid response retrieved...")

                # Open the response as a file
                memfile = rio.MemoryFile(response.read())
                # Because the image returned by WMS doesn't contain georeferencing
                # info, suppress NotGeoreferencedWarning
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=rio_errors.NotGeoreferencedWarning
                    )
                    image_file = memfile.open()

            elif isinstance(layersource, FileLayerSource):
                image_file = rio.open(str(layersource.path))
                if layersource.bands is not None:
                    nb_bands = len(layersource.bands)
                else:
                    nb_bands = image_file.profile["count"]
                window = rio_windows.from_bounds(
                    left=bbox_local[0],
                    bottom=bbox_local[1],
                    right=bbox_local[2],
                    top=bbox_local[3],
                    transform=image_file.transform,
                )
                rio_read_kwargs = {
                    "window": window,
                    "out_shape": (nb_bands, size[1], size[0]),
                    "resampling": rasterio.enums.Resampling.nearest,
                    "boundless": True,
                }
            else:
                raise ValueError(f"Unsupported layer source: {layersource}")

            # Read the data we need from the opened file
            if layersource.bands is None:
                # If no specific bands specified, read them all...
                if image_data_output is None:
                    image_data_output = image_file.read(**rio_read_kwargs)
                else:
                    image_data_output = np.append(
                        image_data_output,
                        image_file.read(**rio_read_kwargs),
                        axis=0,
                    )
            elif len(layersource.bands) == 1 and layersource.bands[0] == -1:
                # If 1 band, -1 specified: dirty hack to use greyscale
                # version of rgb image
                image_data_tmp = image_file.read(**rio_read_kwargs)
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
                    # Read the band needed + reshape. Remark: rasterio uses
                    # 1-based indexing instead of 0-based
                    rio_read_kwargs["indexes"] = band + 1
                    image_data_curr = image_file.read(**rio_read_kwargs)
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

            # Set output profile
            # (# of bands will be corrected later if needed)
            if image_profile_output is None:
                image_profile_output = dict(image_file.profile)

                # Set transform. Pararameters to create Affine:
                #   -> (x_pixsize, 0.0, xmin, 0.0, -y_pixsize, ymax)
                transform = rio.Affine(
                    x_pixsize, 0.0, bbox_local[0], 0.0, -y_pixsize, bbox_local[3]
                )
                image_profile_output["transform"] = transform

                assert isinstance(image_data_output, np.ndarray)
                image_profile_output["height"] = image_data_output.shape[1]
                image_profile_output["width"] = image_data_output.shape[2]

        finally:
            if image_file is not None:
                image_file.close()
                image_file = None
            if memfile is not None:
                memfile.close()
                memfile = None

    if image_data_output is None or image_profile_output is None:  # pragma: no cover
        raise RuntimeError("No image data retrieved...")

    return (image_data_output, image_profile_output)


def create_filename(
    crs: pyproj.CRS, bbox, size, image_format: str, layername: str | None = None
) -> str:
    """Create filename.

    Args:
        crs (pyproj.CRS): crs
        bbox (_type_): bbox
        size (_type_): size
        image_format (str): format
        layername (Optional[str], optional): layername. Defaults to None.

    Returns:
        str: filename
    """
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


def _get_driver_for_image_format(image_format: str) -> str:
    # Choose image extension based on format
    if image_format == FORMAT_GEOTIFF:
        return FORMAT_GEOTIFF_DRIVER
    elif image_format == FORMAT_TIFF:
        return FORMAT_TIFF_DRIVER
    elif image_format == FORMAT_JPEG:
        return FORMAT_JPEG_DRIVER
    elif image_format == FORMAT_PNG:
        return FORMAT_PNG_DRIVER
    else:
        raise Exception(
            f"get_ext_for_image_format for image format {image_format} not implemented"
        )


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
    profile: dict | rio_profiles.Profile,
) -> dict | rio_profiles.Profile:
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
        # Don't copy profile keys to cleaned version that are not supported for PNG
        profile_cleaned = {}
        for profile_key in profile:
            if profile_key not in [
                "blockxsize",
                "blockysize",
                "compress",
                "interleave",
                "tiled",
            ]:
                profile_cleaned[profile_key] = profile[profile_key]
    else:
        profile_cleaned = profile.copy()

    return profile_cleaned

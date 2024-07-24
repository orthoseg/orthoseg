"""Module with functions for post-processing prediction masks towards polygons."""

import logging
import math
import shutil
from pathlib import Path
from typing import Optional

import geofileops as gfo
import geopandas as gpd
import numpy as np
import pygeoops
import rasterio as rio
import rasterio.features as rio_features
import rasterio.transform as rio_transform
import shapely.geometry as sh_geom
import skimage.filters.rank
import tensorflow as tf
from skimage.morphology import rectangle

from orthoseg.helpers import vectorfile_helper
from orthoseg.util import vector_util

# Avoid having many info warnings about self intersections from shapely
logging.getLogger("shapely.geos").setLevel(logging.WARNING)

# Get a logger...
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Postprocess to use on all vector outputs
# -------------------------------------------------------------


def postprocess_predictions(
    input_path: Path,
    output_path: Path,
    dissolve: bool,
    dissolve_tiles_path: Optional[Path] = None,
    reclassify_to_neighbour_query: Optional[str] = None,
    simplify_algorithm: Optional[str] = None,
    simplify_tolerance: float = 1,
    simplify_lookahead: int = 8,
    keep_original_file: bool = True,
    keep_intermediary_files: bool = True,
    nb_parallel: int = -1,
    force: bool = False,
) -> list[Path]:
    """Postprocesses the input prediction as specified.

    Args:
        input_path: path to the 'raw' prediction vector file.
        output_path: the base path where the output file(s) will be written to.
        keep_original_file: If True, the output file of the prediction step
            will be retained ofter postprocessing, otherwise it is removed.
        keep_intermediary_files: If True, intermediary postprocessing files are removed.
        dissolve (bool): True if a dissolve needs to be applied
        dissolve_tiles_path (PathLike, optional): Path to a geofile containing
            the tiles to be used for the dissolve. Defaults to None.
        reclassify_to_neighbour_query (str, optional): Defaults to None.
        simplify_algorithm (str, optional): Algorithm to use for simplification. If
            None, no simplification is applied. Defaults to None.
        simplify_tolerance (float): Tolerance to use for the simplification.
            Defaults to 1.
        simplify_lookahead (int): Lookahead to use for simplification. Default to 8.
        nb_parallel (int, optional): number of cpu's to use for postprocessing.
            Use all cpu's if it is -1. Defaults to -1.
        force: False to skip results that already exist, true to
               ignore existing results and overwrite them
    """
    # Init
    if not input_path.exists():
        raise Exception(f"input_path does not exist: {input_path}")

    # The return value is the list of paths created
    output_paths = []

    # Because the geo operations will be applied sequentially if applicable,
    # both the input path and output path will build on the result of the
    # previous operation.
    # Set the initial values to the ones passed in as parameters.
    curr_input_path = input_path
    curr_output_path = output_path

    # Dissolve the predictions if needed
    if dissolve:
        curr_output_path = (
            output_path.parent / f"{output_path.stem}_dissolve{output_path.suffix}"
        )

        # If the dissolved file doesn't exist yet, go for it...
        if not curr_output_path.exists():
            # If column classname present, group on it...
            layerinfo = gfo.get_layerinfo(input_path)
            if "classname" in layerinfo.columns:
                groupby_columns = ["classname"]
            else:
                groupby_columns = []

            # Now we can dissolve
            gfo.dissolve(
                input_path=input_path,
                tiles_path=dissolve_tiles_path,
                output_path=curr_output_path,
                groupby_columns=groupby_columns,
                explodecollections=True,
                nb_parallel=nb_parallel,
                force=force,
            )

            # Add/recalculate columns with area and nbcoords
            gfo.add_column(
                path=curr_output_path,
                name="area",
                type=gfo.DataType.REAL,
                expression="ST_Area(geom)",
                force_update=True,
            )
            gfo.add_column(
                path=curr_output_path,
                name="nbcoords",
                type=gfo.DataType.INTEGER,
                expression="ST_NPoints(geom)",
                force_update=True,
            )

        # The curr_output_path becomes the new current input path
        curr_input_path = curr_output_path
        output_paths.append(curr_output_path)

    if reclassify_to_neighbour_query is not None:
        curr_output_path = (
            output_path.parent / f"{output_path.stem}_reclass{output_path.suffix}"
        )
        vectorfile_helper.reclassify_neighbours(
            input_path=curr_input_path,
            reclassify_column="classname",
            query=reclassify_to_neighbour_query,
            output_path=curr_output_path,
        )
        curr_input_path = curr_output_path
        output_paths.append(curr_output_path)

    # If a simplify algorithm is specified, simplify!
    if simplify_algorithm is not None:
        curr_input_path = curr_output_path
        curr_output_path = (
            curr_output_path.parent
            / f"{curr_output_path.stem}_simpl{curr_output_path.suffix}"
        )

        # If the simplified file doesn't exist yet, go for it...
        if not curr_output_path.exists():
            # Simplify!
            gfo.simplify(
                input_path=curr_input_path,
                output_path=curr_output_path,
                algorithm=gfo.SimplifyAlgorithm(simplify_algorithm),
                tolerance=simplify_tolerance,
                lookahead=simplify_lookahead,
                nb_parallel=nb_parallel,
            )

            # Add/recalculate columns with area and nbcoords
            gfo.add_column(
                path=curr_output_path,
                name="area",
                type=gfo.DataType.REAL,
                expression="ST_Area(geom)",
                force_update=True,
            )
            gfo.add_column(
                path=curr_output_path,
                name="nbcoords",
                type=gfo.DataType.INTEGER,
                expression="ST_NPoints(geom)",
                force_update=True,
            )

        curr_input_path = curr_output_path
        output_paths.append(curr_output_path)

    # If postprocessing steps are defined, the output of the prediction step
    # (input_path) is renamed to ..._orig.gpkg
    if dissolve or reclassify_to_neighbour_query or simplify_algorithm:
        original_file = input_path.parent / f"{input_path.stem}_orig.gpkg"
        input_path.rename(original_file)
        shutil.copy(src=curr_output_path, dst=input_path)

    # Cleanup original file
    if not keep_original_file:
        original_file.unlink()

    # Cleanup intermediary files
    if not keep_intermediary_files:
        for file in output_paths:
            file.unlink()

    return output_paths


def read_prediction_file(
    filepath: Path, border_pixels_to_ignore: int = 0
) -> Optional[gpd.GeoDataFrame]:
    """Read the prediction file specified.

    Args:
        filepath (Path): path to the prediction file.
        border_pixels_to_ignore (int, optional): border pixels the should be ignored.
            Defaults to 0.

    Returns:
        Optional[gpd.GeoDataFrame]: the vectorized and cleaned prediction.
    """
    ext_lower = filepath.suffix.lower()
    if ext_lower == ".geojson":
        return gfo.read_file(filepath)
    elif ext_lower == ".tif":
        return polygonize_pred_from_file(filepath, border_pixels_to_ignore)
    else:
        raise ValueError(f"Unsupported extension: {ext_lower}")


def to_binary_uint8(in_arr: np.ndarray, thresshold_ok: int = 128) -> np.ndarray:
    """Convert input array to binary UINT8.

    Args:
        in_arr (np.ndarray): input array
        thresshold_ok (int, optional): thresshold to use. Defaults to 128.

    Returns:
        np.ndarray: result
    """
    # Check input parameters
    if in_arr.dtype != np.uint8:
        raise ValueError(f"Input should be dtype = uint8, not: {in_arr.dtype}")

    # First copy to new numpy array, otherwise input array is changed
    out_arr = np.copy(in_arr)
    out_arr[out_arr >= thresshold_ok] = 255
    out_arr[out_arr < thresshold_ok] = 0

    return out_arr


def postprocess_for_evaluation(
    image_filepath: Path,
    image_crs: str,
    image_transform,
    image_pred_filepath: Path,
    image_pred_uint8_cleaned_bin: np.ndarray,
    class_id: int,
    class_name: str,
    nb_classes: int,
    output_dir: Path,
    output_suffix: Optional[str] = None,
    input_image_dir: Optional[Path] = None,
    input_mask_dir: Optional[Path] = None,
    border_pixels_to_ignore: int = 0,
    force: bool = False,
):
    """This function postprocesses a prediction for manual evaluation.

    To make it easy to evaluate visually if the result is OK by creating images of the
    different stages of the prediction logic by creating the following output:
        - the input image
        - the mask image as digitized in the train files (if available)
        - the "raw" prediction image
        - the "raw" polygonized prediction, as an image
        - the simplified polygonized prediction, as an image

    The filenames start with a prefix:
        - if a mask is available, the % overlap between the result and the mask
        - if no mask is available, the % of pixels that is white

    Args:
        image_filepath (Path): _description_
        image_crs (str): _description_
        image_transform (_type_): _description_
        image_pred_filepath (Path): _description_
        image_pred_uint8_cleaned_bin (np.ndarray): _description_
        class_id (int): _description_
        class_name (str): _description_
        nb_classes (int): _description_
        output_dir (Path): _description_
        output_suffix (Optional[str], optional): _description_. Defaults to None.
        input_image_dir (Optional[Path], optional): _description_. Defaults to None.
        input_mask_dir (Optional[Path], optional): _description_. Defaults to None.
        border_pixels_to_ignore (int, optional): _description_. Defaults to 0.
        force (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        _type_: _description_
    """
    logger.debug(f"Start postprocess for {image_pred_filepath}")
    all_black = False
    try:
        # If the image wasn't saved, it must have been all black
        if image_pred_filepath is None:
            all_black = True

        # Make sure the output dir exists...
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine the prefix to use for the output filenames
        pred_prefix_str = ""
        """
        def jaccard_similarity(im1: np.ndarray, im2: np.ndarray):
            if im1.shape != im2.shape:
                message = f"input shapes mismatch: im1: {im1.shape}, im2: {im2.shape}"
                logger.critical(message)
                raise ValueError(message)

            intersection = np.logical_and(im1, im2)
            union = np.logical_or(im1, im2)

            sum_union = float(union.sum())
            if sum_union == 0.0:
                # If 0 positive pixels in union: perfect prediction, so 1
                return 1
            else:
                sum_intersect = intersection.sum()
                return sum_intersect/sum_union
        """

        # If there is a mask dir specified... use the groundtruth mask
        if input_mask_dir is not None and input_mask_dir.exists():
            # Read mask file and get all needed info from it...
            mask_filepath = Path(
                str(image_filepath).replace(str(input_image_dir), str(input_mask_dir))
            )

            # Check if this file exists, if not, look for similar files
            if not mask_filepath.exists():
                files = list(mask_filepath.parent.glob(mask_filepath.stem + "*"))
                if len(files) == 1:
                    mask_filepath = files[0]
                else:
                    message = (
                        f"Error finding mask file with {mask_filepath.stem + '*'}: "
                        f"{len(files)} mask(s) found"
                    )
                    logger.error(message)
                    raise Exception(message)

            with rio.open(mask_filepath) as mask_ds:
                # Read pixels
                mask_arr = mask_ds.read(1)

            # Make the pixels at the borders of the mask black so they are
            # ignored in the comparison
            if border_pixels_to_ignore and border_pixels_to_ignore > 0:
                mask_arr[0:border_pixels_to_ignore, :] = 0  # Left border
                mask_arr[-border_pixels_to_ignore:, :] = 0  # Right border
                mask_arr[:, 0:border_pixels_to_ignore] = 0  # Top border
                mask_arr[:, -border_pixels_to_ignore:] = 0  # Bottom border

            # If there is more than 1 class, extract the seperate masks
            # per class with one-hot encoding
            if nb_classes > 1:
                mask_categorical_arr = tf.keras.utils.to_categorical(
                    mask_arr, nb_classes, dtype=rio.uint8
                )
                mask_arr = (mask_categorical_arr[:, :, class_id]) * 255

            # similarity = jaccard_similarity(mask_arr, image_pred)
            # Use accuracy as similarity... is more practical than jaccard
            similarity = (
                np.array(np.equal(mask_arr, image_pred_uint8_cleaned_bin)).sum()
                / image_pred_uint8_cleaned_bin.size
            )
            pred_prefix_str = f"{similarity:0.3f}_"

            # Write mask
            mask_copy_dest_filepath = (
                output_dir
                / f"{pred_prefix_str}{image_filepath.stem}_{class_name}_mask.tif"
            )
            # if not mask_copy_dest_filepath.exists():
            with rio.open(
                mask_copy_dest_filepath,
                "w",
                driver="GTiff",
                compress="lzw",
                height=mask_arr.shape[1],
                width=mask_arr.shape[0],
                count=1,
                dtype=rio.uint8,
                crs=image_crs,
                transform=image_transform,
            ) as dst:
                dst.write(mask_arr, 1)

        else:
            # If all_black, no need to calculate again
            if all_black:
                pct_black = 1
            else:
                # Calculate percentage black pixels
                pct_black = 1 - (
                    (image_pred_uint8_cleaned_bin.sum() / 250)
                    / image_pred_uint8_cleaned_bin.size
                )

            # If the result after segmentation is all black, set all_black
            if pct_black == 1:
                # Force the prefix to be high so it is clear they are entirely black
                pred_prefix_str = "1.001_"
                all_black = True
            else:
                pred_prefix_str = f"{pct_black:0.3f}_"

            # If there are few white pixels, don't save it,
            # because we are in evaluetion mode anyway...
            # if similarity >= 0.95:
            # continue

        # Copy the input image if it doesn't exist yet in output path
        output_basefilepath = (
            output_dir / f"{pred_prefix_str}{image_filepath.stem}{output_suffix}"
        )
        image_dest_filepath = Path(str(output_basefilepath) + image_filepath.suffix)
        if not image_dest_filepath.exists():
            shutil.copyfile(image_filepath, image_dest_filepath)

        # Rename the prediction file so it also contains the prefix,...
        if image_pred_filepath is not None:
            image_dest_filepath = Path(
                f"{output_basefilepath!s}_pred{image_pred_filepath.suffix}"
            )
            if not image_dest_filepath.exists():
                shutil.move(str(image_pred_filepath), image_dest_filepath)

        # If all_black, we are ready now
        if all_black:
            logger.debug("All black prediction, no use proceding")
            return

        # Write a cleaned-up version for evaluation as well
        # polygonize_pred_for_evaluation(
        #        image_pred_uint8_bin=image_pred_uint8_cleaned_bin,
        #        image_crs=image_crs,
        #        image_transform=image_transform,
        #        output_basefilepath=output_basefilepath)

    except Exception as ex:
        message = (
            f"Error postprocessing prediction for {image_filepath}:\n"
            f"    file: {image_pred_filepath}!!!"
        )
        raise Exception(message) from ex


def polygonize_pred_for_evaluation(
    image_pred_uint8_bin, image_crs: str, image_transform, output_basefilepath: Path
):
    """Polygonize a prediction in a way it is easy to evaluate manually/visually.

    Args:
        image_pred_uint8_bin (_type_): _description_
        image_crs (str): _description_
        image_transform (_type_): _description_
        output_basefilepath (Path): _description_
    """
    # Polygonize result
    try:
        # Returns a list of tupples with (geometry, value)
        polygonized_records = list(
            rio_features.shapes(
                image_pred_uint8_bin,
                mask=image_pred_uint8_bin,
                transform=image_transform,
            )
        )

        # If nothing found, we can return
        if len(polygonized_records) == 0:
            logger.debug("This prediction didn't result in any polygons")
            return

        # Convert shapes to geopandas geodataframe
        geoms = []
        for geom, _ in polygonized_records:
            geoms.append(sh_geom.shape(geom))
        geoms_gdf = gpd.GeoDataFrame(geoms, columns=["geometry"])
        geoms_gdf.crs = image_crs

        image_shape = image_pred_uint8_bin.shape
        image_width = image_shape[0]
        image_height = image_shape[1]

        # For easier evaluation, write the cleaned version as raster
        # Write the standard cleaned output to file
        logger.debug("Save binary prediction")
        image_pred_cleaned_filepath = Path(f"{output_basefilepath!s}_pred_bin.tif")
        with rio.open(
            image_pred_cleaned_filepath,
            "w",
            driver="GTiff",
            compress="lzw",
            height=image_height,
            width=image_width,
            count=1,
            dtype=rio.uint8,
            crs=image_crs,
            transform=image_transform,
        ) as dst:
            dst.write(image_pred_uint8_bin, 1)

        # If the input image contained a tranform, also create an image
        # based on the simplified vectors
        if image_transform[0] != 0 and len(geoms) > 0:
            # Simplify geoms
            geoms_simpl = []
            geoms_simpl_vis = []
            for geom in geoms:
                # The simplify of shapely uses the deuter-pecker algo
                # preserve_topology is slower bu makes sure no polygons are removed
                geom_simpl = geom.simplify(0.5, preserve_topology=True)
                if not geom_simpl.is_empty:
                    geoms_simpl.append(geom_simpl)

            # Write simplified wkt result to raster for comparing.
            if len(geoms_simpl) > 0:
                # TODO: doesn't support multiple classes
                logger.debug("Before writing simpl rasterized file")
                image_pred_simpl_filepath = (
                    f"{output_basefilepath!s}_pred_cleaned_simpl.tif"
                )
                with rio.open(
                    image_pred_simpl_filepath,
                    "w",
                    driver="GTiff",
                    compress="lzw",
                    height=image_height,
                    width=image_width,
                    count=1,
                    dtype=rio.uint8,
                    crs=image_crs,
                    transform=image_transform,
                ) as dst:
                    # create a generator of geom, value pairs to use in rasterizing
                    logger.debug("Before rasterize")
                    burned = rio_features.rasterize(
                        shapes=geoms_simpl,
                        out_shape=(image_height, image_width),
                        fill=0,
                        default_value=255,
                        dtype=rio.uint8,
                        transform=image_transform,
                    )
                    dst.write(burned, 1)

            # Write simplified wkt result to raster for comparing. Use the same
            if len(geoms_simpl_vis) > 0:
                # file profile as created before for writing the raw prediction result
                # TODO: doesn't support multiple classes
                logger.debug(
                    "Before writing simpl with visvangali algo rasterized file"
                )
                image_pred_simpl_filepath = (
                    f"{output_basefilepath!s}_pred_cleaned_simpl_vis.tif"
                )
                with rio.open(
                    image_pred_simpl_filepath,
                    "w",
                    driver="GTiff",
                    compress="lzw",
                    height=image_height,
                    width=image_width,
                    count=1,
                    dtype=rio.uint8,
                    crs=image_crs,
                    transform=image_transform,
                ) as dst:
                    # create a generator of geom, value pairs to use in rasterizing
                    logger.debug("Before rasterize")
                    burned = rio_features.rasterize(
                        shapes=geoms_simpl_vis,
                        out_shape=(image_height, image_width),
                        fill=0,
                        default_value=255,
                        dtype=rio.uint8,
                        transform=image_transform,
                    )
                    dst.write(burned, 1)

    except Exception as ex:
        message = f"Exception while polygonizing to file {output_basefilepath}!"
        raise RuntimeError(message) from ex


def polygonize_pred_from_file(
    image_pred_filepath: Path,
    border_pixels_to_ignore: int = 0,
    save_to_file: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    """Polygonize a prediction from a file.

    Args:
        image_pred_filepath (Path): path to the file to be read.
        border_pixels_to_ignore (int, optional): number of pixels around the input image
            to ignore. Defaults to 0.
        save_to_file (bool, optional): If True, save te result to a file.
            Defaults to False.

    Returns:
        Optional[gpd.GeoDataFrame]: _description_
    """
    try:
        with rio.open(image_pred_filepath) as image_ds:
            # Read geo info
            image_crs = image_ds.profile["crs"]
            image_transform = image_ds.transform

            # Read pixels and change from (channels, width, height) to
            # (width, height, channels) and normalize to values between 0 and 1
            image_data = image_ds.read()

        # Create binary version
        # image_data = rio_plot.reshape_as_image(image_data)
        image_pred_uint8_bin = to_binary_uint8(image_data, 125)

        output_basefilepath = None
        if save_to_file is True:
            output_basefilepath = image_pred_filepath.parent / image_pred_filepath.stem
        result_gdf = polygonize_pred(
            image_pred_uint8_bin=image_pred_uint8_bin,
            image_crs=image_crs,
            image_transform=image_transform,
            output_basefilepath=output_basefilepath,
        )

        if result_gdf is None:
            logger.warning(
                f"Prediction didn't result in any polygons: {image_pred_filepath}"
            )

        return result_gdf

    except Exception as ex:
        raise RuntimeError(
            f"Error in polygonize_pred_from_file on {image_pred_filepath}"
        ) from ex


def polygonize_pred_multiclass_to_file(
    image_pred_arr: np.ndarray,
    image_crs: str,
    image_transform,
    classes: list,
    output_vector_path: Path,
    min_probability: float = 0.5,
    postprocess: dict = {},
    border_pixels_to_ignore: int = 0,
    create_spatial_index: bool = True,
) -> dict:
    """Polygonize a multiclass prediction to a file.

    Args:
        image_pred_arr (np.ndarray): _description_
        image_crs (str): _description_
        image_transform (_type_): _description_
        classes (list): _description_
        output_vector_path (Path): _description_
        min_probability (float, optional): _description_. Defaults to 0.5.
        postprocess (dict, optional): _description_. Defaults to {}.
        border_pixels_to_ignore (int, optional): _description_. Defaults to 0.
        create_spatial_index (bool, optional): _description_. Defaults to True.

    Returns:
        dict: _description_
    """
    # Polygonize the result...
    result_gdf = polygonize_pred_multiclass(
        image_pred_uint8=image_pred_arr,
        image_crs=image_crs,
        image_transform=image_transform,
        classes=classes,
        min_probability=min_probability,
        postprocess=postprocess,
        border_pixels_to_ignore=border_pixels_to_ignore,
    )

    # If there were polygons, save them...
    if result_gdf is not None:
        gfo.to_file(
            result_gdf,
            output_vector_path,
            append=True,
            index=False,
            force_multitype=True,
            create_spatial_index=create_spatial_index,
        )
        return {"nb_features_witten": len(result_gdf), "columns": result_gdf.columns}
    else:
        return {"nb_features_witten": None}


def polygonize_pred_multiclass(
    image_pred_uint8: np.ndarray,
    image_crs: str,
    image_transform,
    classes: list,
    min_probability: float = 0.5,
    postprocess: dict = {},
    border_pixels_to_ignore: int = 0,
) -> Optional[gpd.GeoDataFrame]:
    """Polygonize a multiclass prediction.

    Args:
        image_pred_uint8 (np.ndarray): _description_
        image_crs (str): _description_
        image_transform (_type_): _description_
        classes (list): _description_
        min_probability (float, optional): _description_. Defaults to 0.5.
        postprocess (dict, optional): _description_. Defaults to {}.
        border_pixels_to_ignore (int, optional): _description_. Defaults to 0.

    Returns:
        Optional[gpd.GeoDataFrame]: _description_
    """
    # Init
    """
    for channel_id in range(0, nb_channels):
        image_pred_curr_arr = image_pred_arr[:,:,channel_id]

        # Clean prediction
        image_pred_uint8_cleaned_curr = clean_prediction(
                image_pred_arr=image_pred_curr_arr,
                border_pixels_to_ignore=border_pixels_to_ignore)
    """

    # Reverse the one-hot decoding so each class has it's own number in the array,
    # but ignore prediction probability < min_probability
    image_pred_uint8[image_pred_uint8 < math.floor(255 * min_probability)] = 0
    image_pred_decoded_arr = np.argmax(image_pred_uint8, axis=2).astype(np.uint8)

    # Make the pixels at the borders of the prediction black so they are ignored
    if border_pixels_to_ignore and border_pixels_to_ignore > 0:
        image_pred_decoded_arr[0:border_pixels_to_ignore, :] = 0  # Left border
        image_pred_decoded_arr[-border_pixels_to_ignore:, :] = 0  # Right border
        image_pred_decoded_arr[:, 0:border_pixels_to_ignore] = 0  # Top border
        image_pred_decoded_arr[:, -border_pixels_to_ignore:] = 0  # Bottom border

    # Postprocessing on the raster output
    if len(postprocess) > 0:
        # If fill_gaps_modal_size is asked...
        if (
            "filter_background_modal_size" in postprocess
            and postprocess["filter_background_modal_size"] is not None
            and postprocess["filter_background_modal_size"] > 0
        ):
            filter_background_modal_size = postprocess["filter_background_modal_size"]
            image_pred_decoded_modal_arr = skimage.filters.rank.modal(
                image_pred_decoded_arr,
                rectangle(filter_background_modal_size, filter_background_modal_size),
            )
            np.copyto(
                image_pred_decoded_arr,
                image_pred_decoded_modal_arr,
                where=image_pred_decoded_arr == 0,
            )

    # Polygonize
    # If a reclassify query is specified don't mask so the query is also applied to
    # the background
    mask_background = True
    if (
        len(postprocess) > 0
        and postprocess.get("reclassify_to_neighbour_query", None) is not None
    ):
        mask_background = False

    result_gdf = polygonize_pred(
        image_pred_uint8_bin=image_pred_decoded_arr,
        image_crs=image_crs,
        image_transform=image_transform,
        mask_background=mask_background,
        classnames=classes,
    )
    if result_gdf is None:
        return

    # Calculate the bounds of the image in projected coordinates
    image_shape = image_pred_decoded_arr.shape
    image_width = image_shape[0]
    image_height = image_shape[1]
    image_bounds = rio_transform.array_bounds(
        image_height, image_width, image_transform
    )
    x_pixsize = get_pixelsize_x(image_transform)
    y_pixsize = get_pixelsize_y(image_transform)
    border_bounds = (
        image_bounds[0] + border_pixels_to_ignore * x_pixsize,
        image_bounds[1] + border_pixels_to_ignore * y_pixsize,
        image_bounds[2] - border_pixels_to_ignore * x_pixsize,
        image_bounds[3] - border_pixels_to_ignore * y_pixsize,
    )

    # Postprocessing on the vectorized result
    if len(postprocess) > 0:
        # If a reclassify query is specified
        reclassify_query = postprocess.get("reclassify_to_neighbour_query", None)
        if reclassify_query is not None:
            result_gdf = vector_util.reclassify_neighbours(
                result_gdf,
                reclassify_column="classname",
                query=reclassify_query,
                border_bounds=border_bounds,
                class_background=classes[0],
            )

        # If a simplify is asked...
        simplify = postprocess.get("simplify", None)
        if simplify is not None:
            # Define the bounds of the image as linestring, so points on this
            # border are preserved during the simplify
            border_polygon = sh_geom.box(*border_bounds)
            assert border_polygon.exterior is not None
            border_lines = sh_geom.LineString(border_polygon.exterior.coords)

            # Determine of topological or normal simplify needs to be used
            simplify_topological = simplify["simplify_topological"]
            if simplify_topological is None:
                simplify_topological = True if len(classes) > 2 else False

            assert isinstance(result_gdf.geometry, gpd.GeoSeries)
            result_gdf.geometry = pygeoops.simplify(
                geometry=result_gdf.geometry,
                algorithm=simplify["simplify_algorithm"],
                tolerance=simplify["simplify_tolerance"],
                lookahead=simplify["simplify_lookahead"],
                preserve_common_boundaries=simplify_topological,
                keep_points_on=border_lines,
            )

            # Remove geom rows that became empty after simplify + explode
            assert result_gdf.geometry is not None
            result_gdf = result_gdf[~result_gdf.geometry.is_empty]
            result_gdf = result_gdf[~result_gdf.geometry.isna()]
            if len(result_gdf) == 0:
                return None
            result_gdf = result_gdf.explode(ignore_index=True)

    assert isinstance(result_gdf, gpd.GeoDataFrame)
    return result_gdf


def polygonize_pred(
    image_pred_uint8_bin,
    image_crs: str,
    image_transform,
    mask_background: bool = True,
    classnames: Optional[list[str]] = None,
    output_basefilepath: Optional[Path] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Polygonize a prediction.

    Args:
        image_pred_uint8_bin (_type_): _description_
        image_crs (str): _description_
        image_transform (_type_): _description_
        mask_background (bool, optional): _description_. Defaults to True.
        classnames (Optional[List[str]], optional): _description_. Defaults to None.
        output_basefilepath (Optional[Path], optional): _description_. Defaults to None.

    Raises:
        Exception: _description_

    Returns:
        Optional[gpd.GeoDataFrame]: _description_
    """
    # Polygonize result
    try:
        # Returns a list of tupples with (geometry, value)
        mask = None
        if mask_background:
            mask = image_pred_uint8_bin
        polygonized_records = list(
            rio_features.shapes(
                image_pred_uint8_bin,
                mask=mask,
                transform=image_transform,
            )
        )

        # If nothing found, we can return
        if len(polygonized_records) == 0:
            return None

        # Convert shapes to geopandas geodataframe
        data = [
            (sh_geom.shape(geom), int(value)) for geom, value in polygonized_records
        ]
        result_gdf = gpd.GeoDataFrame(
            data, columns=["geometry", "value"], crs=image_crs
        )

        # Add the classname if provided
        if classnames is not None:
            result_gdf["classname"] = [
                classnames[value] for _, value in result_gdf["value"].T.items()
            ]
            result_gdf = result_gdf.drop(columns=["value"])

        assert isinstance(result_gdf, gpd.GeoDataFrame)
        return result_gdf

    except Exception as ex:
        message = f"Exception while polygonizing to file {output_basefilepath}"
        raise Exception(message) from ex


def clean_and_save_prediction(
    input_image_filepath: Path,
    image_crs: str,
    image_transform: str,
    output_dir: Path,
    image_pred_arr: np.ndarray,
    classes: list,
    input_image_dir: Optional[Path] = None,
    input_mask_dir: Optional[Path] = None,
    border_pixels_to_ignore: int = 0,
    min_probability: float = 0.5,
    evaluate_mode: bool = False,
    force: bool = False,
) -> bool:
    """Clean the prediction and save it.

    Args:
        input_image_filepath (Path): _description_
        image_crs (str): _description_
        image_transform (str): _description_
        output_dir (Path): _description_
        image_pred_arr (np.ndarray): _description_
        classes (list): _description_
        input_image_dir (Optional[Path], optional): _description_. Defaults to None.
        input_mask_dir (Optional[Path], optional): _description_. Defaults to None.
        border_pixels_to_ignore (int, optional): _description_. Defaults to 0.
        min_probability (float, optional): _description_. Defaults to 0.5.
        evaluate_mode (bool, optional): _description_. Defaults to False.
        force (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_

    Returns:
        bool: _description_
    """
    # If nb. channels in prediction > 1, skip the first as it is the background
    image_pred_shape = image_pred_arr.shape
    nb_channels = image_pred_shape[2]
    if nb_channels > 1:
        channel_start = 1
    else:
        channel_start = 0

    for channel_id in range(channel_start, nb_channels):
        image_pred_curr_arr = image_pred_arr[:, :, channel_id]

        # Clean prediction
        image_pred_uint8_cleaned_curr = clean_prediction(
            image_pred_arr=image_pred_curr_arr,
            border_pixels_to_ignore=border_pixels_to_ignore,
        )

        # If the cleaned result contains useful values or in evaluate mode... save
        if (
            min_probability == 0
            or np.any(
                image_pred_uint8_cleaned_curr >= math.floor(min_probability * 255)
            )
            or evaluate_mode is True
        ):
            # Find the class name in the classes list
            class_name = None
            for class_id, (classname) in enumerate(classes):
                if class_id == channel_id:
                    class_name = classname
                    break
            if class_name is None:
                raise Exception(f"No classname found for channel_id {channel_id}")

            # Now save prediction
            output_suffix = f"_{class_name}"
            image_pred_filepath = save_prediction_uint8(
                image_filepath=input_image_filepath,
                image_pred_uint8_cleaned=image_pred_uint8_cleaned_curr,
                image_crs=image_crs,
                image_transform=image_transform,
                output_dir=output_dir,
                output_suffix=output_suffix,
                force=force,
            )

            # Postprocess for evaluation
            if evaluate_mode is True:
                # Create binary version and postprocess
                image_pred_uint8_cleaned_bin = to_binary_uint8(
                    image_pred_uint8_cleaned_curr, 125
                )
                postprocess_for_evaluation(
                    image_filepath=input_image_filepath,
                    image_crs=image_crs,
                    image_transform=image_transform,
                    image_pred_filepath=image_pred_filepath,
                    image_pred_uint8_cleaned_bin=image_pred_uint8_cleaned_bin,
                    output_dir=output_dir,
                    output_suffix=output_suffix,
                    input_image_dir=input_image_dir,
                    input_mask_dir=input_mask_dir,
                    class_id=channel_id,
                    class_name=class_name,
                    nb_classes=nb_channels,
                    border_pixels_to_ignore=border_pixels_to_ignore,
                    force=force,
                )

    return True


def clean_prediction(
    image_pred_arr: np.ndarray,
    border_pixels_to_ignore: int = 0,
    output_color_depth: str = "binary",
) -> np.ndarray:
    """Cleans a prediction result and returns a cleaned, uint8 array.

    Args:
        image_pred_arr (np.array): The prediction as returned by keras.
        border_pixels_to_ignore (int, optional): Border pixels to ignore. Defaults to 0.
        output_color_depth (str, optional): Color depth desired. Defaults to '2'.
            * binary: 0 or 255
            * full: 256 different values

    Returns:
        np.array: The cleaned result.
    """
    # Input should be float32
    if image_pred_arr.dtype not in [np.float32, np.uint8]:
        raise Exception(
            f"image prediction is in an unsupported type: {image_pred_arr.dtype}"
        )
    if output_color_depth not in ["binary", "full"]:
        raise Exception(f"Unsupported output_color_depth: {output_color_depth}")

    # Reshape from 3 to 2 dims if necessary (width, height, nb_channels).
    # Check the number of channels of the output prediction
    image_pred_shape = image_pred_arr.shape
    if len(image_pred_shape) > 2:
        n_channels = image_pred_shape[2]
        if n_channels > 1:
            raise Exception("Invalid input, should be one channel!")
        # Reshape array from 3 dims (width, height, nb_channels) to 2.
        image_pred_uint8 = np.reshape(
            image_pred_arr, (image_pred_shape[0], image_pred_shape[1])
        )

    # Convert to uint8 if necessary
    if image_pred_arr.dtype == np.float32:
        image_pred_uint8 = np.array((image_pred_arr * 255), dtype=np.uint8)
    else:
        image_pred_uint8 = image_pred_arr

    # Convert to binary if needed
    if output_color_depth == "binary":
        image_pred_uint8[image_pred_uint8 >= 127] = 255
        image_pred_uint8[image_pred_uint8 < 127] = 0

    # Make the pixels at the borders of the prediction black so they are ignored
    image_pred_uint8_cropped = image_pred_uint8
    if border_pixels_to_ignore and border_pixels_to_ignore > 0:
        image_pred_uint8_cropped[0:border_pixels_to_ignore, :] = 0  # Left border
        image_pred_uint8_cropped[-border_pixels_to_ignore:, :] = 0  # Right border
        image_pred_uint8_cropped[:, 0:border_pixels_to_ignore] = 0  # Top border
        image_pred_uint8_cropped[:, -border_pixels_to_ignore:] = 0  # Bottom border

    return image_pred_uint8_cropped


def save_prediction_uint8(
    image_filepath: Path,
    image_pred_uint8_cleaned: np.ndarray,
    image_crs: str,
    image_transform: str,
    output_dir: Path,
    output_suffix: str = "",
    border_pixels_to_ignore: Optional[int] = None,
    force: bool = False,
) -> Path:
    """Save the prediction as UINT8.

    Args:
        image_filepath (Path): _description_
        image_pred_uint8_cleaned (np.ndarray): _description_
        image_crs (str): _description_
        image_transform (str): _description_
        output_dir (Path): _description_
        output_suffix (str, optional): _description_. Defaults to "".
        border_pixels_to_ignore (Optional[int], optional): _description_.
            Defaults to None.
        force (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_

    Returns:
        Path: _description_
    """
    # Init
    # If no decent transform metadata, stop!
    if image_transform is None or image_transform[0] == 0:
        message = f"No transform found for {image_filepath}: {image_transform}"
        logger.error(message)
        raise Exception(message)

    # Make sure the output dir exists...
    if not output_dir.exists():
        output_dir.mkdir()

    # Write prediction to file
    output_filepath = output_dir / f"{image_filepath.stem}{output_suffix}_pred.tif"
    logger.debug("Save +- original prediction")
    image_shape = image_pred_uint8_cleaned.shape
    image_width = image_shape[0]
    image_height = image_shape[1]
    with rio.open(
        str(output_filepath),
        "w",
        driver="GTiff",
        tiled="no",
        compress="lzw",
        predictor=2,
        num_threads=4,
        height=image_height,
        width=image_width,
        count=1,
        dtype=rio.uint8,
        crs=image_crs,
        transform=image_transform,
    ) as dst:
        dst.write(image_pred_uint8_cleaned, 1)

    return output_filepath


# -------------------------------------------------------------
# Helpers for working with Affine objects...
# -------------------------------------------------------------


def get_pixelsize_x(transform) -> float:
    """Get the x pixel size from the transform.

    Args:
        transform (_type_): input transform

    Returns:
        float: the x pixel size.
    """
    return transform[0]


def get_pixelsize_y(transform):
    """Get the y pixel size from the transform.

    Args:
        transform (_type_): input transform

    Returns:
        float: the y pixel size.
    """
    return -transform[4]

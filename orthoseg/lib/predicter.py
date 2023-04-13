# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images.
"""

from concurrent import futures
import csv
import datetime
import json
import logging
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from typing import List, Optional

import geofileops as gfo
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.crs as rio_crs
import rasterio.plot as rio_plot
import tensorflow as tf
import keras.models

import orthoseg.lib.postprocess_predictions as postp
from orthoseg.util.progress_util import ProgressLogger
from orthoseg.util import general_util

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def predict_dir(
    model: keras.models.Model,
    input_image_dir: Path,
    output_image_dir: Path,
    output_vector_path: Optional[Path],
    classes: list,
    min_probability: float = 0.5,
    postprocess: dict = {},
    border_pixels_to_ignore: int = 0,
    projection_if_missing: Optional[str] = None,
    input_mask_dir: Optional[Path] = None,
    batch_size: int = 16,
    evaluate_mode: bool = False,
    cancel_filepath: Optional[Path] = None,
    nb_parallel_postprocess: int = 1,
    max_prediction_errors: int = 100,
    force: bool = False,
):
    """
    Create a prediction for all the images in the directories specified
    using the model specified.

    If evaluate_mode is False, the output folder(s) will contain:
        * the "raw" prediction for every image (if there are white pixels)
        * a geojson with the vectorized prediction, with a column "onborder"
          for each feature that is 1 if the feature is on the border of the
          tile, taking the border_pixels_to_ignore in account if applicable.
          This columns can be used to speed up union operations afterwards,
          because only features on the border of tiles need to be unioned.

    If evaluate_mode is True, the results will all be put in the root of the
    output folder, and the following files will be outputted:
        * the original image
        * the mask that was provided, if available
        * the "raw" prediction
        * a "cleaned" version of the prediction
    The files will in this case be prefixed with a number so they are ordered
    in a way that is interesting for evaluation. If a mask was available, this
    prefix will be the % overlap of the mask and the prediction. If no mask is
    available, the prefix is the % white pixels in the prediction.

    Args
        input_image_dir (Pathlike): dir where the input images are located
        output_base_dir (Pathlike): dir where the output will be put
        output_vector_path (Pathlike): the path to write the vector output to
        classes (list): a list of the different class names. Mandatory
            if more than background + 1 class.
        postprocess (dict, optional): specifies which postprocessing should be applied
            to the prediction. Default is {}, so no postprocessing.
        border_pixels_to_ignore: because the segmentation at the borders of the
            input images images is not as good, you can specify that x
            pixels need to be ignored
        input_mask_dir: optional dir where the mask images are located
        projection_if_missing: Normally the projection should be in the raster file. If
            it is not, you can explicitly specify one.
        batch_size: batch size to use while predicting. This must be choosen
            depending on the neural network architecture and available
            memory on you GPU.
        evaluate_mode: True to run in evaluate mode
        cancel_filepath: If the file in this path exists, processing stops asap
        nb_parallel_postprocess (int, optional): The number of parallel
            processes used to vectorize,... the predictions. If -1, all
            available CPU's are used. Defaults to 1.
        max_prediction_errors (int, optional): the maximum number of errors that is
            tolerated before stopping prediction. If -1, no limit. Defaults to 100.
        force: False to skip images that already have a prediction, true to
            ignore existing predictions and overwrite them
    """

    # Init
    if not input_image_dir.exists():
        logger.warning(f"input_image_dir doesn't exist, so return: {input_image_dir}")
        return
    if output_vector_path is not None and output_vector_path.exists():
        logger.warning(f"output file exists already, so return: {output_vector_path}")
        return
    tmp_dir = Path(tempfile.gettempdir()) / Path(__file__).stem

    logger.info(f"Start predict for input_image_dir: {input_image_dir}")

    # Eager and not eager prediction seems +- the same performance-wise
    # model.run_eagerly = False

    # If we are using evaluate mode, change the output dir...
    if evaluate_mode:
        output_image_dir = Path(str(output_image_dir) + "_eval")

    # Create the output dir's if they don't exist yet...
    for dir in [output_image_dir, tmp_dir]:
        if not dir.exists():
            dir.mkdir()

    # Write prediction config used, so it can be used for postprocessing
    prediction_config_path = output_image_dir / "prediction_config.json"
    with open(prediction_config_path, "w") as pred_conf_file:
        pred_conf = {}
        pred_conf["border_pixels_to_ignore"] = border_pixels_to_ignore
        pred_conf["classes"] = classes
        json.dump(pred_conf, pred_conf_file)

    # Get list of all image files to process and to skip
    image_filepaths: List[Path] = []
    input_ext = [".png", ".tif", ".jpg"]
    for input_ext_cur in input_ext:
        image_filepaths.extend(input_image_dir.rglob("*" + input_ext_cur))
    image_filepaths = sorted(image_filepaths)
    nb_images = len(image_filepaths)
    logger.info(
        f"Found {nb_images} {input_ext} images to predict on in {input_image_dir}"
    )

    # If force is false, get list of all existing predictions
    # Getting the list once is way faster than checking file per file later on!
    images_done_log_filepath = output_image_dir / "images_done.txt"
    image_done_filenames = set()
    if force is False:
        # First read the listing files if they exists
        if images_done_log_filepath.exists():
            with images_done_log_filepath.open() as f:
                for filename in f:
                    image_done_filenames.add(filename.rstrip())
        if len(image_done_filenames) > 0:
            logger.info(
                f"{len(image_done_filenames)} images in were processed before, "
                f"they will be skipped"
            )

    # Clear error file
    images_error_log_filepath = output_image_dir / "images_error.csv"
    images_error_log_filepath.unlink(missing_ok=True)

    # Write output to tmp files, so we know if the process completed correctly or not
    pred_tmp_output_path = None
    if output_vector_path is not None:
        pred_tmp_output_path = output_image_dir / f"{output_vector_path.stem}_tmp.gpkg"
        pred_tmp_output_lock_path = Path(f"{str(pred_tmp_output_path)}.lock")
        # if lock file exists, remove it:
        if pred_tmp_output_lock_path.exists():
            pred_tmp_output_lock_path.unlink()

    # Loop through all files to process them
    nb_parallel_read = batch_size * 6
    if nb_parallel_postprocess == -1:
        nb_parallel_postprocess = multiprocessing.cpu_count()
    predict_images = []
    nb_to_process = nb_images
    nb_processed = 0
    nb_errors = 0
    read_sleep_logged = False
    progress = None
    postp_future_to_input_path = {}
    read_future_to_input_path = {}
    image_id = -1
    last_image_reached = False

    # We don't want the postprocess workers to block the entire system,
    # so make them a bit nicer
    def init_postprocess_worker():
        general_util.setprocessnice(15)

    with futures.ThreadPoolExecutor(
        nb_parallel_read
    ) as read_pool, futures.ProcessPoolExecutor(
        nb_parallel_postprocess, initializer=init_postprocess_worker()
    ) as postprocess_pool:

        # Start looping.
        # If ready to stop, the code below will break
        perf_time_start = datetime.datetime.now()
        while True:

            # If we are ready, stop!
            if (
                last_image_reached is True
                and len(read_future_to_input_path) == 0
                and len(postp_future_to_input_path) == 0
            ):
                break

            # If the cancel file exists, stop processing...
            if cancel_filepath is not None and cancel_filepath.exists():
                print()
                logger.info(f"Cancel file found, so stop: {cancel_filepath}")
                break

            # Get the next filepath to be processed
            image_filepath = None
            if image_id < (len(image_filepaths) - 1):
                image_id += 1
                image_filepath = image_filepaths[image_id]

                # Check if the image has been processed already
                if force is False and image_filepath.name in image_done_filenames:
                    logger.debug(
                        "Predict for image has already been done before and force is "
                        f"False, so skip: {image_filepath.name}"
                    )
                    nb_to_process -= 1
                    continue

                nb_processed += 1

                # Schedule file to be read
                read_future = read_pool.submit(
                    read_image,  # Function
                    image_filepath,  # Arg 1
                    projection_if_missing,
                )  # Arg 2)
                read_future_to_input_path[read_future] = image_filepath
            else:
                last_image_reached = True

            # Prepare batch_size images that have been read for prediction
            while (
                len(read_future_to_input_path) > 0 and len(predict_images) < batch_size
            ):

                # If the read pool is not full, first schedule extra reads
                if (
                    last_image_reached is False
                    and len(read_future_to_input_path) < nb_parallel_read
                ):
                    break

                # Prepare the images that have been read for predicting
                futures_done = [
                    future
                    for future in read_future_to_input_path
                    if future.done() is True
                ]
                for future in futures_done:

                    # Prediction can only handle batch_size images
                    if len(predict_images) >= batch_size:
                        break

                    try:
                        # Get the result from the read
                        read_result = future.result()
                        image_filepath_read = read_result["image_filepath"]

                        # Prepare the filepath for the output
                        output_suffix = ".tif"
                        if evaluate_mode:
                            # In evaluate mode, put everyting in output base dir for
                            # easier comparison
                            output_image_pred_dir = output_image_dir

                            # Prepare complete filepath for image prediction
                            output_image_pred_path = (
                                output_image_dir / image_filepath_read.stem
                            )
                        else:
                            # If saving predictions to images for real, keep hierarchic
                            # structure if present
                            tmp_output_filepath = Path(
                                str(image_filepath_read).replace(
                                    str(input_image_dir), str(output_image_dir)
                                )
                            )
                            output_image_pred_dir = tmp_output_filepath.parent
                            output_image_pred_path = (
                                output_image_pred_dir
                                / f"{image_filepath_read.stem}_pred{output_suffix}"
                            )

                        predict_images.append(
                            {
                                "input_image_filepath": image_filepath_read,
                                "output_pred_filepath": output_image_pred_path,
                                "output_image_pred_dir": output_image_pred_dir,
                                "image_crs": read_result["image_crs"],
                                "image_transform": read_result["image_transform"],
                                "image_data": read_result["image_data"],
                            }
                        )

                    finally:
                        # Remove from queue...
                        del read_future_to_input_path[future]

                # If not at last image + not enough images yet for predict + read
                # queue is full, sleep
                if (
                    last_image_reached is False
                    and len(predict_images) < batch_size
                    and len(read_future_to_input_path) >= nb_parallel_read
                ):
                    if read_sleep_logged is False:
                        logger.info("Wait for images to be read")
                        read_sleep_logged = True
                    time.sleep(0.01)

            # If batch_size images are ready for prediction or we are at
            # the last images: predict
            if len(predict_images) == batch_size or (
                last_image_reached is True and len(predict_images) > 0
            ):
                read_sleep_logged = False
                perf_time_now = datetime.datetime.now()
                perfinfo = f"waiting for read took {perf_time_now-perf_time_start}"
                perf_time_start = perf_time_now

                # Predict!
                logger.debug(f"Start prediction for {len(predict_images)} images")
                perf_time_start = datetime.datetime.now()
                curr_batch_image_list = [
                    curr_batch_image_info["image_data"]
                    for curr_batch_image_info in predict_images
                ]
                curr_batch_image_arr = np.stack(curr_batch_image_list)
                curr_batch_image_pred_arr = model.predict_on_batch(curr_batch_image_arr)

                perf_time_now = datetime.datetime.now()
                perfinfo += f", predict took {perf_time_now-perf_time_start}"
                perf_time_start = perf_time_now

                # In tf > 2.1 a tf.tensor object is returned, but we want an ndarray
                if type(curr_batch_image_pred_arr) is tf.Tensor:
                    curr_batch_image_pred_arr = np.array(
                        curr_batch_image_pred_arr.numpy()  # type: ignore
                    )
                else:
                    curr_batch_image_pred_arr = np.array(curr_batch_image_pred_arr)

                # Save predictions
                # Remark: trying to parallelize this doesn't seem to help at all!
                logger.debug("Start post-processing")
                for batch_image_id, image_info in enumerate(predict_images):
                    try:
                        # If not in evaluate mode... save to vector in background
                        if (
                            evaluate_mode is False
                            and output_vector_path is not None
                            and pred_tmp_output_path is not None
                        ):
                            # Prepare prediction array...
                            #   - convert to uint8 to reduce pickle size/time
                            image_pred_arr_uint8 = (
                                (curr_batch_image_pred_arr[batch_image_id, :, :, :])
                                * 255
                            ).astype(np.uint8)
                            future = postprocess_pool.submit(
                                postp.polygonize_pred_multiclass_to_file,
                                image_pred_arr_uint8,
                                image_info["image_crs"],
                                image_info["image_transform"],
                                classes,
                                pred_tmp_output_path,
                                min_probability,
                                postprocess,
                                border_pixels_to_ignore,
                            )
                            postp_future_to_input_path[future] = image_info[
                                "input_image_filepath"
                            ]

                        else:
                            # Saving the predictions as images at the moment only used
                            # for evaluate mode...
                            # TODO: would ideally be moved to the background
                            # processing as well to simplify code here...
                            postp.clean_and_save_prediction(
                                image_image_filepath=image_info["input_image_filepath"],
                                image_crs=image_info["image_crs"],
                                image_transform=image_info["image_transform"],
                                image_pred_arr=curr_batch_image_pred_arr[
                                    batch_image_id
                                ],
                                output_dir=image_info["output_image_pred_dir"],
                                input_image_dir=input_image_dir,
                                input_mask_dir=input_mask_dir,
                                border_pixels_to_ignore=border_pixels_to_ignore,
                                min_probability=min_probability,
                                evaluate_mode=evaluate_mode,
                                classes=classes,
                                force=force,
                            )

                            # Write filepath to file with files that are done
                            with images_done_log_filepath.open(
                                "a+"
                            ) as image_donelog_file:
                                image_donelog_file.write(
                                    f"{image_info['input_image_filepath'].name}\n"
                                )
                    except Exception as ex:
                        nb_errors += 1
                        image_path = image_info["input_image_filepath"]
                        _handle_error(image_path, ex, images_error_log_filepath)

                perf_time_now = datetime.datetime.now()
                perfinfo += (
                    f", scheduling postprocessings took {perf_time_now-perf_time_start}"
                )
                perf_time_start = perf_time_now

                # Poll for completed postprocessings
                postp_sleep_logged = False
                while len(postp_future_to_input_path) > 0:

                    # If not at last file, get results from all futures that are
                    # done, if at last file, wait till all are done
                    if last_image_reached is False:
                        futures_done = [
                            future
                            for future in postp_future_to_input_path
                            if future.done() is True
                        ]
                    else:
                        logger.info("Wait for last batch")
                        futures_done = futures.wait(postp_future_to_input_path).done
                    for future in futures_done:
                        # Get the result from the polygonization
                        try:
                            # Get the result (= exception when something went wrong)
                            result = future.result()
                            logger.debug(
                                f"result for {postp_future_to_input_path[future].name}:"
                                f" {result}"
                            )

                            # Write filepath to file with files that are done
                            with images_done_log_filepath.open(
                                "a+"
                            ) as image_donelog_file:
                                image_donelog_file.write(
                                    postp_future_to_input_path[future].name + "\n"
                                )
                        except Exception as ex:
                            nb_errors += 1
                            image_path = postp_future_to_input_path[future]
                            _handle_error(image_path, ex, images_error_log_filepath)

                        finally:
                            # Remove from queue...
                            del postp_future_to_input_path[future]

                    # Wait till number below thresshold to evade huge waiting
                    # list (and memory issues)
                    if len(postp_future_to_input_path) > nb_parallel_postprocess * 2:
                        if postp_sleep_logged is False:
                            logger.info(
                                "Postprocessing takes longer than prediction, so wait"
                            )
                            postp_sleep_logged = True
                        time.sleep(0.01)
                    else:
                        # No need to wait (anymore)...
                        if postp_sleep_logged is True:
                            logger.info(
                                "Waited enough for postprocessing to catch up..."
                            )

                        perf_time_now = datetime.datetime.now()
                        perfinfo += (
                            f", after postproces: took {perf_time_now-perf_time_start}"
                        )
                        perf_time_start = perf_time_now
                        break

                # Reset variable for next batch
                predict_images = []

                # Log the progress and prediction speed
                if len(perfinfo) > 0:
                    logger.debug(perfinfo)
                if progress is not None:
                    progress.step(nb_steps=batch_size)
                # Init progress only when some imags were already processed, as the
                # first are very slow.
                if progress is None and nb_processed > 0:
                    progress = ProgressLogger(
                        message=f"predict to {output_image_dir.parent.name}/{output_image_dir.name}",  # noqa: E501
                        nb_steps_total=nb_to_process,
                        nb_steps_done=batch_size,
                    )

                # If max number errors reached, stop processings
                if max_prediction_errors >= 0 and nb_errors >= max_prediction_errors:
                    break

        # If errors occured, raise error
        if images_error_log_filepath.exists():
            errors = pd.read_csv(
                images_error_log_filepath,
                usecols=["filename", "error"]
                # ).to_string(justify="left", index=False)
            ).to_html(justify="left", index=False)
            raise Exception(f"Error(s) occured while predicting:\n{errors}")

        # If alle images were processed, rename to real output file + cleanup
        if (
            last_image_reached is True
            and output_vector_path is not None
            and pred_tmp_output_path is not None
            and pred_tmp_output_path.exists()
        ):
            output_vector_path.parent.mkdir(parents=True, exist_ok=True)
            gfo.move(pred_tmp_output_path, output_vector_path)
            gfo.rename_layer(output_vector_path, output_vector_path.stem)
            shutil.rmtree(output_image_dir)


def _handle_error(image_path: Path, ex: Exception, log_path: Path):
    # Print exception + trace
    exception_trace = traceback.format_exc()
    exception_trace_print = exception_trace.replace("\n", "\n\t")
    logger.error(f"Error postprocessing pred for {image_path}: {exception_trace_print}")

    # Write error to error log file
    first_error = False
    if not log_path.exists():
        first_error = True
    with log_path.open("a+") as log_file:
        if first_error:
            log_file.write("filename,error,traceback\n")
        writer = csv.writer(log_file)
        exception_trace_csv = exception_trace.replace("\n", "\\n")
        fields = [image_path.name, ex, exception_trace_csv]
        writer.writerow(fields)


def read_image(
    image_filepath: Path, projection_if_missing: Optional[str] = None
) -> dict:

    # Read input file
    # Because sometimes a read seems to fail, retry up to 3 times...
    retry_count = 0
    while True:
        try:
            with rio.open(str(image_filepath)) as image_ds:
                # Read geo info
                image_crs = image_ds.profile["crs"]
                image_transform = image_ds.transform

                # Read pixels + change from (channels, width, height) to
                # (width, height, channels) + normalize to between 0 and 1
                image_data = image_ds.read()
                image_data = rio_plot.reshape_as_image(image_data)
                image_data = image_data / 255.0

            # Read worked, so jump out of the loop...
            break
        except Exception as ex:
            retry_count += 1
            logger.warning(f"Read failed, retry nb {retry_count} for {image_filepath}")
            if retry_count >= 3:
                message = f"Read failed {retry_count} times for {image_filepath}: {ex}"
                logger.error(message)
                raise Exception(message)

    # The read was successfull, now check if there was a projection in the
    # file and/or if one was provided
    if image_crs is None:
        if projection_if_missing is not None:
            image_crs = rio_crs.CRS.from_string(projection_if_missing)
        else:
            message = (
                f"Image has no proj and projection_if_missing is None: {image_filepath}"
            )
            logger.error(message)
            raise Exception(message)

    # Now return the result
    result = {
        "image_data": image_data,
        "image_crs": image_crs,
        "image_transform": image_transform,
        "image_filepath": image_filepath,
    }
    return result


# If the script is ran directly...
if __name__ == "__main__":
    raise Exception("Not implemented")

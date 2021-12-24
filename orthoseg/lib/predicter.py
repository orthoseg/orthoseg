# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images.
"""

from concurrent import futures
import datetime
import json
import logging
import multiprocessing
from pathlib import Path
import shutil
import tempfile
from typing import List, Optional

from geofileops import geofile
import numpy as np
import rasterio as rio
import rasterio.crs as rio_crs
import rasterio.plot as rio_plot
import tensorflow as tf
import tensorflow.keras.models as kr_models

import orthoseg.lib.postprocess_predictions as postp
from orthoseg.helpers.progress_helper import ProgressHelper
from orthoseg.util import general_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def predict_dir(
        model: kr_models.Model,
        input_image_dir: Path,
        output_image_dir: Path,
        output_vector_path: Optional[Path],         
        classes: list,
        prediction_cleanup_params = None,
        border_pixels_to_ignore: int = 0,
        min_pixelvalue_for_save: int = 127,
        projection_if_missing: str = None,
        input_mask_dir: Optional[Path] = None,
        batch_size: int = 16,                
        evaluate_mode: bool = False,
        cancel_filepath: Optional[Path] = None,
        force: bool = False):
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
        prediction_cleanup_params (dict, optional): parameters to specify which
            cleanups of the prediction need to be executed. 
            Default is None. 
        border_pixels_to_ignore: because the segmentation at the borders of the
            input images images is not as good, you can specify that x 
            pixels need to be ignored
        min_pixelvalue_for_save: the minimum pixel value that should be 
                present in the prediction to save the prediction
        input_mask_dir: optional dir where the mask images are located
        projection_if_missing: Normally the projection should be in the raster file. If it 
                    is not, you can explicitly specify one.
        batch_size: batch size to use while predicting. This must be choosen 
                depending on the neural network architecture and available 
                memory on you GPU.
        evaluate_mode: True to run in evaluate mode
        cancel_filepath: If the file in this path exists, processing stops asap
        force: False to skip images that already have a prediction, true to
               ignore existing predictions and overwrite them
    """
    
    # Init
    if not input_image_dir.exists():
        logger.warn(f"In predict_dir, but input_image_dir doesn't exist, so return: {input_image_dir}")
        return
    if output_vector_path is not None and output_vector_path.exists():
        logger.warn(f"In predict_dir, output file exists already, so return: {output_vector_path}")
        return
    tmp_dir = Path(tempfile.gettempdir()) / Path(__file__).stem

    logger.info(f"Start predict for input_image_dir: {input_image_dir}")

    # Eager and not eager prediction seems +- the same performance-wise
    #model.run_eagerly = False
    
    # If we are using evaluate mode, change the output dir...
    if evaluate_mode:
        output_image_dir = Path(str(output_image_dir) + '_eval')

    # Create the output dir's if they don't exist yet...
    for dir in [output_image_dir, tmp_dir]:
        if not dir.exists():
            dir.mkdir()

    # Write prediction config used, so it can be used for postprocessing
    prediction_config_path = output_image_dir / "prediction_config.json"
    with open(prediction_config_path, 'w') as pred_conf_file:
        pred_conf = {}
        pred_conf['border_pixels_to_ignore'] = border_pixels_to_ignore
        pred_conf['classes'] = classes
        json.dump(pred_conf, pred_conf_file)

    # Get list of all image files to process...
    image_filepaths: List[Path] = []
    input_ext = ['.png', '.tif', '.jpg']
    for input_ext_cur in input_ext:
        image_filepaths.extend(input_image_dir.rglob('*' + input_ext_cur))
    nb_images = len(image_filepaths)
    logger.info(f"Found {nb_images} {input_ext} images to predict on in {input_image_dir}")
    
    # If force is false, get list of all existing predictions
    # Getting the list once is way faster than checking file per file later on!
    images_done_log_filepath = output_image_dir / 'images_done.txt'
    images_error_log_filepath = output_image_dir / 'images_error.txt'
    image_done_filenames = set()
    if force is False:
        # First read the listing files if they exists
        if images_done_log_filepath.exists():
            with images_done_log_filepath.open() as f:
                for filename in f:
                    image_done_filenames.add(filename.rstrip())
        if images_error_log_filepath.exists():                    
            with images_error_log_filepath.open() as f:
                for filename in f:
                    image_done_filenames.add(filename.rstrip())
        if len(image_done_filenames) > 0:
            logger.info(f"Found {len(image_done_filenames)} predicted images in {images_done_log_filepath} and {images_error_log_filepath}, they will be skipped")

    # First write output to tmp files, so we know if the process completed correctly or not...
    pred_tmp_output_path = None
    if output_vector_path is not None:
        pred_tmp_output_path = output_image_dir / f"{output_vector_path.stem}_tmp.gpkg"
        pred_tmp_output_lock_path = Path(f"{str(pred_tmp_output_path)}.lock")
        # if lock file exists, remove it:
        if pred_tmp_output_lock_path.exists():
            pred_tmp_output_lock_path.unlink()

    # Loop through all files to process them...
    curr_batch_image_infos = []
    nb_to_process = nb_images
    nb_processed = 0
    progress = None
    image_filepaths_sorted = sorted(image_filepaths)
    future_to_input_path = {}
    nb_parallel_postprocess = multiprocessing.cpu_count()

    def init_postprocess_worker():
        # We don't want the postprocess workers to block the entire system, 
        # so make them a bit nicer 
        general_util.setprocessnice(15)

    with futures.ThreadPoolExecutor(batch_size) as pool, \
         futures.ProcessPoolExecutor(
                nb_parallel_postprocess, 
                initializer=init_postprocess_worker()) as postprocess_pool:
        
        for image_id, image_filepath in enumerate(image_filepaths_sorted):
            
            # If the cancel file exists, stop processing...
            if cancel_filepath is not None and cancel_filepath.exists():
                print()
                logger.info(f"Cancel file found, so stop: {cancel_filepath}")
                break

            # If force is false and prediction exists... skip
            if force is False:
                if image_filepath.name in image_done_filenames:
                    logger.debug(f"Predict for image has already been done before and force is False, so skip: {image_filepath.name}")
                    nb_to_process -= 1
                    continue
                    
            # Prepare the filepath for the output
            output_suffix = '.tif'
            if evaluate_mode:
                # In evaluate mode, put everyting in output base dir for easier 
                # comparison
                output_image_pred_dir = output_image_dir

                # Prepare complete filepath for image prediction
                output_image_pred_path = output_image_dir / image_filepath.stem
            else:
                # If saving predictions to images for real, keep hierarchic structure if present
                tmp_output_filepath = Path(str(image_filepath).replace(str(input_image_dir), str(output_image_dir)))
                output_image_pred_dir = tmp_output_filepath.parent
                output_image_pred_path = output_image_pred_dir / f"{image_filepath.stem}_pred{output_suffix}"
               
            nb_processed += 1
            
            # Append the image info to the batch array so they can be treated in 
            # bulk if the batch size is reached
            logger.debug(f"Start predict for image {image_filepath}")                   
            curr_batch_image_infos.append({'input_image_filepath': image_filepath,
                                           'output_pred_filepath': output_image_pred_path,
                                           'output_image_pred_dir': output_image_pred_dir})
            
            # If the batch size is reached or we are at the last images
            nb_images_in_batch = len(curr_batch_image_infos)
            curr_batch_image_infos_ext = []
            if(nb_images_in_batch == batch_size or image_id == (nb_images-1)):
                
                # Init progress only at 2nd batch, as the first is very slow
                if progress is None and nb_processed > batch_size:
                    progress = ProgressHelper(
                            message=f"predict to {output_image_dir.parent.name}/{output_image_dir.name}",
                            nb_steps_total=nb_to_process, 
                            nb_steps_done=batch_size)
                
                ## Read all input images for the batch (in parallel) ##
                perf_start_time = datetime.datetime.now()
                logger.debug(f"Start reading input {nb_images_in_batch} images")
                
                # Put arguments to pass to map in lists
                read_arg_filepaths = [info.get('input_image_filepath') for info in curr_batch_image_infos]
                read_arg_projections = [projection_if_missing for info in curr_batch_image_infos]
                
                # Exec read in parallel and extract result
                read_results = pool.map(
                        read_image,            # Function 
                        read_arg_filepaths,    # Arg 1-list
                        read_arg_projections)  # Arg 2-list
                curr_batch_image_list = [] 
                for batch_image_id, read_result in enumerate(read_results):
                    curr_batch_image_infos_ext.append(
                            {'input_image_filepath': curr_batch_image_infos[batch_image_id]['input_image_filepath'],
                             'output_pred_filepath': curr_batch_image_infos[batch_image_id]['output_pred_filepath'],
                             'output_image_pred_dir': curr_batch_image_infos[batch_image_id]['output_image_pred_dir'],
                             'image_crs': read_result['image_crs'],
                             'image_transform': read_result['image_transform']})
                    curr_batch_image_list.append(read_result['image_data']) 
                perfinfo = f"read took {datetime.datetime.now()-perf_start_time}"
                
                ## Predict! ##
                logger.debug(f"Start prediction for {nb_images_in_batch} images")
                perf_start_time = datetime.datetime.now()
                curr_batch_image_arr = np.stack(curr_batch_image_list)
                curr_batch_image_pred_arr = model.predict_on_batch(curr_batch_image_arr)
                
                # In tf 2.1 a tf.tensor object is returned, but we want an ndarray
                if type(curr_batch_image_pred_arr) is tf.Tensor:
                    curr_batch_image_pred_arr = np.array(curr_batch_image_pred_arr.numpy())
                else:
                    curr_batch_image_pred_arr = np.array(curr_batch_image_pred_arr)
                perfinfo += f", predict took {datetime.datetime.now()-perf_start_time}"

                ## Save predictions ##
                # Remark: trying to parallelize this doesn't seem to help at all!
                logger.debug("Start post-processing")    
                perf_start_time = datetime.datetime.now()
                for batch_image_id, image_info in enumerate(curr_batch_image_infos_ext):
                    try:
                        # Saving the predictions as images at the moment only used 
                        # for evaluate mode...
                        # TODO: would ideally be moved to the background 
                        # processing as well to simplify code here...
                        if evaluate_mode is True:
                            postp.clean_and_save_prediction(
                                    image_image_filepath=image_info['input_image_filepath'],
                                    image_crs=image_info['image_crs'],
                                    image_transform=image_info['image_transform'],
                                    image_pred_arr=curr_batch_image_pred_arr[batch_image_id],
                                    output_dir=image_info['output_image_pred_dir'],
                                    input_image_dir=input_image_dir,
                                    input_mask_dir=input_mask_dir,
                                    border_pixels_to_ignore=border_pixels_to_ignore,
                                    min_pixelvalue_for_save=min_pixelvalue_for_save,
                                    evaluate_mode=evaluate_mode,
                                    classes=classes,
                                    force=force)
                            
                            # Write filepath to file with files that are done
                            with images_done_log_filepath.open('a+') as image_donelog_file:
                                image_donelog_file.write(f"{image_info['input_image_filepath'].name}\n")
                        
                        # If not in evaluate mode... save to vector in background
                        elif output_vector_path is not None:
                            # Prepare prediction array...
                            #   - 1st channel is background, don't postprocess
                            #   - convert to uint8 to reduce pickle size/time    
                            image_pred_arr_uint8 = ((curr_batch_image_pred_arr[batch_image_id,:,:,1:]) * 255).astype(np.uint8)
                            future = postprocess_pool.submit(
                                    postp.polygonize_pred_multiclass_to_file,
                                    image_pred_arr_uint8,
                                    image_info['image_crs'],
                                    image_info['image_transform'],
                                    min_pixelvalue_for_save,
                                    classes[1:],               # The first class is the background so skip that
                                    pred_tmp_output_path,
                                    prediction_cleanup_params,
                                    border_pixels_to_ignore)
                            future_to_input_path[future] = image_info['input_image_filepath']

                    except:
                        logger.exception(f"Error postprocessing pred for {image_info['input_image_filepath']}")
    
                        # Write filepath to file with errors
                        with images_error_log_filepath.open('a+') as image_errorlog_file:
                            image_errorlog_file.write(image_info['input_image_filepath'].name + '\n')
                
                # Poll for completed postprocessings 
                sleeping_logged = False 
                while True:
                    
                    # If not at last file, get results from all futures that are 
                    # done, if at last file, wait till all are done
                    if image_id < (nb_images-1):
                        futures_done = [future for future in future_to_input_path if future.done() is True]
                    else:
                        logger.info("Wait for last batch")
                        futures_done = futures.wait(future_to_input_path).done
                    for future in futures_done:
                        # Get the result from the polygonization
                        try:
                            # Get the result (= exception when something went wrong)
                            future.result()

                            # Write filepath to file with files that are done
                            with images_done_log_filepath.open('a+') as image_donelog_file:
                                image_donelog_file.write(future_to_input_path[future].name + '\n')
                        except Exception as ex:
                            # Write filepath to file with errors
                            logger.exception(f"Error postprocessing result for {future_to_input_path[future].name}")
                            with images_error_log_filepath.open('a+') as image_errorlog_file:
                                image_errorlog_file.write(future_to_input_path[future].name + '\n')
                        finally:
                            # Remove from queue...
                            del future_to_input_path[future]

                    # Wait till number below thresshold to evade huge waiting 
                    # list (and memory issues)
                    if len(future_to_input_path) > nb_parallel_postprocess*2:
                        if sleeping_logged is False:
                            logger.info(f"Postprocessing seems to take longer than prediction, so wait for it to catch up")
                            sleeping_logged = True
                    else:
                        # No need to wait (anymore)...
                        if sleeping_logged is True:
                            logger.info(f"Waited enough for postprocessing to catch up, so continue predicting") 
                        break

                perfinfo += f", clean+save took {datetime.datetime.now()-perf_start_time}"
                logger.debug(perfinfo)                          

                ## Log the progress and prediction speed ##
                if progress is not None:
                    progress.step(nb_steps=batch_size)

                # Reset variable for next batch
                curr_batch_image_infos = []

            # If we are ready, rename to real output file
            if(image_id == (nb_images-1) 
               and output_vector_path is not None 
               and pred_tmp_output_path is not None and pred_tmp_output_path.exists()):
                geofile.move(pred_tmp_output_path, output_vector_path)
                geofile.rename_layer(output_vector_path, output_vector_path.stem)
                shutil.rmtree(output_image_dir)

def read_image(
        image_filepath: Path,
        projection_if_missing: str = None) -> dict:

    # Read input file
    # Because sometimes a read seems to fail, retry up to 3 times...
    retry_count = 0
    while True:
        try:
            with rio.open(str(image_filepath)) as image_ds:
                # Read geo info
                image_crs = image_ds.profile['crs']
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
                message = f"STOP: Read failed {retry_count} times for {image_filepath}, with exception: {ex}"
                logger.error(message)
                raise Exception(message)
        
    # The read was successfull, now check if there was a projection in the 
    # file and/or if one was provided
    if image_crs is None:
        if projection_if_missing is not None:
            image_crs = rio_crs.CRS.from_string(projection_if_missing)
        else:
            message = f"Image doesn't contain projection and no projection_if_missing provided!: {image_filepath}"
            logger.error(message)
            raise Exception(message)
        
    # Now return the result
    result = {'image_data': image_data,
              'image_crs': image_crs,
              'image_transform': image_transform,
              'image_filepath': image_filepath}
    return result
        
# If the script is ran directly...
if __name__ == '__main__':
    raise Exception("Not implemented")
    
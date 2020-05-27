# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images.
"""

from concurrent import futures
import datetime
import logging
from pathlib import Path
from typing import List, Optional

from tensorflow import keras as kr
import numpy as np
import rasterio as rio
import rasterio.plot as rio_plot

import orthoseg.lib.postprocess_predictions as postp

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
        model: kr.models.Model,
        input_image_dir: Path,
        output_base_dir: Path,         
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
        input_image_dir: dir where the input images are located
        output_base_dir: dir where the output will be put
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
    
    # Check if input params are ok
    if not input_image_dir.exists():
        logger.warn(f"In predict_dir, but input_image_dir doesn't exist, so return: {input_image_dir}")
        return

    logger.info(f"Start predict for input_image_dir: {input_image_dir}")

    # Eager and not eager prediction seems +- the same performance-wise
    #model.run_eagerly = False
    
    # If we are using evaluate mode, change the output dir...
    if evaluate_mode:
        output_base_dir = Path(str(output_base_dir) + '_eval')
        
    # Create the output dir's if they don't exist yet...
    for dir in [output_base_dir]:
        if not dir.exists():
            dir.mkdir()

    # Get list of all image files to process...
    image_filepaths: List[Path] = []
    input_ext = ['.png', '.tif', '.jpg']
    for input_ext_cur in input_ext:
        image_filepaths.extend(input_image_dir.rglob('*' + input_ext_cur))
    nb_todo = len(image_filepaths)
    logger.info(f"Found {nb_todo} {input_ext} images to predict on in {input_image_dir}")
    
    # If force is false, get list of all existing predictions
    # Getting the list once is way faster than checking file per file later on!
    images_done_log_filepath = output_base_dir / 'images_done.txt'
    images_error_log_filepath = output_base_dir / 'images_error.txt'
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

    # Loop through all files to process them...
    curr_batch_image_infos = []
    nb_processed = 0
    start_time = None
    start_time_batch_read = None
    progress_log_time = None
    image_filepaths_sorted = sorted(image_filepaths)
    with images_error_log_filepath.open('a+') as image_errorlog_file, \
         images_done_log_filepath.open('a+') as image_donelog_file, \
         futures.ThreadPoolExecutor(batch_size) as pool:
        
        for i, image_filepath in enumerate(image_filepaths_sorted):
            
            # If the cancel file exists, stop processing...
            if cancel_filepath is not None and cancel_filepath.exists():
                logger.info(f"Cancel file found, so stop: {cancel_filepath}")
                break

            # If force is false and prediction exists... skip
            if force is False:
               if image_filepath.name in image_done_filenames:
                   logger.debug(f"Predict for image has already been done before and force is False, so skip: {image_filepath.name}")
                   continue
                    
            # Prepare the filepath for the output
            if evaluate_mode:
                # In evaluate mode, put everyting in output base dir for easier 
                # comparison
                tmp_output_filepath = output_base_dir / image_filepath.stem
            else:
                tmp_output_filepath = Path(str(image_filepath).replace(str(input_image_dir), str(output_base_dir)))
                tmp_output_filepath = tmp_output_filepath.parent / tmp_output_filepath.stem
            output_dir = tmp_output_filepath.parent
            output_pred_filepath = output_dir / f"{tmp_output_filepath.stem}_pred.tif"
            
            # Init start time after first batch is ready, as it is really slow
            if start_time is None and start_time_batch_read is not None:
                start_time = datetime.datetime.now()
            nb_processed += 1
            
            # Append the image info to the batch array so they can be treated in 
            # bulk if the batch size is reached
            logger.debug(f"Start predict for image {image_filepath}")                   
            curr_batch_image_infos.append({'input_image_filepath': image_filepath,
                                           'output_pred_filepath': output_pred_filepath,
                                           'output_dir': output_dir})
            
            # If the batch size is reached or we are at the last images
            nb_images_in_batch = len(curr_batch_image_infos)
            curr_batch_image_infos_ext = []
            if(nb_images_in_batch == batch_size or i == (nb_todo-1)):
                start_time_batch_read = datetime.datetime.now()
                
                # Read all input images for the batch (in parallel)
                logger.debug(f"Start reading input {nb_images_in_batch} images")
                
                # Put arguments to pass to map in lists
                read_arg_filepaths = [info.get('input_image_filepath') for info in curr_batch_image_infos]
                read_arg_projections = [projection_if_missing for info in curr_batch_image_infos]
                
                # Exec read in parallel and extract result
                read_results = pool.map(read_image,            # Function 
                                        read_arg_filepaths,    # Arg 1-list
                                        read_arg_projections)  # Arg 2-list
                curr_batch_image_list = [] 
                for j, read_result in enumerate(read_results):
                    curr_batch_image_infos_ext.append(
                            {'input_image_filepath': curr_batch_image_infos[j]['input_image_filepath'],
                             'output_pred_filepath': curr_batch_image_infos[j]['output_pred_filepath'],
                             'output_dir': curr_batch_image_infos[j]['output_dir'],
                             'image_crs': read_result['image_crs'],
                             'image_transform': read_result['image_transform']})
                    curr_batch_image_list.append(read_result['image_data']) 

                # Predict!
                curr_batch_image_arr = np.stack(curr_batch_image_list)
                logger.debug(f"Start prediction for {nb_images_in_batch} images")
                #images = [info.get('image_data') for info in curr_batch_image_infos_ext]
                #image_array = np.array(images, copy=False)
                curr_batch_image_pred_arr = model.predict_on_batch(curr_batch_image_arr)
                
                # In tf 2.1 a tf.tensor object is returned, but we want an ndarray
                if type(curr_batch_image_pred_arr) is not np.ndarray:
                    curr_batch_image_pred_arr = curr_batch_image_pred_arr.numpy()
                
                # Postprocess predictions
                # Remark: trying to parallelize this doesn't seem to help at all!
                logger.debug("Start post-processing")    
                for j, image_info in enumerate(curr_batch_image_infos_ext):    
                    try:
                        clean_and_save_prediction(
                                image_image_filepath=image_info['input_image_filepath'],
                                image_crs=image_info['image_crs'],
                                image_transform=image_info['image_transform'],
                                image_pred_arr=curr_batch_image_pred_arr[j],
                                output_dir=image_info['output_dir'],
                                input_image_dir=input_image_dir,
                                input_mask_dir=input_mask_dir,
                                border_pixels_to_ignore=border_pixels_to_ignore,
                                min_pixelvalue_for_save=min_pixelvalue_for_save,
                                evaluate_mode=evaluate_mode,
                                force=force)

                        # Write line to file with done files...
                        image_donelog_file.write(image_info['input_image_filepath'].name + '\n')
                    except:
                        logger.exception(f"Error postprocessing pred for {image_info['input_image_filepath']}")
    
                        # Write line to file with error files...
                        image_errorlog_file.write(image_info['input_image_filepath'].name + '\n')
                logger.debug("Post-processing ready")

                # Log the progress and prediction speed
                if start_time is not None:
                    time_passed_s = (datetime.datetime.now()-start_time).total_seconds()
                    time_passed_lastbatch_s = (datetime.datetime.now()-start_time_batch_read).total_seconds()
                    if time_passed_s > 0 and time_passed_lastbatch_s > 0:
                        nb_per_hour = (nb_processed/time_passed_s) * 3600
                        nb_per_hour_lastbatch = (nb_images_in_batch/time_passed_lastbatch_s) * 3600
                        hours_to_go = (int)((nb_todo - i)/nb_per_hour)
                        min_to_go = (int)((((nb_todo - i)/nb_per_hour)%1)*60)
                        message = f"{hours_to_go:3d}:{min_to_go:2d} left for {nb_todo-i} todo at {nb_per_hour:0.0f}/h ({nb_per_hour_lastbatch:0.0f}/h last batch) in ...{str(input_image_dir)[-30:]}"
                        print(f"\r{message}", end='', flush=True)

                        # Once every 15 minutes, log progress to log file
                        time_passed_progress_log_s = 0
                        if progress_log_time is not None:
                            time_passed_progress_log_s = (datetime.datetime.now()-progress_log_time).total_seconds()
                        if progress_log_time is None or time_passed_progress_log_s > (15*60):
                            logger.info(message)
                            progress_log_time = datetime.datetime.now()
                
                # Reset variable for next batch
                curr_batch_image_infos = []
    
    # Loop through all files to process them...
    """
    # QUEUE-LIKE IMPLEMENTATION -> MORE COMPLICATED + SLOWER!!!
    image_filepaths_sorted = sorted(image_filepaths)
    nb_processed = 0
    nb_read_busy = 0
    last_image_started = -1
    images_busy = {}
    start_time = None
    prev_start_time_batch_predict = None
    with open(images_error_log_filepath, "a+") as image_errorlog_file, \
         open(images_done_log_filepath, "a+") as image_donelog_file, \
         futures.ProcessPoolExecutor(batch_size) as read_pool:
         #futures.ProcessPoolExecutor(batch_size) as postpredict_save_pool:
        while True:
    
            # If the maximum number of images to treat simultaenously isn't met yet, start processing another
            if(len(images_busy) <= (batch_size * 8)
                and last_image_started < (nb_todo-1)):

                # Get the next image to start processing
                last_image_started += 1
                image_filepath = image_filepaths_sorted[last_image_started]
                
                # If force is false and prediction exists... skip
                if force is False:
                    filename = os.path.basename(image_filepath)
                    if filename in image_done_filenames:
                        logger.debug(f"Predict for image has already been done before and force is False, so skip: {filename}")
                        continue

                # Prepare the filepath for the output
                image_filepath_noext, _ = os.path.splitext(image_filepath)
                if evaluate_mode:
                    # In evaluate mode, put everyting in output base dir for easier 
                    # comparison
                    _, image_filename_noext = os.path.split(image_filepath_noext)
                    tmp_output_filepath = os.path.join(output_base_dir, image_filename_noext)
                else:
                    tmp_output_filepath = image_filepath_noext.replace(input_image_dir, output_base_dir)
                output_pred_filepath = f"{tmp_output_filepath}_pred.tif"       
                output_dir, _ = os.path.split(output_pred_filepath)
                
                # Add image to dict with the images we are busy with...
                images_busy[image_filepath] = {
                        'image_filepath': image_filepath,
                        'output_pred_filepath': output_pred_filepath,
                        'output_dir': output_dir,
                        'read_future': None,
                        'read_ready': False,
                        'predict_ready': False,
                        'postprocess_save_future': None,
                        'postprocess_save_ready': False}

                # Start reading the image
                logger.debug(f"Start read for image {image_filepath}")
                nb_read_busy += 1
                read_future = read_pool.submit(read_image,              # Function 
                                               image_filepath,          # Arg 1
                                               projection_if_missing)
                images_busy[image_filepath]['read_future'] = read_future

                images_ready_to_predict = [image_filepath for image_filepath in images_busy if images_busy[image_filepath]['read_ready'] == True]
                if (len(images_ready_to_predict) + nb_read_busy) < batch_size * 3:
                    continue
            else:
                logger.info(f"Max number of images in queue: {len(images_busy)}")
                # Sleep a little bit...
                # TODO: check if this is the best spot to sleep
                #time.sleep(0.001)

            # Check if there are reads ready...
            for image_busy_filepath in images_busy:
                # If read result hasn't been collected yet, check if it is still busy
                read_future = images_busy[image_busy_filepath]['read_future']
                if read_future is not None:
                    # If still running, not ready yet
                    if read_future.running() is True:
                        continue
                    else:
                        try:
                            # Get the results from the read
                            read_result = read_future.result()
                            nb_read_busy -= 1
                            images_busy[image_busy_filepath]['image_crs'] = read_result['image_crs']
                            images_busy[image_busy_filepath]['image_transform'] = read_result['image_transform']
                            images_busy[image_busy_filepath]['image_data'] = read_result['image_data']  
                        except:
                            logger.exception(f"Error postprocessing pred for {image_busy_filepath}")
                            # Write line to file with error files...
                            image_errorlog_file.write(os.path.basename(image_busy_filepath) + '\n')
                        finally:
                            # Mark read as completed
                            images_busy[image_busy_filepath]['read_ready'] = True
                            images_busy[image_busy_filepath]['read_future'] = None

            '''
            nb_images_in_batch = len(curr_batch_image_infos)
            curr_batch_image_infos_ext = []
            if(nb_images_in_batch == batch_size or i == (nb_todo-1)):
            '''

            # If enough images have been read to predict, predict...
            images_ready_to_predict = [image_filepath for image_filepath in images_busy if images_busy[image_filepath]['read_ready'] == True]
            nb_images_ready_to_predict = len(images_ready_to_predict)
            logger.info(f"nb_images_ready_to_predict: {nb_images_ready_to_predict}") 
            start_time_batch_predict = None
            if(nb_images_ready_to_predict >= batch_size 
                or (last_image_started >= (nb_todo-1) and nb_images_ready_to_predict > 0)):

                # Get the first batch_size images to predict them...
                start_time_batch_predict = datetime.datetime.now()
                images_to_predict = images_ready_to_predict[:batch_size]
                nb_images_to_predict = len(images_to_predict)
                
                # First batch is very slow, so slow down init of start time 
                if start_time is None and nb_processed > 0:
                    start_time = datetime.datetime.now()

                # Predict!
                logger.info(f"Start prediction for {nb_images_to_predict} images")
                image_datas = [images_busy[image_path].get('image_data') for image_path in images_to_predict]
                image_pred_arr = model.predict_on_batch(np.asarray(image_datas))

                # Copy the results to the images_busy dict
                for j, image_path in enumerate(images_to_predict):
                    images_busy[image_path]['image_pred_data'] = image_pred_arr[j]
                    images_busy[image_path]['predict_ready'] = True
                            
            # Check if there are images that still need postprocessing, or are ready postprocessing
            for image_busy_filepath in images_busy:
                # If prediction isn't ready yet or postprocessing is already ready... continue 
                if(images_busy[image_busy_filepath]['predict_ready'] is False
                    or images_busy[image_busy_filepath]['postprocess_save_ready'] is True):
                    continue

                # Image is ready to postprocess, if its postprocess isn't busy yet, start it.
                # Remark: saving synchronously is apparently faster than using threads or processes
                clean_and_save_prediction(
                        image_image_filepath=image_busy_filepath,                                   
                        image_crs=images_busy[image_busy_filepath]['image_crs'],         
                        image_transform=images_busy[image_busy_filepath]['image_transform'],   
                        output_dir=images_busy[image_busy_filepath]['output_dir'],
                        image_pred_arr=images_busy[image_busy_filepath]['image_pred_data'],
                        input_image_dir=None,                                                
                        input_mask_dir=input_mask_dir,
                        border_pixels_to_ignore=border_pixels_to_ignore,
                        evaluate_mode=evaluate_mode,
                        force=force)
                images_busy[image_busy_filepath]['postprocess_save_ready'] = True
                images_busy[image_busy_filepath]['postprocess_save_future'] = None
                nb_processed += 1
                # Write line to file with done files...
                image_donelog_file.write(os.path.basename(image_busy_filepath) + '\n')

                '''
                # Assync saving is slower!!!
                postprocess_save_future = images_busy[image_busy_filepath]['postprocess_save_future']
                if postprocess_save_future is None:
                    
                    # First clean result
                    image_pred_uint8_cleaned = clean_prediction(
                            image_pred_arr=images_busy[image_busy_filepath]['image_pred_data'], 
                            border_pixels_to_ignore=border_pixels_to_ignore)
                    if image_pred_uint8_cleaned is None:
                        # Image is already treated...
                        images_busy[image_busy_filepath]['postprocess_save_ready'] = True
                        images_busy[image_busy_filepath]['postprocess_save_future'] = None
                        nb_processed += 1
                        # Write line to file with done files...
                        image_donelog_file.write(os.path.basename(image_busy_filepath) + '\n')
                    else:
                        postprocess_save_future = postpredict_save_pool.submit(
                                save_prediction,                                        # Function
                                image_busy_filepath,                                    # Arg 1
                                images_busy[image_busy_filepath]['image_crs'],          # Arg 2
                                images_busy[image_busy_filepath]['image_transform'],    # ...
                                images_busy[image_busy_filepath]['output_dir'],
                                image_pred_uint8_cleaned, #images_busy[image_busy_filepath]['image_pred_data'],
                                None,                                                
                                input_mask_dir,
                                border_pixels_to_ignore,
                                evaluate_mode,
                                force)
                        images_busy[image_busy_filepath]['postprocess_save_future'] = postprocess_save_future 
                else:
                    # Postprocess is still running, so skip this image
                    if postprocess_save_future.running() is True:
                        continue
                    else:
                        # Postprocess is finished, try to get the results 
                        try:
                            _ = postprocess_save_future.result()
                            # Write line to file with done files...
                            image_donelog_file.write(os.path.basename(image_busy_filepath) + '\n')
                        except:
                            logger.exception(f"Error postprocessing pred for {image_busy_filepath}")
                            # Write line to file with error files...
                            image_errorlog_file.write(os.path.basename(image_busy_filepath) + '\n')
                        finally:
                            nb_processed += 1
                            images_busy[image_busy_filepath]['postprocess_save_ready'] = True
                            images_busy[image_busy_filepath]['postprocess_save_future'] = None
                '''

            # Remove images that are ready from images_busy
            images_ready = [image_filepath for image_filepath in images_busy if images_busy[image_filepath]['postprocess_save_ready'] == True]
            for image_ready in images_ready:
                del images_busy[image_ready]
        
            # Log the progress and prediction speed... if there was a prediction this loop...
            if start_time_batch_predict is not None and start_time is not None:
                time_passed_s = (datetime.datetime.now()-start_time).total_seconds()
                if prev_start_time_batch_predict is not None:
                    time_passed_lastbatch_s = (start_time_batch_predict-prev_start_time_batch_predict).total_seconds()
                else:
                    time_passed_lastbatch_s = (datetime.datetime.now()-start_time_batch_predict).total_seconds()
                if nb_processed > 0 and time_passed_s > 0 and time_passed_lastbatch_s > 0:
                    nb_per_hour = (nb_processed/time_passed_s) * 3600
                    nb_per_hour_lastbatch = (batch_size/time_passed_lastbatch_s) * 3600
                    hours_to_go = (int)((nb_todo-last_image_started)/nb_per_hour)
                    min_to_go = (int)((((nb_todo-last_image_started)/nb_per_hour)%1)*60)
                    print(f"\r{hours_to_go}:{min_to_go} left for {nb_todo-last_image_started} todo at {nb_per_hour:0.0f}/h ({nb_per_hour_lastbatch:0.0f}/h last batch) in ...{input_image_dir[-30:]}",
                            end='', flush=True)
                prev_start_time_batch_predict = start_time_batch_predict

            # If we are ready... stop
            if len(images_busy) == 0 and last_image_started >= (nb_todo-1):
                break
    """

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
                
                # Read pixelsn change from (channels, width, height) to 
                # (width, height, channels) and normalize to values between 0 and 1
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
            image_crs = rio.crs.CRS.from_string(projection_if_missing)
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

def clean_and_save_prediction(
        image_image_filepath: Path,
        image_crs: str,
        image_transform: str,
        output_dir: Path,
        image_pred_arr: np.array,
        input_image_dir: Optional[Path] = None,
        input_mask_dir: Optional[Path] = None,
        border_pixels_to_ignore: int = 0,
        min_pixelvalue_for_save: int = 127,
        evaluate_mode: bool = False,
        force: bool = False) -> bool:

    # If nb. channels in prediction > 1, skip the first as it is the background
    image_pred_shape = image_pred_arr.shape
    nb_channels = image_pred_shape[2]
    if nb_channels > 1:
        channel_start = 1
    else:
        channel_start = 0

    for channel in range(channel_start, nb_channels):
        # TODO add proper support for multiple channels!
        image_pred_curr_arr = image_pred_arr[:,:,channel]

        # Clean prediction
        image_pred_uint8_cleaned_curr = clean_prediction(
                image_pred_arr=image_pred_curr_arr, 
                border_pixels_to_ignore=border_pixels_to_ignore)
        
        # If the cleaned result contains useful values... save
        if(min_pixelvalue_for_save == 0 
           or np.any(image_pred_uint8_cleaned_curr >= min_pixelvalue_for_save)):
            save_prediction(
                    image_image_filepath=image_image_filepath,
                    image_crs=image_crs,
                    image_transform=image_transform,
                    output_dir=output_dir,
                    image_pred_uint8_cleaned=image_pred_uint8_cleaned_curr,
                    input_image_dir=input_image_dir,
                    input_mask_dir=input_mask_dir,
                    output_suffix=f"_{channel}",
                    border_pixels_to_ignore=border_pixels_to_ignore,
                    evaluate_mode=evaluate_mode,
                    force=force)

    return True

def save_prediction(
        image_image_filepath: Path,
        image_crs: str,
        image_transform: str,
        output_dir: Path,
        image_pred_uint8_cleaned: np.array,
        input_image_dir: Optional[Path],
        input_mask_dir: Optional[Path],
        output_suffix: str,
        border_pixels_to_ignore: int,
        evaluate_mode: bool = False,
        force: bool = False) -> bool:

    # Save prediction
    image_pred_filepath = save_prediction_uint8(
            image_filepath=image_image_filepath,
            image_pred_uint8_cleaned=image_pred_uint8_cleaned,
            image_crs=image_crs,
            image_transform=image_transform,
            output_dir=output_dir,
            output_suffix=output_suffix,
            force=force)

    # Postprocess for evaluation
    if evaluate_mode is True:
        # Create binary version and postprocess
        image_pred_uint8_cleaned_bin = postp.to_binary_uint8(image_pred_uint8_cleaned, 125)                       
        postp.postprocess_for_evaluation(
                image_filepath=image_image_filepath,
                image_crs=image_crs,
                image_transform=image_transform,
                image_pred_filepath=image_pred_filepath,
                image_pred_uint8_cleaned_bin=image_pred_uint8_cleaned_bin,
                output_dir=output_dir,
                output_suffix=output_suffix,
                input_image_dir=input_image_dir,
                input_mask_dir=input_mask_dir,
                border_pixels_to_ignore=border_pixels_to_ignore,
                force=force)
    return True

def clean_prediction(
        image_pred_arr: np.array,
        border_pixels_to_ignore: int = 0,
        output_color_depth: str = 'binary') -> np.array:
    """
    Cleans a prediction result and returns a cleaned, uint8 array.
    
    Args:
        image_pred_arr (np.array): The prediction as returned by keras.
        border_pixels_to_ignore (int, optional): Border pixels to ignore. Defaults to 0.
        output_color_depth (str, optional): Color depth desired. Defaults to '2'.
            * binary: 0 or 255
            * decimal: ten different values: 0, 25, 50,... 255
            * full: 256 different values
    
    Returns:
        np.array: The cleaned result.
    """

    # Input should be float32
    if image_pred_arr.dtype != np.float32:
        raise Exception(f"image prediction is of the wrong type: {image_pred_arr.dtype}") 
    
    # Reshape from 3 to 2 dims if necessary (width, height, nb_channels).
    # Check the number of channels of the output prediction
    image_pred_shape = image_pred_arr.shape
    if len(image_pred_shape) > 2:
        n_channels = image_pred_shape[2]
        if n_channels > 1:
            raise Exception("Invalid input, should be one channel!")
        # Reshape array from 3 dims (width, height, nb_channels) to 2.
        image_pred_uint8 = image_pred_arr.reshape((image_pred_shape[0], image_pred_shape[1]))   

    # Convert to uint8
    if output_color_depth == 'binary':
        image_pred_uint8 = (image_pred_arr * 255).astype(np.uint8)
        image_pred_uint8[image_pred_uint8 >= 127] = 255
        image_pred_uint8[image_pred_uint8 < 127] = 0
    elif output_color_depth == 'full':
        image_pred_uint8 = (image_pred_arr * 255).astype(np.uint8)
    elif output_color_depth == 'decimal':
        image_pred_uint8 = (image_pred_arr * 10).astype(np.uint8)
        image_pred_uint8 = image_pred_uint8 * 25
    else:
        raise Exception(f"Unsupported output_color_depth: {output_color_depth}")
    
    # Make the pixels at the borders of the prediction black so they are ignored
    image_pred_uint8_cropped = image_pred_uint8
    if border_pixels_to_ignore and border_pixels_to_ignore > 0:
        image_pred_uint8_cropped[0:border_pixels_to_ignore,:] = 0    # Left border
        image_pred_uint8_cropped[-border_pixels_to_ignore:,:] = 0    # Right border
        image_pred_uint8_cropped[:,0:border_pixels_to_ignore] = 0    # Top border
        image_pred_uint8_cropped[:,-border_pixels_to_ignore:] = 0    # Bottom border
    
    return image_pred_uint8_cropped

def save_prediction_uint8(
        image_filepath: Path,
        image_pred_uint8_cleaned: np.array,
        image_crs: str,
        image_transform: str,
        output_dir: Path,
        output_suffix: str = '',
        border_pixels_to_ignore: int = None,
        force: bool = False) -> Path:

    ##### Init #####
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
    with rio.open(str(output_filepath), 'w', driver='GTiff', tiled='no',
                  compress='lzw', predictor=2, num_threads=4,
                  height=image_height, width=image_width, 
                  count=1, dtype=rio.uint8, crs=image_crs, transform=image_transform) as dst:
        dst.write(image_pred_uint8_cleaned, 1)
    
    return output_filepath
        
# If the script is ran directly...
if __name__ == '__main__':
    raise Exception("Not implemented")
    
# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images

@author: Pieter Roggemans
"""

import logging
import os
import glob
import datetime
import math
import concurrent.futures as futures

import numpy as np
import pandas as pd
import keras as kr
import rasterio as rio
import rasterio.plot as rio_plot

import models.model_factory as mf
import models.model_helper as mh
import postprocess as postp

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train(traindata_dir: str,
          validationdata_dir: str,
          model_encoder: str,
          model_decoder: str,
          model_save_dir: str,
          model_save_base_filename: str,
          image_width: int = 512,
          image_height: int = 512,
          image_subdir: str = "image",
          mask_subdir: str = "mask",
          model_preload_filepath: str = None,
          batch_size: int = 32,
          nb_epoch: int = 100,
          augmented_subdir: str = None):
    """
    Create a new or load an existing neural network and train it using 
    data from the train and validation directories specified.
    
    The best models will be saved to model_save_dir. The filenames of the 
    models will be constructed like this:
    {model_save_base_filename}_{combined_acc}_{train_acc}_{validation_acc}_{epoch}
        * combined_acc: average of train_acc and validation_acc
        * train_acc: the jaccard coëficient of train dataset for the model
        * validation_acc: the jaccard coëficient of validation dataset 
    In the scripts, if the "best model" is mentioned, this is the one with the 
    highest "combined_acc".
    
    Args
        traindata_dir: dir where the train data is located
        validationdata_dir: dir where the validation data is located
        model_encoder: encoder of the neural network to use
        model_decoder: decoder of the neural network to use
        model_save_dir: dir where (intermediate) best models will be saved
        model_save_base_filename: base filename to use when saving models
        image_width: width the input images will be rescaled to for training
        image_height: height the input images will be rescaled to for training
        image_subdir: subdir where the images can be found in traindata_dir and validationdata_dir
        mask_subdir: subdir where the corresponding masks can be found in traindata_dir and validationdata_dir
        model_preload_filepath: filepath to the model to continue training on, 
                or None if you want to start from scratch
        batch_size: batch size to use while training. This must be 
                choosen depending on the neural network architecture
                and available memory on you GPU.
        nb_epoch: maximum number of epochs to train
    """
    
    # These are the augmentations that will be applied to the input training images/masks
    # Remark: fill_mode + cval are defined as they are so missing pixels after eg. rotation
    #         are filled with 0, and so the mask will take care that they are +- ignored.
    data_gen_train_args = dict(rotation_range=90.0,
                               fill_mode='constant',
                               cval=0,
                               rescale=1./255,
                               width_shift_range=0.05,
                               height_shift_range=0.05,
                               shear_range=0.0,
                               zoom_range=0.05,
                               horizontal_flip=True,
                               vertical_flip=True)

    # Create the train generator
    traindata_augmented_dir = None
    if augmented_subdir is not None:
        traindata_augmented_dir = os.path.join(traindata_dir, augmented_subdir)
        if not os.path.exists(traindata_augmented_dir):
            os.makedirs(traindata_augmented_dir)
            
    train_gen = create_train_generator(input_data_dir=traindata_dir,
                            image_subdir=image_subdir, mask_subdir=mask_subdir,
                            aug_dict=data_gen_train_args, batch_size=batch_size,
                            target_size=(image_width, image_height),
                            class_mode=None,
                            save_to_dir=traindata_augmented_dir)

    # If there is a validation data dir specified, create extra generator
    if validationdata_dir:
        data_gen_validation_args = dict(rescale=1./255)
        validation_gen = create_train_generator(input_data_dir=validationdata_dir,
                                image_subdir=image_subdir, mask_subdir=mask_subdir,
                                aug_dict=data_gen_validation_args, batch_size=batch_size,
                                target_size=(image_width, image_height),
                                class_mode=None,
                                save_to_dir=None)
    else:
        validation_gen = None

    # Get the max epoch number from the log file if it exists...
    start_epoch = 0
    start_learning_rate = 1e-4  # Best set to 0.0001 to start (1e-3 is not ok)
    csv_log_filepath = f"{model_save_dir}{os.sep}{model_save_base_filename}" + '_log.csv'
    if os.path.exists(csv_log_filepath):
        logger.info(f"train_log csv exists: {csv_log_filepath}")
        if not model_preload_filepath:
            message = f"STOP: log file exists but preload model file not specified!!!"
            logger.critical(message)
            raise Exception(message)
        
        train_log_csv = pd.read_csv(csv_log_filepath, sep=';')
        logger.debug(f"train_log csv contents:\n{train_log_csv}")
        start_epoch = train_log_csv['epoch'].max()
        start_learning_rate = train_log_csv['lr'].min()
    logger.info(f"start_epoch: {start_epoch}, start_learning_rate: {start_learning_rate}")
   
    # Create a model
    model_json_filename = f"{model_encoder}+{model_decoder}.json"
    model_json_filepath = os.path.join(model_save_dir, model_json_filename)
    if not model_preload_filepath:
        # If no existing model provided, create it from scratch
        # Get the model we want to use
        model = mf.get_model(encoder=model_encoder, decoder=model_decoder, 
                             n_channels=3, n_classes=1)
        
        # Save the model architecture to json if it doesn't exist yet
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        if not os.path.exists(model_json_filepath):
            with open(model_json_filepath, 'w') as dst:
                dst.write(model.to_json())
    else:
        # If a preload model is provided, load that if it exists...
        if not os.path.exists(model_preload_filepath):
            message = f"Error: preload model file doesn't exist: {model_preload_filepath}"
            logger.critical(message)
            raise Exception(message)
        
        # First load the model from json file
        logger.info(f"Load model from {model_json_filepath}")
        with open(model_json_filepath, 'r') as src:
            model_json = src.read()
        model = kr.models.model_from_json(model_json)
        
        # Load the weights
        logger.info(f"Load weights from {model_preload_filepath}")
        model.load_weights(model_preload_filepath)
        logger.info("Model weights loaded")
    
    # Now prepare the model for training
    # Default learning rate for Adam: lr=1e-3, but doesn't seem to work well for unet
    model = mf.compile_model(model=model,
                             optimizer=kr.optimizers.Adam(lr=start_learning_rate), 
                             loss='binary_crossentropy')

    # Define some callbacks for the training
    # Reduce the learning rate if the loss doesn't improve anymore
    reduce_lr = kr.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                               patience=20, min_lr=1e-20)

    # Custom callback that saves the best models using both train and 
    # validation metric
    model_checkpoint_saver = mh.ModelCheckpointExt(
                                model_save_dir, 
                                model_save_base_filename,
                                acc_metric_train='jaccard_coef_round',
                                acc_metric_validation='val_jaccard_coef_round')

    # Callbacks for logging
    tensorboard_log_dir = f"{model_save_dir}{os.sep}{model_save_base_filename}_tensorboard_log"
    tensorboard_logger = kr.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
    csv_logger = kr.callbacks.CSVLogger(csv_log_filepath, 
                                        append=True, separator=';')

    # Stop if no more improvement
    early_stopping = kr.callbacks.EarlyStopping(monitor='jaccard_coef_round', 
                                                patience=200,  
                                                restore_best_weights=False)
    
    # Prepare the parameters to pass to fit...
    # Supported filetypes to train/validate on
    input_ext = ['.tif', '.jpg', '.png']

    # Calculate the size of the input datasets
    #train_dataset_size = len(glob.glob(f"{traindata_dir}{os.sep}{image_subdir}{os.sep}*.*"))
    train_dataset_size = 0
    for input_ext_cur in input_ext:
        train_dataset_size += len(glob.glob(f"{traindata_dir}{os.sep}{image_subdir}{os.sep}*{input_ext_cur}"))

    #validation_dataset_size = len(glob.glob(f"{validationdata_dir}{os.sep}{image_subdir}{os.sep}*.*"))
    validation_dataset_size = 0    
    for input_ext_cur in input_ext:
        validation_dataset_size += len(glob.glob(f"{validationdata_dir}{os.sep}{image_subdir}{os.sep}*{input_ext_cur}"))
    
    # Calculate the number of steps within an epoch
    # Remark: number of steps per epoch should be at least 1, even if nb samples < batch size...
    train_steps_per_epoch = math.ceil(train_dataset_size/batch_size)
    validation_steps_per_epoch = math.ceil(validation_dataset_size/batch_size)
    
    # Start training
    logger.info(f"Start training with batch_size: {batch_size}, train_dataset_size: {train_dataset_size}, train_steps_per_epoch: {train_steps_per_epoch}, validation_dataset_size: {validation_dataset_size}, validation_steps_per_epoch: {validation_steps_per_epoch}")
    try:        
        model.fit_generator(train_gen, 
                            steps_per_epoch=train_steps_per_epoch, 
                            epochs=nb_epoch,
                            validation_data=validation_gen,
                            validation_steps=validation_steps_per_epoch,       # Number of items in validation/batch_size
                            callbacks=[model_checkpoint_saver, 
                                       reduce_lr, early_stopping,
                                       tensorboard_logger, csv_logger],
                            initial_epoch=start_epoch)
    finally:        
        # Release the memory from the GPU...
        #from keras import backend as K
        #K.clear_session()
        kr.backend.clear_session()

def create_train_generator(input_data_dir, image_subdir, mask_subdir,
                           aug_dict, batch_size=32,
                           image_color_mode="rgb", mask_color_mode="grayscale",
                           save_to_dir=None, 
                           image_save_prefix="image", mask_save_prefix="mask",
                           flag_multi_class=False, num_class=2,
                           target_size=(256,256), seed=1, class_mode=None):
    """
    Creates a generator to generate and augment train images. The augmentations
    specified in aug_dict will be applied. For the augmentations that can be 
    specified in aug_dict look at the documentation of 
    keras.preprocessing.image.ImageDataGenerator
    
    For more info about the other parameters, check keras flow_from_directory.

    Remarks: * use the same seed for image_datagen and mask_datagen to ensure 
               the transformation for image and mask is the same
             * set save_to_dir = "your path" to check results of the generator
    """
    
    image_datagen = kr.preprocessing.image.ImageDataGenerator(**aug_dict)
    mask_datagen = kr.preprocessing.image.ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        directory=input_data_dir,
        classes=[image_subdir],
        class_mode=class_mode,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        directory=input_data_dir,
        classes=[mask_subdir],
        class_mode=class_mode,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    return train_generator

def predict_dir(model,
                input_image_dir: str,
                output_base_dir: str,         
                border_pixels_to_ignore: int = 0,                
                projection_if_missing: str = None,
                input_mask_dir: str = None,
                batch_size: int = 16,                
                evaluate_mode: bool = False,
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
                                 input images images is not as good, you can
                                 specify that x pixels need to be ignored
        input_mask_dir: optional dir where the mask images are located
        batch_size: batch size to use while predicting. This must be 
                    choosen depending on the neural network architecture
                    and available memory on you GPU.
        evaluate_mode: True to run in evaluate mode
        projection: Normally the projection should be in the raster file. If it 
                    is not, you can explicitly specify one.
        force: False to skip images that already have a prediction, true to
               ignore existing predictions and overwrite them
    """
    
    logger.info(f"Start predict for input_image_dir: {input_image_dir}")

    # If we are using evaluate mode, change the output dir...
    if evaluate_mode:
        output_base_dir = output_base_dir + '_eval'
        
    # Create the output dir's if they don't exist yet...
    for dir in [output_base_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Get list of all image files to process...
    image_filepaths = []
    input_ext = ['.tif', '.jpg']
    for input_ext_cur in input_ext:
        image_filepaths.extend(glob.glob(f"{input_image_dir}{os.sep}**{os.sep}*{input_ext_cur}", recursive=True))
    nb_todo = len(image_filepaths)
    logger.info(f"Found {nb_todo} {input_ext} images to predict on in {input_image_dir}")
    
    # If force is false, get list of all existing predictions
    # Getting the list once is way faster than checking file per file later on!
    images_done_log_filepath = os.path.join(output_base_dir, "images_done.txt")
    images_error_log_filepath = os.path.join(output_base_dir, 
                                             "images_error.txt")
    if force is False:
        # First read the listing file if it exists
        image_done_filenames = set()
        if os.path.exists(images_done_log_filepath):
            with open(images_done_log_filepath) as f:
                for filename in f:
                    image_done_filenames.add(filename.rstrip())
        if len(image_done_filenames) > 0:
            logger.info(f"Found {len(image_done_filenames)} predicted images in output dir, they will be skipped")
        
    # Loop through all files to process them...
    curr_batch_image_infos = []
    nb_processed = 0
    image_filepaths_sorted = sorted(image_filepaths)
    with futures.ThreadPoolExecutor(batch_size) as pool:
        
        for i, image_filepath in enumerate(image_filepaths_sorted):
    
            # If force is false and prediction exists... skip
            if force is False:
               filename = os.path.basename(image_filepath)
               if filename in image_done_filenames:
                   logger.debug(f"Predict for image has already been done before and force is False, so skip: {filename}")
                   continue
                    
            # Prepare the filepath for the output
            image_filepath_noext = os.path.splitext(image_filepath)[0]
            if evaluate_mode:
                # In evaluate mode, put everyting in output base dir for easier 
                # comparison
                image_dir, image_filename_noext = os.path.split(image_filepath_noext)
                tmp_output_filepath = os.path.join(output_base_dir, image_filename_noext)
            else:
                tmp_output_filepath = image_filepath_noext.replace(input_image_dir,
                                                            output_base_dir)
            output_pred_filepath = f"{tmp_output_filepath}_pred.tif"       
            output_dir, output_pred_filename = os.path.split(output_pred_filepath)
            
            logger.debug(f"Start predict for image {image_filepath}")
                    
            # Init start time at the first file that isn't skipped
            if nb_processed == 0:
                start_time = datetime.datetime.now()
            nb_processed += 1
            
            # Append the image info to the batch array so they can be treated in 
            # bulk if the batch size is reached
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
                for j, read_result in enumerate(read_results):
                    curr_batch_image_infos_ext.append(
                            {'input_image_filepath': curr_batch_image_infos[j]['input_image_filepath'],
                             'output_pred_filepath': curr_batch_image_infos[j]['output_pred_filepath'],
                             'output_dir': curr_batch_image_infos[j]['output_dir'],
                             'image_crs': read_result['image_crs'],
                             'image_transform': read_result['image_transform'],
                             'image_data': read_result['image_data']})

                # Predict!
                logger.debug(f"Start prediction for {nb_images_in_batch} images")
                images = [info.get('image_data') for info in curr_batch_image_infos_ext]
                curr_batch_image_pred_arr = model.predict_on_batch(
                        np.asarray(images))
                
                # TODO: possibly add this check in exception handler to make the 
                # error message clearer!
                # Check if the image size is OK for the segmentation model
                '''
                m.check_image_size(decoder=decoder,
                                   input_width=image_data.shape[0], 
                                   input_height=image_data.shape[1])
                '''
                
                # Postprocess predictions
                logger.debug("Start post-processing")    
                for j, image_info in enumerate(curr_batch_image_infos_ext):    
                    try:
                        # Postprocess
                        postp.postprocess_prediction(
                                image_filepath=image_info['input_image_filepath'],
                                output_dir=image_info['output_dir'],
                                image_crs=image_info['image_crs'],
                                image_transform=image_info['image_transform'],
                                image_pred_arr=curr_batch_image_pred_arr[j],
                                input_mask_dir=input_mask_dir,
                                border_pixels_to_ignore=border_pixels_to_ignore,
                                evaluate_mode=evaluate_mode,
                                force=force)
    
                        # Write line to file with done files...
                        with open(images_done_log_filepath, "a+") as f:
                            f.write(os.path.basename(
                                    image_info['input_image_filepath']) + '\n')
                    except:
                        logger.error(f"Error postprocessing pred for {image_info['input_image_filepath']}")
    
                        # Write line to file with done files...
                        with open(images_error_log_filepath, "a+") as f:
                            f.write(os.path.basename(
                                    image_info['input_image_filepath']) + '\n')
                        
                logger.debug("Post-processing ready")
            
                # Log the progress and prediction speed
                time_passed_s = (datetime.datetime.now()-start_time).total_seconds()
                time_passed_lastbatch_s = (datetime.datetime.now()-start_time_batch_read).total_seconds()
                if time_passed_s > 0 and time_passed_lastbatch_s > 0:
                    nb_per_hour = (nb_processed/time_passed_s) * 3600
                    nb_per_hour_lastbatch = (nb_images_in_batch/time_passed_lastbatch_s) * 3600
                    hours_to_go = (int)((nb_todo - i)/nb_per_hour)
                    min_to_go = (int)((((nb_todo - i)/nb_per_hour)%1)*60)
                    print(f"\r{hours_to_go}:{min_to_go} left for {nb_todo-i} todo at {nb_per_hour:0.0f}/h ({nb_per_hour_lastbatch:0.0f}/h last batch) in ...{input_image_dir[-30:]}",
                          end='', flush=True)
                
                # Reset variable for next batch
                curr_batch_image_infos = []

def read_image(image_filepath: str,
               projection_if_missing: str = None):

    # Read input file
    # Because sometimes a read seems to fail, retry upt to 3 times...
    retry_count = 0
    while True:
        try:
            with rio.open(image_filepath) as image_ds:
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
            image_crs = projection_if_missing
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

def save_prediction_uint8(output_filepath: str,
                          image_pred_arr,
                          image_crs: str,
                          image_transform,
                          border_pixels_to_ignore: int = None,
                          force: bool = False):

    # Make sure the output dir exists...
    output_dir = os.path.split(output_filepath)[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Input should be float32
    if image_pred_arr.dtype != np.float32:
        raise Exception(f"image prediction is of the wrong type: {image_pred_arr.dtype}") 
    
    # Convert to uint8
    image_pred_uint8 = (image_pred_arr * 255).astype(np.uint8)
    
    # Reshape array from 4 dims (image_id, width, height, nb_channels) to 2.
    image_pred_uint8 = image_pred_uint8.reshape((image_pred_uint8.shape[0], image_pred_uint8.shape[1]))

    # Make the pixels at the borders of the prediction black so they are ignored
    image_pred_uint8_cropped = image_pred_uint8
    if border_pixels_to_ignore and border_pixels_to_ignore > 0:
        image_pred_uint8_cropped[0:border_pixels_to_ignore,:] = 0    # Left border
        image_pred_uint8_cropped[-border_pixels_to_ignore:,:] = 0    # Right border
        image_pred_uint8_cropped[:,0:border_pixels_to_ignore] = 0    # Top border
        image_pred_uint8_cropped[:,-border_pixels_to_ignore:] = 0    # Bottom border

    # Check if the result is entirely black... if so, don't save
    thresshold = 127
    if not np.any(image_pred_uint8_cropped >= thresshold):
        logger.debug('Prediction is entirely black!')
        return
        
    # Write prediction to file
    logger.debug("Save original prediction")
    image_width = image_pred_arr.shape[0]
    image_height = image_pred_arr.shape[1]
    with rio.open(output_filepath, 'w', driver='GTiff', compress='lzw',
                  height=image_height, width=image_width, 
                  count=1, dtype=rio.uint8, crs=image_crs, transform=image_transform) as dst:
        dst.write(image_pred_uint8_cropped, 1)
        
# If the script is ran directly...
if __name__ == '__main__':
    message = "Main is not implemented"
    logger.error(message)
    raise Exception(message)
    

# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images

@author: Pieter Roggemans
"""

import logging
import os
import glob
import datetime

import numpy as np
import pandas as pd
import keras as kr

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
          image_subdir: str = "image",
          mask_subdir: str = "mask",
          model_preload_filepath: str = None,
          batch_size: int = 32,
          nb_epoch: int = 100,
          augmented_subdir: str = None):

    image_width = 512
    image_height = 512

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
    model_json_filepath = os.path.join(model_save_dir, 
                                       model_json_filename)
    if not model_preload_filepath:
        # If no existing model provided, create it from scratch
        # Get the model we want to use
        model = mf.get_model(encoder=model_encoder,
                            decoder=model_decoder, 
                            n_channels=3, n_classes=1)
        
        # Save the model architecture to json if it doesn't exist yet
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

    # Stop if no more omprovement
    early_stopping = kr.callbacks.EarlyStopping(monitor='jaccard_coef_round', 
                                                patience=200,  
                                                restore_best_weights=False)
    
    # Start training
    train_dataset_size = len(glob.glob(f"{traindata_dir}{os.sep}{image_subdir}{os.sep}*.*"))
    train_steps_per_epoch = int(train_dataset_size/batch_size)
    validation_dataset_size = len(glob.glob(f"{validationdata_dir}{os.sep}{image_subdir}{os.sep}*.*"))
    validation_steps_per_epoch = int(validation_dataset_size/batch_size)
    model.fit_generator(train_gen, 
                        steps_per_epoch=train_steps_per_epoch, epochs=nb_epoch,
                        validation_data=validation_gen,
                        validation_steps=validation_steps_per_epoch,       # Number of items in validation/batch_size
                        callbacks=[model_checkpoint_saver, 
                                   reduce_lr, early_stopping,
                                   tensorboard_logger, csv_logger],
                        initial_epoch=start_epoch)

def create_train_generator(input_data_dir, image_subdir, mask_subdir,
                           aug_dict, batch_size=32,
                           image_color_mode="rgb", mask_color_mode="grayscale",
                           save_to_dir=None, 
                           image_save_prefix="image", mask_save_prefix="mask",
                           flag_multi_class=False, num_class=2,
                           target_size=(256,256), seed=1, class_mode=None):
    '''
    Can generate image and mask at the same time

    Remarks: * use the same seed for image_datagen and mask_datagen to ensure the
               transformation for image and mask is the same
             * if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
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

def predict(model,
            input_image_dir: str,
            output_base_dir: str,
            border_pixels_to_ignore: int = 0,
            input_mask_dir: str = None,
            batch_size: int = 16,
            evaluate_mode: bool = False,
            force: bool = False):

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
    nb_files = len(image_filepaths)
    logger.info(f"Found {nb_files} {input_ext} images to predict on in {input_image_dir}")
    
    # If force is false, get list of all existing predictions
    # Getting the list once is way faster than checking file per file later on!
    images_done_log_filename = "images_done.txt"
    images_done_log_filepath = os.path.join(output_base_dir, images_done_log_filename)
    if force is False:
        # First read the listing file if it exists
        image_done_filenames = set()
        if os.path.exists(images_done_log_filepath):
            with open(images_done_log_filepath) as f:
                for filename in f:
                    image_done_filenames.add(filename.rstrip())
            
        logger.info(f"Found {len(image_done_filenames)} predicted images in output dir, they will be skipped")
        #logger.info(f"Found {image_done_filenames}")
        
    # Loop through all files to process them...
    curr_batch_image_data_arr = []
    curr_batch_image_info_arr = []
    nb_predicted = 0
    image_filepaths_sorted = sorted(image_filepaths)
    for i, image_filepath in enumerate(image_filepaths_sorted):

        # If force is false and prediction exists... skip
        if force is False:
           filename = os.path.basename(image_filepath)
           if filename in image_done_filenames:
               logger.debug(f"Predict for image has already been done before and force is False, so skip: {filename}")
               continue
                
        # Prepare the filepath for the output
        if evaluate_mode:
            # In evaluate mode, put everyting in output base dir for easier 
            # comparison
            image_dir, image_filename = os.path.split(image_filepath)
            tmp_filepath = os.path.join(output_base_dir, image_filename)
        else:
            tmp_filepath = image_filepath.replace(input_image_dir,
                                                  output_base_dir)
        output_dir, image_pred_filename = os.path.split(tmp_filepath)
        image_pred_filename_noext = os.path.splitext(image_pred_filename)[0]       
        image_pred_filename = f"{image_pred_filename_noext}_pred.tif"
        logger.debug(f"Start predict for image {image_filepath}")
                
        # Init start time at the first file that isn't skipped
        if nb_predicted == 0:
            start_time = datetime.datetime.now()
        nb_predicted += 1
        
        # Append the image info to the batch array so they can be treated in 
        # bulk if the batch size is reached
        curr_batch_image_info_arr.append({'image_filepath': image_filepath,
                                          'output_dir': output_dir})
        
        # If the batch size is reached or we are at the last images
        nb_images_in_batch = len(curr_batch_image_info_arr)
        if(nb_images_in_batch == batch_size or i == (nb_files-1)):

            start_time_batch_read = datetime.datetime.now()
            
            # Read all input images for the batch 
            # Remark: reading them in parallel doesn't work because the image 
            # data is too large to give back as return value in future.result()
            # TODO: try using predict_generator for paralellisation
            logger.debug(f"Start reading input {nb_images_in_batch} images")

            for j, image_info in enumerate(curr_batch_image_info_arr):
                
                image_data = postp.read_image(image_filepath=image_info['image_filepath'])
                curr_batch_image_data_arr.append(image_data)
                    
            # Predict!
            logger.debug(f"Start prediction for {nb_images_in_batch} images")
            curr_batch_image_pred_arr = model.predict_on_batch(
                    np.asarray(curr_batch_image_data_arr))
            
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
            for j, image_info in enumerate(curr_batch_image_info_arr):
                
                postp.postprocess_prediction(image_filepath=image_info['image_filepath'],
                                       output_dir=image_info['output_dir'],
                                       image_pred_arr=curr_batch_image_pred_arr[j],
                                       input_mask_dir=input_mask_dir,
                                       border_pixels_to_ignore=border_pixels_to_ignore,
                                       evaluate_mode=evaluate_mode,
                                       force=force)
                
                # Write line to file with done files...
                with open(images_done_log_filepath, "a+") as f:
                    f.write(os.path.basename(image_info['image_filepath']) + '\n')

            logger.debug("Post-processing ready")
        
            # Log the progress and prediction speed
            time_passed_s = (datetime.datetime.now()-start_time).total_seconds()
            time_passed_lastbatch_s = (datetime.datetime.now()-start_time_batch_read).total_seconds()
            if time_passed_s > 0 and time_passed_lastbatch_s > 0:
                images_per_hour = (nb_predicted/time_passed_s) * 3600
                images_per_hour_lastbatch = (batch_size/time_passed_lastbatch_s) * 3600
                hours_to_go = (int)((nb_files - i)/images_per_hour)
                min_to_go = (int)((((nb_files - i)/images_per_hour)%1)*60)
                print(f"{hours_to_go}:{min_to_go} left for {nb_files-i} images at {images_per_hour:0.0f}/h ({images_per_hour_lastbatch:0.0f}/h last batch) in ...{input_image_dir[-30:]}")
            
            # Reset variables for next batch
            curr_batch_image_data_arr = []
            curr_batch_image_info_arr = []
            
# If the script is ran directly...
if __name__ == '__main__':
    message = "Main is not implemented"
    logger.error(message)
    raise Exception(message)
    
# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images.
"""

import logging
import math
import os
from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow import keras as kr
#import keras as kr

import pandas as pd

import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh
import orthoseg.postprocess_predictions as postp

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train(
        traindata_dir: Path,
        validationdata_dir: Path,
        model_encoder: str,
        model_decoder: str,
        model_save_dir: Path,
        model_save_base_filename: str,
        image_augment_dict: dict, 
        mask_augment_dict: dict,
        model_preload_filepath: Optional[Path] = None,
        nb_classes: int = 1,
        class_weights: list = None,
        nb_channels: int = 3,
        image_width: int = 512,
        image_height: int = 512,
        image_subdir: str = "image",
        mask_subdir: str = "mask",
        batch_size: int = 32,
        nb_epoch: int = 100,
        save_augmented_subdir: str = None):
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
        image_augment_dict: augmentation  
        mask_augment_dict:  
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
    ##### Some checks on the input parameters #####
    if(nb_classes > 1 
       and class_weights is not None
       and nb_classes != len(class_weights)):
            raise Exception(f"The number of class weight ({class_weights}) should equal the number of classes ({nb_classes})!")

    ##### Init #####
    # These are the augmentations that will be applied to the input training images/masks
    # Remark: fill_mode + cval are defined as they are so missing pixels after eg. rotation
    #         are filled with 0, and so the mask will take care that they are +- ignored.

    # Create the train generator
    train_gen = create_train_generator(
            input_data_dir=traindata_dir,
            image_subdir=image_subdir, 
            mask_subdir=mask_subdir,
            image_augment_dict=image_augment_dict, 
            mask_augment_dict=mask_augment_dict, 
            batch_size=batch_size,
            target_size=(image_width, image_height), 
            nb_classes=nb_classes, 
            save_to_subdir=save_augmented_subdir, 
            seed=2)

    # Create validation generator
    validation_augment_dict = dict(rescale=1./255)
    validation_gen = create_train_generator(
            input_data_dir=validationdata_dir,
            image_subdir=image_subdir, 
            mask_subdir=mask_subdir,
            image_augment_dict=validation_augment_dict,
            mask_augment_dict=validation_augment_dict, 
            batch_size=batch_size,
            target_size=(image_width, image_height), 
            nb_classes=nb_classes, 
            save_to_subdir=save_augmented_subdir,
            shuffle=False, 
            seed=3)

    # Get the max epoch number from the log file if it exists...
    start_epoch = 0
    start_learning_rate = 1e-4  # Best set to 0.0001 to start (1e-3 is not ok)
    csv_log_filepath = model_save_dir / (model_save_base_filename + '_log.csv')
    if csv_log_filepath.exists() and os.path.getsize(csv_log_filepath) > 0:
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
    model_json_filepath = model_save_dir / model_json_filename
    if nb_classes > 1:
        model_activation = 'softmax'
    else:
        model_activation = 'sigmoid'
    
    # If no existing model provided, create it from scratch
    if not model_preload_filepath:
        # Get the model we want to use
        model = mf.get_model(
                encoder=model_encoder, decoder=model_decoder, 
                nb_channels=nb_channels, nb_classes=nb_classes, activation=model_activation)
        
        # Save the model architecture to json if it doesn't exist yet
        if not model_save_dir.exists():
            model_save_dir.mkdir(parents=True)
        if not model_json_filepath.exists():
            with model_json_filepath.open('w') as dst:
                dst.write(model.to_json())
    else:
        # If a preload model is provided, load that if it exists...
        if not model_preload_filepath.exists():
            message = f"Error: preload model file doesn't exist: {model_preload_filepath}"
            logger.critical(message)
            raise Exception(message)
        
        '''
        # First load the model from json file
        logger.info(f"Load model from {model_json_filepath}")
        with open(model_json_filepath, 'r') as src:
            model_json = src.read()
        model = kr.models.model_from_json(model_json)
        
        # Load the weights
        logger.info(f"Load weights from {model_preload_filepath}")
        model.load_weights(model_preload_filepath)
        logger.info("Model weights loaded")
        '''
        # Load the existing model
        # Remark: compiling during load crashes, so compile 'manually'
        logger.info(f"Load model from {model_preload_filepath}")
        model = mf.load_model(model_preload_filepath, compile=False)

    # Now prepare the model for training
    nb_gpu = len(tf.config.experimental.list_physical_devices('GPU'))

    # TODO: because of bug in tensorflow 1.14, multi GPU doesn't work (this way),
    # so always use standard model
    if nb_gpu <= 1:
        model_for_train = model
        logger.info(f"Train using single GPU or CPU, with nb_gpu: {nb_gpu}")
    else:
        # If multiple GPU's available, create multi_gpu_model
        try:
            model_for_train = kr.utils.multi_gpu_model(model, gpus=nb_gpu, cpu_relocation=True)
            logger.info(f"Train using multiple GPUs: {nb_gpu}, batch size becomes: {batch_size*nb_gpu}")
            batch_size *= nb_gpu
        except ValueError:
            logger.info("Train using single GPU or CPU")
            model_for_train = model

    # If we started a model from scratch, compile it to prepare for training
    # Remark: compile shouldn't be done explicitly when using a preload model, 
    #         but compiling during load crashes, so for now always compile.
    #if not model_preload_filepath:
    optimizer = kr.optimizers.Adam(lr=start_learning_rate)
    if model_activation == 'softmax':
        if class_weights is not None: 
            loss = 'weighted_categorical_crossentropy'
        else:
            loss = 'categorical_crossentropy'
    else:
        loss = 'binary_crossentropy' 
    logger.info(f"Compile model with loss: {loss}, class_weights: {class_weights}")
    model_for_train = mf.compile_model(
            model=model_for_train, optimizer=optimizer, loss=loss, class_weights=class_weights)

    # Define some callbacks for the training
    # Reduce the learning rate if the loss doesn't improve anymore
    reduce_lr = kr.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.2, patience=20, min_lr=1e-20, verbose=True)

    # Custom callback that saves the best models using both train and 
    # validation metric
    # Remark: the save of the model should be done on the standard model, not
    #         on the parallel_model, otherwise issues to use it afterwards
    if nb_gpu > 1:
        model_template_for_save = model
    else:
        model_template_for_save = None
    model_checkpoint_saver = mh.ModelCheckpointExt(
            model_save_dir=model_save_dir, 
            model_save_base_filename=model_save_base_filename,
            monitor_metric_train='loss',
            monitor_metric_validation='val_loss',
            monitor_metric_mode='min',
            save_format='h5',
            save_best_only=True,
            model_template_for_save=model_template_for_save)

    # Callbacks for logging
    tensorboard_log_dir = model_save_dir / (model_save_base_filename + '_tensorboard_log')
    tensorboard_logger = kr.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir))
    csv_logger = kr.callbacks.CSVLogger(str(csv_log_filepath), append=True, separator=';')

    # Stop if no more improvement
    early_stopping = kr.callbacks.EarlyStopping(
            monitor='val_loss', patience=200, restore_best_weights=False)
    
    # Prepare the parameters to pass to fit...
    # Supported filetypes to train/validate on
    input_ext = ['.tif', '.jpg', '.png']

    # Calculate the size of the input datasets
    #train_dataset_size = len(glob.glob(f"{traindata_dir}/{image_subdir}/*.*"))
    train_dataset_size = 0
    for input_ext_cur in input_ext:
        traindata_image_dir = traindata_dir / image_subdir
        train_dataset_size += len(list(traindata_image_dir.rglob('*' + input_ext_cur)))
    validation_dataset_size = 0    
    for input_ext_cur in input_ext:
        validationdata_image_dir =validationdata_dir / image_subdir
        validation_dataset_size += len(list(validationdata_image_dir.rglob('*' + input_ext_cur)))
    
    # Calculate the number of steps within an epoch
    # Remark: number of steps per epoch should be at least 1, even if nb samples < batch size...
    train_steps_per_epoch = math.ceil(train_dataset_size/batch_size)
    validation_steps_per_epoch = math.ceil(validation_dataset_size/batch_size)
    
    # Start training
    logger.info(f"Start training with batch_size: {batch_size}, train_dataset_size: {train_dataset_size}, train_steps_per_epoch: {train_steps_per_epoch}, validation_dataset_size: {validation_dataset_size}, validation_steps_per_epoch: {validation_steps_per_epoch}")
    try:        
        model_for_train.fit(
                train_gen, 
                steps_per_epoch=train_steps_per_epoch, 
                epochs=nb_epoch,
                validation_data=validation_gen,
                validation_steps=validation_steps_per_epoch,       # Number of items in validation/batch_size
                callbacks=[model_checkpoint_saver, 
                           reduce_lr, early_stopping,
                           tensorboard_logger,
                           csv_logger],
                #class_weight={'0': 1, '1': 10, '2': 2},
                initial_epoch=start_epoch)
        
    finally:
        # Release the memory from the GPU...
        #from keras import backend as K
        #K.clear_session()
        kr.backend.clear_session()

def create_train_generator(
        input_data_dir: Path, 
        image_subdir: str, 
        mask_subdir: str,
        image_augment_dict: dict, 
        mask_augment_dict: dict, 
        batch_size: int = 32,
        image_color_mode: str = "rgb", 
        mask_color_mode: str = "grayscale",
        save_to_subdir: str = None, 
        image_save_prefix: str = 'image', 
        mask_save_prefix: str = 'mask',
        nb_classes: int = 1,
        target_size: Tuple[int, int] = (256,256), 
        shuffle: bool = True,
        seed: int = 1):
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
    image_datagen = kr.preprocessing.image.ImageDataGenerator(**image_augment_dict)
    mask_datagen = kr.preprocessing.image.ImageDataGenerator(**mask_augment_dict)

    # Format save_to_dir
    # Remark: flow_from_directory doesn't support Path, so supply str immediately as well,
    # otherwise, if str(Path) is used later on, it becomes 'None' instead of None !!!
    save_to_dir = None
    save_to_dir_str = None
    if save_to_subdir is not None:
        save_to_dir = input_data_dir / save_to_subdir
        if not save_to_dir.exists():
            save_to_dir.mkdir(parents=True)
        save_to_dir_str = str(save_to_dir)

    image_generator = image_datagen.flow_from_directory(
            directory=str(input_data_dir),
            classes=[image_subdir],
            class_mode=None,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir_str,
            save_prefix=image_save_prefix,
            shuffle=shuffle,
            seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
            directory=str(input_data_dir),
            classes=[mask_subdir],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir_str,
            save_prefix=mask_save_prefix,
            shuffle=shuffle,
            seed=seed)

    train_generator = zip(image_generator, mask_generator)

    for image, mask in train_generator:

        # Rename files in save dir to make compare between image and mask easy    
        if save_to_dir is not None:
            def rename_prefix_to_suffix(save_dir: Path, save_prefix: str):
                """ Rename file so the filename prefix is moved to being a suffix  """
                paths = save_dir.glob(f"{save_prefix}_*.png")
                for path in paths:
                    dir = path.parent
                    filename_noext = path.stem
                    ext = path.suffix
                    rename_filename_noext = f"{filename_noext.replace(save_prefix + '_', '')}_{save_prefix}"
                    for index in range(1, 999):
                        rename_path = dir / f"{rename_filename_noext}_{index}{ext}"
                        # If the path to rename to exists already, add an index to keep file name unique
                        if not rename_path.exists():
                            path.rename(rename_path)
                            break                           
                        else:
                            continue
                        raise Exception(f"No new filename found for {path}")

            rename_prefix_to_suffix(save_to_dir, mask_save_prefix)
            rename_prefix_to_suffix(save_to_dir, image_save_prefix)

        # If loss is categorical_crossentropy -> one-hot encode masks
        if nb_classes > 1:
            mask = kr.utils.to_categorical(mask, nb_classes)
        yield (image, mask)
    
# If the script is ran directly...
if __name__ == '__main__':
    raise Exception("Not implemented")
    
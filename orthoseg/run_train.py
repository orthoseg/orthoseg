# -*- coding: utf-8 -*-
"""
Module to make it easy to start a training session.
"""

import logging
import os
from pathlib import Path

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow import keras as kr
#import keras as kr

from orthoseg.helpers import config_helper as conf
import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh
import orthoseg.prepare_traindatasets as prep
from orthoseg import segment_predict
from orthoseg import segment_train

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def run_training_session():
    """
    Run a training session.
    """
    ##### Init #####
    # TODO: add something to delete old data, predictions???

    # First check if the segment_subject has a valid name
    segment_subject = conf.general['segment_subject']
    if segment_subject == 'MUST_OVERRIDE':
        raise Exception("The segment_subject parameter needs to be overridden in the subject specific config file!!!") 
    elif '_' in segment_subject:
        raise Exception(f"The segment_subject parameter should not contain '_', so this is invalid: {segment_subject}!!!") 

    # Create the output dir's if they don't exist yet...
    for dir in [conf.dirs.getpath('project_dir'), conf.dirs.getpath('training_dir')]:
        if dir and not dir.exists():
            dir.mkdir()
    
    # If the training data doesn't exist yet, create it
    # Get the config needeed

    # Get the label input info, and clean it so it is practical to use
    label_files = conf.train.getdict('label_datasources')
    for label_file_key in label_files:
        # Convert the str file paths to Path objects
        label_files[label_file_key]['locations_path'] = Path(
                label_files[label_file_key]['locations_path'])
        label_files[label_file_key]['data_path'] = Path(
                label_files[label_file_key]['data_path'])
    first_label_file = list(label_files)[0]
    train_image_layer = label_files[first_label_file]['image_layer']
    train_projection = conf.image_layers[train_image_layer]['projection']
    label_names_burn_values = conf.train.getdict('label_names_burn_values')
    nb_classes = len(label_names_burn_values)
    # If more than 1 class... add a class for the background!
    if nb_classes > 1:
        nb_classes += 1

    # First the "train" training dataset
    force_model_traindata_version = conf.train.getint('force_model_traindata_version')
    if force_model_traindata_version > -1:
        training_dir = conf.dirs.getpath('training_train_basedir') / f"{force_model_traindata_version:02d}"
        traindata_version = force_model_traindata_version
    else:
        logger.info("Prepare train, validation and test data")
        training_dir, traindata_version = prep.prepare_traindatasets(
                label_files=label_files,
                label_names_burn_values=label_names_burn_values,
                image_layers=conf.image_layers,
                training_dir=conf.dirs.getpath('training_dir'),
                image_pixel_x_size=conf.train.getfloat('image_pixel_x_size'),
                image_pixel_y_size=conf.train.getfloat('image_pixel_y_size'),
                image_pixel_width=conf.train.getint('image_pixel_width'),
                image_pixel_height=conf.train.getint('image_pixel_height'))
    logger.info(f"Traindata dir to use is {training_dir}, with traindata_version: {traindata_version}")

    traindata_dir = training_dir / 'train'
    validationdata_dir = training_dir / 'validation'
    testdata_dir = training_dir / 'test'
    
    # Get the best model that already exists for this train dataset
    model_dir = conf.dirs.getpath('model_dir')
    segment_subject = conf.general['segment_subject']
    model_architecture = conf.model['architecture']
    hyperparams_version = conf.train.getint('hyperparams_version')
    best_model_curr_train_version = mh.get_best_model(
            model_dir=model_dir, 
            segment_subject=segment_subject,
            traindata_version=traindata_version,
            model_architecture=model_architecture,
            hyperparams_version=hyperparams_version)

    # Check if training is needed
    resume_train = conf.train.getboolean('resume_train')
    if resume_train is False:
        # If no (best) model found, training needed!
        if best_model_curr_train_version is None:
            train_needed = True
        elif conf.train.getboolean('force_train') is True:
            train_needed = True
        else:
            logger.info("JUST PREDICT, without training: preload_existing_model is false and model found")
            train_needed = False
    else:
        # We want to preload an existing model and models were found
        if best_model_curr_train_version is None:
            logger.info(f"PRELOAD model and continue TRAINING it: {best_model_curr_train_version['filename']}")
            train_needed = True
        else:
            message = "STOP: preload_existing_model is true but no model was found!"
            logger.error(message)
            raise Exception(message)
    
    # If training is needed
    if train_needed is True:

        # If a model already exists, use it to predict (possibly new) training and 
        # validation dataset. This way it is possible to have a quick check on errors
        # in (new) added labels in the datasets.

        # Get the current best model that already exists for this subject
        best_recent_model = mh.get_best_model(model_dir=model_dir)
        if best_recent_model is not None:
            try:
                logger.info(f"Load model + weights from {best_recent_model['filepath']}")    
                model = mf.load_model(best_recent_model['filepath'], compile=False)            
                logger.info("Loaded model + weights")

                # Prepare output subdir to be used for predictions
                predict_out_subdir, _ = os.path.splitext(best_recent_model['filename'])
                
                # Predict training dataset
                segment_predict.predict_dir(
                        model=model,
                        input_image_dir=traindata_dir / 'image',
                        output_base_dir=traindata_dir / predict_out_subdir,
                        projection_if_missing=train_projection,
                        input_mask_dir=traindata_dir / 'mask',
                        batch_size=conf.train.getint('batch_size_predict'), 
                        evaluate_mode=True)
                    
                # Predict validation dataset
                segment_predict.predict_dir(
                        model=model,
                        input_image_dir=validationdata_dir / 'image',
                        output_base_dir=validationdata_dir / predict_out_subdir,
                        projection_if_missing=train_projection,
                        input_mask_dir=validationdata_dir / 'mask',
                        batch_size=conf.train.getint('batch_size_predict'), 
                        evaluate_mode=True)
                del model
            except Exception as ex:
                logger.warn(f"Exception trying to predict with old model: {ex}")
        
        # Now we can really start training
        logger.info('Start training')

        model_preload_filepath = None
        if best_model_curr_train_version is not None:
            model_preload_filepath = best_model_curr_train_version['filepath']
        elif conf.train.getboolean('preload_with_previous_traindata'):
            best_model_for_architecture = mh.get_best_model(
                    model_dir=model_dir, 
                    segment_subject=segment_subject,
                    model_architecture=model_architecture,
                    traindata_version=traindata_version)
            if best_model_for_architecture is not None:
                model_preload_filepath = best_model_for_architecture['filepath']
        
        hyperparams = mh.HyperParams(
                image_augmentations=conf.train.getdict('image_augmentations'),
                mask_augmentations=conf.train.getdict('mask_augmentations'),
                hyperparams_version=hyperparams_version,
                nb_classes=nb_classes,
                batch_size=conf.train.getint('batch_size_fit'), 
                nb_epoch=conf.train.getint('max_epoch'))
                
        segment_train.train(
                traindata_dir=traindata_dir,
                validationdata_dir=validationdata_dir,
                model_save_dir=model_dir,
                segment_subject=segment_subject,
                traindata_version=traindata_version,                
                model_architecture=model_architecture,
                hyperparams=hyperparams,
                model_preload_filepath=model_preload_filepath,
                nb_channels=conf.model.getint('nb_channels'),
                image_width=conf.train.getint('image_pixel_width'),
                image_height=conf.train.getint('image_pixel_height')) 
    
        # Now get the best model found during training
        best_model_curr_train_version = mh.get_best_model(
                model_dir=model_dir, 
                segment_subject=segment_subject,
                model_architecture=model_architecture,
                traindata_version=traindata_version)

    # Now predict on the train,... data  
    logger.info(f"PREDICT test data with best model: {best_model_curr_train_version['filename']}")
    
    # Load prediction model...
    logger.info(f"Load model + weights from {best_model_curr_train_version['filepath']}")    
    model = mf.load_model(best_model_curr_train_version['filepath'], compile=False)            
    logger.info("Loaded model + weights")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir, _ = os.path.splitext(best_model_curr_train_version['filename'])
    
    # Predict training dataset
    segment_predict.predict_dir(
            model=model,
            input_image_dir=traindata_dir / 'image',
            output_base_dir=traindata_dir / predict_out_subdir,
            projection_if_missing=train_projection,
            input_mask_dir=traindata_dir / 'mask',
            batch_size=conf.train.getint('batch_size_predict'), 
            evaluate_mode=True)
    
    # Predict validation dataset
    segment_predict.predict_dir(
            model=model,
            input_image_dir=validationdata_dir / 'image',
            output_base_dir=validationdata_dir / predict_out_subdir,
            projection_if_missing=train_projection,
            input_mask_dir=validationdata_dir / 'mask',
            batch_size=conf.train.getint('batch_size_predict'), 
            evaluate_mode=True)

    # Predict test dataset, if it exists
    if testdata_dir is not None and testdata_dir.exists():
        segment_predict.predict_dir(
                model=model,
                input_image_dir=testdata_dir / 'image',
                output_base_dir=testdata_dir / predict_out_subdir, 
                projection_if_missing=train_projection,
                input_mask_dir=testdata_dir / 'mask',
                batch_size=conf.train.getint('batch_size_predict'), 
                evaluate_mode=True)
    
    # Predict extra test dataset with random images in the roi, to add to 
    # train and/or validation dataset if inaccuracies are found
    # -> this is very useful to find false positives to improve the datasets
    if conf.dirs.getpath('predictsample_image_input_dir').exists():
        segment_predict.predict_dir(
                model=model,
                input_image_dir=conf.dirs.getpath('predictsample_image_input_dir'),
                output_base_dir=conf.dirs.getpath('predictsample_image_output_basedir') / predict_out_subdir,
                projection_if_missing=train_projection,
                batch_size=conf.train.getint('batch_size_predict'), 
                evaluate_mode=True)
    
if __name__ == '__main__':
    None
    

# -*- coding: utf-8 -*-
"""
Execute an orthoseg task.
"""

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import List

import tensorflow as tf
from tensorflow import keras as kr  

from orthoseg.helpers import config_helper as conf 
from orthoseg.helpers import log_helper
from orthoseg.lib import postprocess_predictions as postp
from orthoseg.lib import predicter
from orthoseg.lib import prepare_traindatasets as prep
from orthoseg.lib import trainer
import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh
from orthoseg.util import ows_util

def orthoseg_argstr(argstr):
    args = shlex.split(argstr)
    orthoseg_args(args)

def orthoseg_args(args):

    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument("--config_dir", type=str, required=True,
            help="The config to perform the action with")
    required.add_argument("--config", type=str, required=True,
            help="The config to perform the action with")
    required.add_argument("--action", type=str, required=True,
            help="The action you want to perform")
    
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    # Add back help 
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')

    # Interprete arguments
    args = parser.parse_args(args)

    ##### Run! #####
    orthoseg(
            config_dir=Path(args.config_dir),
            action=args.action, 
            config=args.config)

def orthoseg(
        config_dir: Path,
        action: str,
        config: str):

    # Get needed config + load it
    print(f"Start {action} on {config}")
    config_filepaths = get_needed_config_files(config_dir=config_dir, config_filename=config)
    layer_config_filepath = config_dir / 'image_layers.ini'
    conf.read_config(config_filepaths, layer_config_filepath)
    
    # Main initialisation of the logging
    global logger
    logger = log_helper.main_log_init(conf.dirs.getpath('log_training_dir'), __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Now start the appropriate action 
    try:
        if(action == 'train'):
            run_training_session()
        elif(action == 'predict'):
            run_prediction()
        elif(action == 'load_images'):
            load_images()
        elif(action == 'load_testsample_images'):
            load_images(load_testsample_images=True)
        elif(action == 'postprocess'):
            postprocess_predictions()
        else:
            raise Exception(f"Unsupported action: {action}")
    except Exception as ex:
        message = f"OrthoSeg ERROR in action {action} on {config}"
        logger.exception(message)
        raise Exception(message) from ex

def get_needed_config_files(
        config_dir: Path,
        config_filename: str = None) -> List[Path]:

    # General settings need to be first in list
    config_filepaths = [config_dir / '_project_defaults.ini']

    # Specific settings for the subject if one is specified
    if(config_filename is not None):
        config_filepath = config_dir / config_filename
        if not os.path.exists(config_filepath):
            raise Exception(f"Config file specified does not exist: {config_filepath}")
        config_filepaths.append(config_filepath)

    # Local overrule settings
    config_filepaths.append(config_dir / 'local_overrule.ini')

    return config_filepaths

def load_images(load_testsample_images: bool = False):

    ##### Init #####   
    # Use different setting depending if testsample or all images
    if load_testsample_images:
        output_image_dir=conf.dirs.getpath('predictsample_image_input_dir')

        # Use the same image size as for the training, that is the most 
        # convenient to check the quality
        image_pixel_width = conf.train.getint('image_pixel_width')
        image_pixel_height = conf.train.getint('image_pixel_height')
        image_pixel_x_size = conf.train.getfloat('image_pixel_x_size')
        image_pixel_y_size = conf.train.getfloat('image_pixel_y_size')
        image_pixels_overlap = 0
        image_format = ows_util.FORMAT_JPEG
        
        # To create the testsample, fetch only on every ... images
        column_start = 1
        nb_images_to_skip = 50
        
    else:
        output_image_dir=conf.dirs.getpath('predict_image_input_dir')
        
        # Get the image size for the predict
        image_pixel_width = conf.predict.getint('image_pixel_width')
        image_pixel_height = conf.predict.getint('image_pixel_height')
        image_pixel_x_size = conf.predict.getfloat('image_pixel_x_size')
        image_pixel_y_size = conf.predict.getfloat('image_pixel_y_size')
        image_pixels_overlap = conf.predict.getint('image_pixels_overlap')
        image_format = ows_util.FORMAT_JPEG
        
        # For the real prediction dataset, no skipping obviously...
        column_start = 0
        nb_images_to_skip = 0
    
    predict_layer = conf.predict['image_layer']
    wms_server_url = conf.image_layers[predict_layer]['wms_server_url']
    wms_version = conf.image_layers[predict_layer]['wms_version']
    wms_layernames = conf.image_layers[predict_layer]['wms_layernames']
    wms_layerstyles = conf.image_layers[predict_layer]['wms_layerstyles']
    nb_concurrent_calls = conf.image_layers[predict_layer]['nb_concurrent_calls']
    random_sleep = conf.image_layers[predict_layer]['random_sleep']
    projection = conf.image_layers[predict_layer]['projection']
    bbox = conf.image_layers[predict_layer]['bbox']
    grid_xmin = conf.image_layers[predict_layer]['grid_xmin']
    grid_ymin = conf.image_layers[predict_layer]['grid_ymin']
    image_pixels_ignore_border = conf.image_layers[predict_layer]['image_pixels_ignore_border']
    roi_filepath = conf.image_layers[predict_layer]['roi_filepath']

    ows_util.get_images_for_grid(
            wms_server_url=wms_server_url,
            wms_version=wms_version,
            wms_layernames=wms_layernames,
            wms_layerstyles=wms_layerstyles,
            srs=projection,
            output_image_dir=output_image_dir,
            image_gen_bounds=bbox,
            image_gen_roi_filepath=roi_filepath,
            grid_xmin=grid_xmin,
            grid_ymin=grid_ymin,
            image_srs_pixel_x_size=image_pixel_x_size,
            image_srs_pixel_y_size=image_pixel_y_size,
            image_pixel_width=image_pixel_width,
            image_pixel_height=image_pixel_height,
            image_pixels_ignore_border=image_pixels_ignore_border,
            nb_concurrent_calls=nb_concurrent_calls,
            random_sleep=random_sleep,
            image_format=image_format,
            pixels_overlap=image_pixels_overlap,
            column_start=column_start,
            nb_images_to_skip=nb_images_to_skip)

def postprocess_predictions():
    
    ##### Init #####
    # Input dir = the "most recent" prediction result dir for this subject 
    prediction_basedir = Path(f"{conf.dirs['predict_image_output_basedir']}_{conf.general['segment_subject']}_")
    prediction_dirs = sorted(prediction_basedir.parent.glob(f"{prediction_basedir.name}*/"), reverse=True)
    input_dir = prediction_dirs[0]
	
    # Format output dir, partly based on input dir
    # Remove first 2 field from the input dir to get the model info
    input_dir_splitted = input_dir.name.split('_')
    model_info = []
    for i, input_dir_field in enumerate(input_dir_splitted):
        if i >= 2:
            model_info.append(input_dir_field)
    
    output_dir = conf.dirs.getpath('output_vector_dir') / conf.predict['image_layer']
    output_vector_name = f"{'_'.join(model_info)}_{conf.predict['image_layer']}"
    output_filepath = output_dir / f"{output_vector_name}.gpkg"
    
    ##### Go! #####
    border_pixels_to_ignore = conf.predict.getint('image_pixels_overlap')
    postp.postprocess_predictions(
            input_dir=input_dir,
            output_filepath=output_filepath,
            input_ext='.tif',
            border_pixels_to_ignore=border_pixels_to_ignore,
            evaluate_mode=False,
            force=True)

def run_prediction():
    """
    Run a prediction of the input dir given.
    """
    
    # TODO: add something to delete old data, predictions???

    # Create base filename of model to use
    # TODO: is force data version the most logical, or rather implement 
    #       force weights file or ?
    force_model_traindata_version = conf.train.getint('force_model_traindata_version')
    if force_model_traindata_version > -1:
        traindata_version = force_model_traindata_version 
    else:
        traindata_version = mh.get_max_data_version(model_dir=conf.dirs.getpath('model_dir'))
        #logger.info(f"max model_traindata_version found: {model_traindata_version}")
    
    # Get the best model that already exists for this train dataset
    model_architecture = conf.model['architecture']
    hyperparams_version = conf.train.getint('hyperparams_version')
    best_model = mh.get_best_model(
            model_dir=conf.dirs.getpath('model_dir'), 
            segment_subject=conf.general['segment_subject'],
            traindata_version=traindata_version,            
            model_architecture=model_architecture,
            hyperparams_version=hyperparams_version)
    
    # Check if a model was found
    if best_model is False:
        message = f"No model found in model_dir: {conf.dirs.getpath('model_dir')} for traindata_version: {traindata_version}"
        logger.critical(message)
        raise Exception(message)
    else:    
        model_weights_filepath = best_model['filepath']
        logger.info(f"Best model found: {model_weights_filepath}")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir = f"{best_model['basefilename']}_{best_model['epoch']}"
    
    # Try optimizing model with tensorrt
    try:
        # Try import
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        # Import didn't fail, so optimize model
        logger.info('Tensorrt is available, so use optimized model')
        savedmodel_optim_dir = best_model['filepath'].parent / best_model['filepath'].stem + "_optim"
        if not savedmodel_optim_dir.exists():
            # If base model not yet in savedmodel format
            savedmodel_dir = best_model['filepath'].parent / best_model['filepath'].stem
            if not savedmodel_dir.exists():
                logger.info(f"SavedModel format not yet available, so load model + weights from {best_model['filepath']}")
                model = mf.load_model(best_model['filepath'], compile=False)
                logger.info(f"Now save again as savedmodel to {savedmodel_dir}")
                tf.saved_model.save(model, str(savedmodel_dir))
                del model

            # Now optimize model
            logger.info(f"Optimize + save model to {savedmodel_optim_dir}")
            converter = trt.TrtGraphConverterV2(
                    input_saved_model_dir=savedmodel_dir,
                    is_dynamic_op=True,
                    precision_mode='FP16')
            converter.convert()
            converter.save(savedmodel_optim_dir)
        
        logger.info(f"Load optimized model + weights from {savedmodel_optim_dir}")
        model = tf.keras.models.load_model(savedmodel_optim_dir)

    except ImportError as e:
        logger.info('Tensorrt is not available, so load unoptimized model')
        model = mf.load_model(best_model['filepath'], compile=False)

    # Prepare the model for predicting
    nb_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    batch_size = conf.predict.getint('batch_size')
    # TODO: because of bug in tensorflow 1.14, multi GPU doesn't work (this way), 
    # so always use one
    if nb_gpu <= 1:
        model_for_predict = model
        logger.info(f"Predict using single GPU or CPU, with nb_gpu: {nb_gpu}")
    else:
        # If multiple GPU's available, create multi_gpu_model
        try:
            model_for_predict = kr.utils.multi_gpu_model(model, gpus=nb_gpu, cpu_relocation=True)
            logger.info(f"Predict using multiple GPUs: {nb_gpu}, batch size becomes: {batch_size*nb_gpu}")
            batch_size *= nb_gpu
        except ValueError:
            logger.info("Predict using single GPU or CPU")
            model_for_predict = model

    # Predict for entire dataset
    image_layer = conf.image_layers[conf.predict['image_layer']]
    predict_output_dir = Path(f"{conf.dirs['predict_image_output_basedir']}_{predict_out_subdir}")
    predicter.predict_dir(
            model=model_for_predict,
            input_image_dir=conf.dirs.getpath('predict_image_input_dir'),
            output_base_dir=predict_output_dir,
            border_pixels_to_ignore=conf.predict.getint('image_pixels_overlap'),
            projection_if_missing=image_layer['projection'],
            input_mask_dir=None,
            batch_size=batch_size,
            evaluate_mode=False)

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
                predicter.predict_dir(
                        model=model,
                        input_image_dir=traindata_dir / 'image',
                        output_base_dir=traindata_dir / predict_out_subdir,
                        projection_if_missing=train_projection,
                        input_mask_dir=traindata_dir / 'mask',
                        batch_size=conf.train.getint('batch_size_predict'), 
                        evaluate_mode=True)
                    
                # Predict validation dataset
                predicter.predict_dir(
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
                
        trainer.train(
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
    predicter.predict_dir(
            model=model,
            input_image_dir=traindata_dir / 'image',
            output_base_dir=traindata_dir / predict_out_subdir,
            projection_if_missing=train_projection,
            input_mask_dir=traindata_dir / 'mask',
            batch_size=conf.train.getint('batch_size_predict'), 
            evaluate_mode=True)
    
    # Predict validation dataset
    predicter.predict_dir(
            model=model,
            input_image_dir=validationdata_dir / 'image',
            output_base_dir=validationdata_dir / predict_out_subdir,
            projection_if_missing=train_projection,
            input_mask_dir=validationdata_dir / 'mask',
            batch_size=conf.train.getint('batch_size_predict'), 
            evaluate_mode=True)

    # Predict test dataset, if it exists
    if testdata_dir is not None and testdata_dir.exists():
        predicter.predict_dir(
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
        predicter.predict_dir(
                model=model,
                input_image_dir=conf.dirs.getpath('predictsample_image_input_dir'),
                output_base_dir=conf.dirs.getpath('predictsample_image_output_basedir') / predict_out_subdir,
                projection_if_missing=train_projection,
                batch_size=conf.train.getint('batch_size_predict'), 
                evaluate_mode=True)
    
if __name__ == '__main__':
    orthoseg_args(sys.argv[1:])

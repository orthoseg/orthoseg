# -*- coding: utf-8 -*-
"""
High-level API to run a segmentation.

@author: Pieter Roggemans
"""

import os
import configparser

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import keras as kr

import log_helper
import segment
import vector.vector_helper as vh
import models.model_helper as mh

def run_prediction(segment_config_filepath: str, 
                   force_model_traindata_version: int = None):
    """
    Run a prediction of the input dir given.
    
    Args
        segment_config_filepath: config(file) to use for the segmentation
        force_model_traindata_version: force version of the train data you want 
                to use the weights from to load in the model
    """
    # Read the configuration
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read(segment_config_filepath)
    settings = conf['settings']
    
    # Run the prediction
    run_prediction2(segment_subject=settings['segment_subject'],
                    input_image_dir=settings['predict_image_dir'],
                    input_projection=settings['input_projection'],
                    border_pixels_to_ignore=int(settings['border_pixels_to_ignore']),
                    model_dir=settings['model_dir'],
                    model_architecture=settings['model_architecture'],
                    model_json_filepath=settings['model_json_filepath'],
                    batch_size_predict=int(settings['batch_size_predict']),
                    log_dir=settings['log_dir'],
                    force_model_traindata_version=force_model_traindata_version)
    
def run_prediction2(segment_subject: str,
                    input_image_dir: str,
                    input_projection: str,                    
                    border_pixels_to_ignore: int,
                    model_dir: str,
                    model_architecture: str,
                    model_json_filepath: str,
                    batch_size_predict: int,
                    log_dir: str,
                    force_model_traindata_version: int = None):
    
    """
    Run a prediction of the input dir given.
    
    Args
        segment_subject: subject you want to segment (eg. "greenhouses")
        input_image_dir: dir where the images you want to predict on are
        input_projection: projection of the input images. If None, the 
                projection encoded in the images is used.
        border_pixels_to_ignore: number of pixels at the border of the input 
                images to ignore               
        model_dir: dir where the models are found
        model_architecture: encoder + decoder of the neural network to use
        model_json_filepath: filename of the json file containing the model definition
        batch_size_predict: batch size to use while predicting. This must be 
                choosen depending on the neural network architecture
                and available memory on you GPU.
        log_dir: dir where logging is written to
        force_model_traindata_version: force version of the train data you 
                want to use the weights from to load in the model
    """    
    # The real work...
    # -------------------------------------------------------------------------    
    
    # Main initialisation of the logging
    logger = log_helper.main_log_init(log_dir, __name__)

    # Create base filename of model to use
    # TODO: is force data version the most logical, or rather implement 
    #       force weights file?
    if force_model_traindata_version is not None:
        model_traindata_version = force_model_traindata_version 
    else:
        model_traindata_version = mh.get_max_data_version(
                model_dir=conf.dirs['model_dir'])
        #logger.info(f"max model_traindata_version found: {model_traindata_version}")
    
    model_base_filename = mh.model_base_filename(segment_subject, 
                                                 model_traindata_version, 
                                                 model_architecture)

    # Get the best model that already exists for this train dataset
    best_model = mh.get_best_model(model_dir=model_dir,
                                   model_base_filename=model_base_filename)
    model_weights_filepath = best_model['filepath']
    logger.info(f"Best model found: {best_model['filename']}")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir = os.path.splitext(best_model['filename'])[0]
    
    # Load prediction model...
    logger.info(f"Load model from {model_json_filepath}")
    with open(model_json_filepath, 'r') as src:
        model_json = src.read()
        model = kr.models.model_from_json(model_json)
    logger.info(f"Load weights from {model_weights_filepath}")                
    model.load_weights(model_weights_filepath)
    logger.info("Model weights loaded")
    
    # Predict for entire dataset
    output_base_dir = f"{input_image_dir}_{predict_out_subdir}"
    segment.predict_dir(model=model,
                        input_image_dir=input_image_dir,
                        output_base_dir=output_base_dir,
                        border_pixels_to_ignore=border_pixels_to_ignore,
                        projection_if_missing=input_projection,
                        input_mask_dir=None,
                        batch_size=batch_size_predict,
                        evaluate_mode=False)
    
    # Now postprocess the vector results, so the end result is one big file
    vh.postprocess_vectors(base_dir=output_base_dir,
                           evaluate_mode=False,
                           force=False)
    
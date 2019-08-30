# -*- coding: utf-8 -*-
"""
High-level API to run a segmentation.
"""

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import keras as kr

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper
import orthoseg.model.model_helper as mh
import orthoseg.segment as segment
import orthoseg.postprocess_predictions as postp

def run_prediction(config_filepaths: str):
    """
    Run a prediction of the input dir given.
    
    Args
        config_filepaths: config files to use for the segmentation
    """
    
    # TODO: add something to delete old data, predictions???
    # Read the configuration
    conf.read_config(config_filepaths)
    
    # Main initialisation of the logging
    logger = log_helper.main_log_init(conf.dirs['log_dir'], __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Create base filename of model to use
    # TODO: is force data version the most logical, or rather implement 
    #       force weights file or ?
    force_model_traindata_version = conf.model.getint('force_model_traindata_version')
    if force_model_traindata_version > -1:
        model_traindata_version = force_model_traindata_version 
    else:
        model_traindata_version = mh.get_max_data_version(model_dir=conf.dirs['model_dir'])
        #logger.info(f"max model_traindata_version found: {model_traindata_version}")
    
    model_base_filename = mh.format_model_base_filename(
            conf.general['segment_subject'], model_traindata_version, 
            conf.model['architecture'])

    # Get the best model that already exists for this train dataset
    best_model = mh.get_best_model(model_dir=conf.dirs['model_dir'],
                                   model_base_filename=model_base_filename)
    
    # Check if a model was found
    if best_model is None:
        message = f"No model found in model_dir: {conf.dirs['model_dir']} for model_base_filename: {model_base_filename}"
        logger.critical(message)
        raise Exception(message)
    else:    
        model_weights_filepath = best_model['filepath']
        logger.info(f"Best model found: {model_weights_filepath}")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir = f"{best_model['segment_subject']}_{best_model['train_data_version']}_{best_model['model_architecture']}_{best_model['epoch']}"
    
    # Load prediction model...
    logger.info(f"Load model from {conf.files['model_json_filepath']}")
    with open(conf.files['model_json_filepath'], 'r') as src:
        model_json = src.read()
        model = kr.models.model_from_json(model_json)
    logger.info(f"Load weights from {model_weights_filepath}")                
    model.load_weights(model_weights_filepath)
    logger.info("Model weights loaded")

    # Predict for entire dataset
    image_datasource = conf.image_datasources[conf.predict['image_datasource_code']]
    predict_output_dir = f"{conf.dirs['predict_image_output_basedir']}_{predict_out_subdir}"
    segment.predict_dir(model=model,
                        input_image_dir=conf.dirs['predict_image_input_dir'],
                        output_base_dir=predict_output_dir,
                        border_pixels_to_ignore=int(conf.predict['image_pixels_overlap']),
                        projection_if_missing=image_datasource['projection'],
                        input_mask_dir=None,
                        batch_size=int(conf.predict['batch_size']),
                        evaluate_mode=False)

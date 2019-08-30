# -*- coding: utf-8 -*-
"""
Module to make it easy to start a training session.
"""

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras as kr

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper
import orthoseg.model.model_helper as mh
import orthoseg.prepare_traindatasets as prep
from orthoseg import segment_predict
from orthoseg import segment_train

def run_training_session(config_filepaths: []):
    """
    Run a training session.
    
    Args
        config_filepaths: config(file) to use for the segmentation
    """
    ##### Init #####
    # TODO: add something to delete old data, predictions???
    
    # Read the configuration
    conf.read_config(config_filepaths)
    
    # Main initialisation of the logging
    logger = log_helper.main_log_init(conf.dirs['log_training_dir'], __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")
    
    # First check if the segment_subject has a valid name
    segment_subject = conf.general['segment_subject']
    if segment_subject == 'MUST_OVERRIDE':
        raise Exception("The segment_subject parameter needs to be overridden in the subject specific config file!!!") 
    elif '_' in segment_subject:
        raise Exception(f"The segment_subject parameter should not contain '_', so this is invalid: {segment_subject}!!!") 

    # Create the output dir's if they don't exist yet...
    for dir in [conf.dirs['project_dir'], conf.dirs['training_dir']]:
        if dir and not os.path.exists(dir):
            os.mkdir(dir)
    
    # If the training data doesn't exist yet, create it
    # Get the image datasource section to use for training
    train_image_datasource_code = conf.train['image_datasource_code']
    train_projection = conf.image_datasources[train_image_datasource_code]['projection']
    
    # First the "train" training dataset
    force_model_traindata_version = conf.model.getint('force_model_traindata_version')
    if force_model_traindata_version > -1:
        traindata_dir = f"{conf.dirs['training_train_basedir']}_{force_model_traindata_version:02d}"
        traindata_version = force_model_traindata_version            
    else:
        logger.info("Prepare train, validation and test data")
        traindata_dir, traindata_version = prep.prepare_traindatasets(
                input_vector_label_filepath=conf.files['input_trainlabels_filepath'],
                image_datasources=conf.image_datasources,
                default_image_datasource_code=train_image_datasource_code,
                output_basedir=conf.dirs['training_train_basedir'],
                image_subdir=conf.dirs['image_subdir'],
                mask_subdir=conf.dirs['mask_subdir'])
    logger.info(f"Traindata dir to use is {traindata_dir}, with traindata_version: {traindata_version}")

    # Now the "validation" training dataset
    validationdata_dir, _ = prep.prepare_traindatasets(
            input_vector_label_filepath=conf.files['input_validationlabels_filepath'],
            image_datasources=conf.image_datasources,
            default_image_datasource_code=train_image_datasource_code,
            output_basedir=conf.dirs['training_validation_basedir'],
            image_subdir=conf.dirs['image_subdir'],
            mask_subdir=conf.dirs['mask_subdir'])

    # Now the "test" training dataset
    if os.path.exists(conf.files['input_testlabels_filepath']):
        testdata_dir, _ = prep.prepare_traindatasets(
                input_vector_label_filepath=conf.files['input_testlabels_filepath'],
                image_datasources=conf.image_datasources,
                default_image_datasource_code=train_image_datasource_code,
                output_basedir=conf.dirs['training_test_basedir'],
                image_subdir=conf.dirs['image_subdir'],
                mask_subdir=conf.dirs['mask_subdir'])
    else:
        testdata_dir = None
        logger.warn(f"Testlabels input file doesn't exist: {conf.files['input_testlabels_filepath']}")

    # Create base filename of model to use
    model_base_filename = mh.format_model_base_filename(
            conf.general['segment_subject'], traindata_version, conf.model['architecture'])
    logger.debug(f"model_base_filename: {model_base_filename}")
    
    # Get the best model that already exists for this train dataset
    best_model = mh.get_best_model(model_dir=conf.dirs['model_dir'],
                                   model_base_filename=model_base_filename)

    # Check if training is needed
    resume_train = conf.model.getboolean('resume_train')
    if resume_train is False:
        # If no (best) model found, training needed!
        if best_model is None:
            train_needed = True
        else:
            logger.info("JUST PREDICT, without training: preload_existing_model is false and model found")
            train_needed = False
    else:
        # We want to preload an existing model and models were found
        if best_model is not None:            
            logger.info(f"PRELOAD model and continue TRAINING it: {best_model['filename']}")
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
        best_model_curr = mh.get_best_model(model_dir=conf.dirs['model_dir'])
        if best_model_curr is not None:
            # Load prediction model...
            model_json_filepath = conf.files['model_json_filepath']
            logger.info(f"Load model from {model_json_filepath}")
            with open(model_json_filepath, 'r') as src:
                model_json = src.read()
                model = kr.models.model_from_json(model_json)
            logger.info(f"Load weights from {best_model_curr['filepath']}")                
            model.load_weights(best_model_curr['filepath'])
            logger.info("Model weights loaded")

            # Prepare output subdir to be used for predictions
            predict_out_subdir = os.path.splitext(best_model_curr['filename'])[0]
            
            # Predict training dataset
            
            segment_predict.predict_dir(
                    model=model,
                    input_image_dir=os.path.join(traindata_dir, conf.dirs['image_subdir']),
                    output_base_dir=os.path.join(traindata_dir, predict_out_subdir),
                    projection_if_missing=train_projection,
                    input_mask_dir=os.path.join(traindata_dir, conf.dirs['mask_subdir']),
                    batch_size=int(conf.train['batch_size_predict']), 
                    evaluate_mode=True)
                
            # Predict validation dataset
            segment_predict.predict_dir(
                    model=model,
                    input_image_dir=os.path.join(validationdata_dir, conf.dirs['image_subdir']),
                    output_base_dir=os.path.join(validationdata_dir, predict_out_subdir),
                    projection_if_missing=train_projection,
                    input_mask_dir=os.path.join(validationdata_dir, conf.dirs['mask_subdir']),
                    batch_size=int(conf.train['batch_size_predict']), 
                    evaluate_mode=True)
            del model
        
        # Now we can really start training
        logger.info('Start training')
        model_preload_filepath = None
        if best_model is not None:
            model_preload_filepath = best_model['filepath']
        segment_train.train(
                traindata_dir=traindata_dir,
                validationdata_dir=validationdata_dir,
                image_subdir=conf.dirs['image_subdir'], 
                mask_subdir=conf.dirs['mask_subdir'],
                model_encoder=conf.model['encoder'], 
                model_decoder=conf.model['decoder'],
                model_save_dir=conf.dirs['model_dir'],
                model_save_base_filename=model_base_filename,
                model_preload_filepath=model_preload_filepath,
                batch_size=int(conf.train['batch_size_fit']), 
                nb_epoch=int(conf.train['max_epoch'])) 
    
    # If we trained, get the new best model
    if train_needed is True:
        best_model = mh.get_best_model(model_dir=conf.dirs['model_dir'],
                                       model_base_filename=model_base_filename)
    logger.info(f"PREDICT test data with best model: {best_model['filename']}")
    
    # Load prediction model...
    model_json_filepath = conf.files['model_json_filepath']
    logger.info(f"Load model from {model_json_filepath}")
    with open(model_json_filepath, 'r') as src:
        model_json = src.read()
        model = kr.models.model_from_json(model_json)
    logger.info(f"Load weights from {best_model['filepath']}")                
    model.load_weights(best_model['filepath'])
    logger.info("Model weights loaded")

    # Prepare output subdir to be used for predictions
    predict_out_subdir = os.path.splitext(best_model['filename'])[0]
    
    # Predict training dataset
    segment_predict.predict_dir(
            model=model,
            input_image_dir=os.path.join(traindata_dir, conf.dirs['image_subdir']),
            output_base_dir=os.path.join(traindata_dir, predict_out_subdir),
            projection_if_missing=train_projection,
            input_mask_dir=os.path.join(traindata_dir, conf.dirs['mask_subdir']),
            batch_size=int(conf.train['batch_size_predict']), 
            evaluate_mode=True)
    
    # Predict validation dataset
    segment_predict.predict_dir(
            model=model,
            input_image_dir=os.path.join(validationdata_dir, conf.dirs['image_subdir']),
            output_base_dir=os.path.join(validationdata_dir, predict_out_subdir),
            projection_if_missing=train_projection,
            input_mask_dir=os.path.join(validationdata_dir, conf.dirs['mask_subdir']),
            batch_size=int(conf.train['batch_size_predict']), 
            evaluate_mode=True)

    # Predict test dataset, if it exists
    if testdata_dir is not None and os.path.exists(testdata_dir):
        segment_predict.predict_dir(
                model=model,
                input_image_dir=os.path.join(testdata_dir, conf.dirs['image_subdir']),
                output_base_dir=os.path.join(testdata_dir, predict_out_subdir),                        
                projection_if_missing=train_projection,
                input_mask_dir=os.path.join(testdata_dir, conf.dirs['mask_subdir']),
                batch_size=int(conf.train['batch_size_predict']), 
                evaluate_mode=True)
    
    # Predict extra test dataset with random images in the roi, to add to 
    # train and/or validation dataset if inaccuracies are found
    # -> this is very useful to find false positives to improve the datasets
    if os.path.exists(conf.dirs['predictsample_image_input_dir']):
        segment_predict.predict_dir(
                model=model,
                input_image_dir=conf.dirs['predictsample_image_input_dir'],
                output_base_dir=(conf.dirs['predictsample_image_output_basedir'] + predict_out_subdir),
                projection_if_missing=train_projection,
                batch_size=int(conf.train['batch_size_predict']), 
                evaluate_mode=True)
    
    # Release the memory from the GPU... 
    # TODO: doesn't work!!!
    kr.backend.clear_session()
    del model
    
    import gc
    for _ in range(20):
        #print(gc.collect())
        gc.collect()

    # TODO: send email if training ready...
    
if __name__ == '__main__':
    None
    

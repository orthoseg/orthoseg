# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:46:44 2019

@author: pierog
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras as kr

import log_helper
import segment as seg
import prepare_traindatasets as prep
import models.model_helper as mh

def run_training_session(segment_subject: str,
                         base_dir: str,
                         wms_server_url: str,
                         wms_server_layer: str,
                         model_encoder: str,
                         model_decoder: str,
                         batch_size_train: int,
                         batch_size_pred: int,
                         nb_epoch: int = 1000,
                         force_traindata_version: int = None,
                         resume_train: bool = False):
    """
    The batch size to use depends on the model architecture, the size of the 
    training images and the available (GPU) memory
    
    Args
        segment_subject: subject you want to segment (eg. "greenhouses")
        base_dir: base dir where you want to put your segmentation projects in
        wms_server_url: url to WMS server you want to use
        wms_server_layer: layer of the WMS server you want to use
        model_encoder: encoder of the neural network to use
        model_decoder: decoder of the neural network to use
        batch_size_train: batch size to use while training. This must be 
                          choosen depending on the neural network architecture
                          and available memory on you GPU.
        batch_size_pred: batch size to use while predicting
        nb_epoch: maximum number of epochs to train
        force_traindata_version: specify version nb. of the traindata to use
        resume_train: use the best existing model as basis to continue training
    """
    
    # Init different dir name for the segementation project
    project_dir = os.path.join(base_dir, segment_subject)       
    input_labels_dir = os.path.join(project_dir, 'input_labels')
    model_dir = os.path.join(project_dir, "models")
    training_dir = os.path.join(project_dir, "training")
            
    # Main initialisation of the logging
    log_dir = os.path.join(training_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)    
    
    # Create the output dir's if they don't exist yet...
    for dir in [project_dir, training_dir, log_dir]:
        if dir and not os.path.exists(dir):
            os.mkdir(dir)

    # Subdirs to use to put images versus masks
    image_subdir = 'image'
    mask_subdir = 'mask'
    
    # If the training data doesn't exist yet, create it    
    # First the "train" training dataset
    traindata_basedir = os.path.join(training_dir, "train")
    if force_traindata_version is None:
        logger.info("Prepare train and validation data")
        input_labels_filename = f"{segment_subject}_trainlabels.shp"
        input_labels_filepath = os.path.join(input_labels_dir, 
                                             input_labels_filename)
        traindata_dir, traindata_version = prep.prepare_traindatasets(
                    input_vector_label_filepath=input_labels_filepath,
                    wms_server_url=wms_server_url,
                    wms_server_layer=wms_server_layer,
                    output_basedir=traindata_basedir,
                    image_subdir=image_subdir,
                    mask_subdir=mask_subdir)
    else:
        traindata_dir = f"{traindata_basedir}_{force_traindata_version:02d}"
        traindata_version = force_traindata_version            
    logger.info(f"Traindata dir to use is {traindata_dir}, with traindata_version: {traindata_version}")

    # Now the "validation" training dataset
    input_labels_filename = f"{segment_subject}_validationlabels.shp"
    input_labels_filepath = os.path.join(input_labels_dir, 
                                         input_labels_filename)
    validationdata_dir, tmp = prep.prepare_traindatasets(
                input_vector_label_filepath=input_labels_filepath,
                wms_server_url=wms_server_url,
                wms_server_layer=wms_server_layer,
                output_basedir=os.path.join(training_dir, "validation"),
                image_subdir=image_subdir,
                mask_subdir=mask_subdir)

    # Now the "validation" training dataset
    logger.info("Prepare train and validation data")
    input_labels_filename = f"{segment_subject}_testlabels.shp"
    input_labels_filepath = os.path.join(input_labels_dir, 
                                         input_labels_filename)
    testdata_dir, tmp = prep.prepare_traindatasets(
                input_vector_label_filepath=input_labels_filepath,
                wms_server_url=wms_server_url,
                wms_server_layer=wms_server_layer,
                output_basedir=os.path.join(training_dir, "test"),
                image_subdir=image_subdir,
                mask_subdir=mask_subdir)
    
    # Create base filename of model to use
    model_architecture = f"{model_encoder}+{model_decoder}"
    model_base_filename = mh.model_base_filename(
            segment_subject, traindata_version, model_architecture)
    logger.debug(f"model_base_filename: {model_base_filename}")
    
    # Get the best model that already exists for this train dataset
    best_model = mh.get_best_model(model_dir=model_dir,
                                   model_base_filename=model_base_filename)

    # Check if training is needed
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
    
    # Start training
    if train_needed is True:
        logger.info('Start training')
        model_preload_filepath = None
        if best_model is not None:
            model_preload_filepath = best_model['filepath']
        seg.train(traindata_dir=traindata_dir,
                  validationdata_dir=validationdata_dir,
                  image_subdir=image_subdir, mask_subdir=mask_subdir,
                  model_encoder=model_encoder, model_decoder=model_decoder,
                  model_save_dir=model_dir,
                  model_save_base_filename=model_base_filename,
                  model_preload_filepath=model_preload_filepath,
                  batch_size=batch_size_train, nb_epoch=nb_epoch)  
        
    # If we trained, get the new best model
    if train_needed is True:
        best_model = mh.get_best_model(model_dir=model_dir,
                                       model_base_filename=model_base_filename)
    logger.info(f"PREDICT test data with best model: {best_model['filename']}")
    
    # Load prediction model...
    model_json_filepath = f"{model_dir}{os.sep}{model_architecture}.json"
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
    seg.predict_dir(model=model,
                    input_image_dir=os.path.join(traindata_dir, image_subdir),
                    output_base_dir=os.path.join(traindata_dir, predict_out_subdir),
                    input_mask_dir=os.path.join(traindata_dir, mask_subdir),
                    batch_size=batch_size_pred, evaluate_mode=True)
    
    # Predict validation dataset
    seg.predict_dir(model=model,
                    input_image_dir=os.path.join(validationdata_dir, image_subdir),
                    output_base_dir=os.path.join(validationdata_dir, predict_out_subdir),
                    input_mask_dir=os.path.join(validationdata_dir, mask_subdir),
                    batch_size=batch_size_pred, evaluate_mode=True)

    # Predict test dataset
    if os.path.exists(testdata_dir):
        seg.predict_dir(model=model,
                        input_image_dir=os.path.join(testdata_dir, image_subdir),
                        output_base_dir=os.path.join(testdata_dir, predict_out_subdir),
                        input_mask_dir=os.path.join(testdata_dir, mask_subdir),
                        batch_size=batch_size_pred, evaluate_mode=True)
    
    # Predict extra test dataset with random images in the roi, to add to 
    # train and/or validation dataset if inaccuracies are found
    # -> this is very useful to find false positives to improve the datasets
    test_random_dir = os.path.join(training_dir, "test-random")
    if os.path.exists(test_random_dir):
        seg.predict_dir(model=model,
                        input_image_dir=os.path.join(test_random_dir, image_subdir),
                        output_base_dir=os.path.join(test_random_dir, predict_out_subdir),
                        batch_size=batch_size_pred, evaluate_mode=True)

if __name__ == '__main__':
    None
    
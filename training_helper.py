# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:46:44 2019

@author: pierog
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras as kr

import log_helper
import segment
import prepare_traindatasets as prep
import models.model_helper as mh

def run_training_session(segment_subject: str,
                         base_dir: str,
                         wms_server_url: str,
                         model_encoder: str,
                         model_decoder: str,
                         batch_size_train: int,
                         batch_size_pred: int,
                         nb_epoch: int = 1000,
                         preload_existing_model: bool = False):
    """
    The batch size to use depends on the model architecture, the size of the 
    training images and the available (GPU) memory
    """
    
    # Main project dir for this subject
    project_dir = os.path.join(base_dir, segment_subject)       
     
    # Dir where input label data can be found
    input_labels_dir = os.path.join(project_dir, 'input_labels')
    input_labels_filename = f"{segment_subject}_trainlabels.shp"
    input_labels_filepath = os.path.join(input_labels_dir, 
                                         input_labels_filename)
            
    # Dir where models will be saved
    model_dir = os.path.join(project_dir, "models")

    # Dir where all training info will be saved
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
    '''
    # TODO: write something to create train, validation and test dataset 
    # from base input file
    # If max_samples > 0 and less than nb labels, take random max_samples
    if(max_samples and max_samples > 0 
        and max_samples < len(input_labels)):
        labels = random.sample(input_labels, max_samples)
    else:
        labels = input_labels
    '''
    traindata_basedir = os.path.join(training_dir, "train")
    force_create_train_data = False 
    logger.info("Prepare train and validation data")
    traindata_dir, traindata_version = prep.prepare_traindatasets(
                input_vector_label_filepath=input_labels_filepath,
                wms_server_url=wms_server_url,
                wms_server_layer='ofw',
                output_basedir=traindata_basedir,
                image_subdir=image_subdir,
                mask_subdir=mask_subdir,
                force=force_create_train_data)
    logger.info(f"Traindata dir to use is {traindata_dir}, with traindata_version: {traindata_version}")

    # Seperate validation dataset for during training...
    # TODO: create validation data based on input vector file as well...
    validationdata_dir = os.path.join(training_dir, "validation")

    # TODO: automatically create a random test dataset here as well if it 
    # doesnt exist?

    # Create base filename of model to use
    model_architecture = f"{model_encoder}+{model_decoder}"
    model_base_filename = mh.model_base_filename(segment_subject,
                                                 traindata_version,
                                                 model_architecture)
    logger.info(f"model_base_filename: {model_base_filename}")
    
    # Get the best model that already exists for this train dataset
    best_model = mh.get_best_model(model_dir=model_dir,
                                   model_base_filename=model_base_filename)

    # Check if training is needed
    if preload_existing_model is False:
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
        segment.train(traindata_dir=traindata_dir,
                      validationdata_dir=validationdata_dir,
                      image_subdir=image_subdir,
                      mask_subdir=mask_subdir,
                      model_encoder=model_encoder,
                      model_decoder=model_decoder,
                      model_save_dir=model_dir,
                      model_save_base_filename=model_base_filename,
                      model_preload_filepath=model_preload_filepath,
                      batch_size=batch_size_train,
                      nb_epoch=nb_epoch)  
        
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
    segment.predict(model=model,
                    input_image_dir=os.path.join(traindata_dir, image_subdir),
                    output_base_dir=os.path.join(traindata_dir, predict_out_subdir),
                    input_mask_dir=os.path.join(traindata_dir, mask_subdir),
                    batch_size=batch_size_pred,
                    evaluate_mode=True)
    
    # Predict validation dataset
    segment.predict(model=model,
                    input_image_dir=os.path.join(validationdata_dir, image_subdir),
                    output_base_dir=os.path.join(validationdata_dir, predict_out_subdir),
                    input_mask_dir=os.path.join(validationdata_dir, mask_subdir),
                    batch_size=batch_size_pred,
                    evaluate_mode=True)
    
    # Predict extra test dataset with random images in the roi, to add to 
    # train and/or validation dataset if inaccuracies are found
    # -> this is very useful to find false positives to improve the datasets
    test_random_dir = os.path.join(training_dir, "test_random")
    if os.path.exists(test_random_dir):
        segment.predict(model=model,
                        input_image_dir=os.path.join(test_random_dir, image_subdir),
                        output_base_dir=os.path.join(test_random_dir, predict_out_subdir),
                        batch_size=batch_size_pred,
                        evaluate_mode=True)

    # Predict for test dataset with all known input data we have
    test_dir = os.path.join(training_dir, "test")
    if os.path.exists(test_dir):
        segment.predict(model=model,
                        input_image_dir=os.path.join(test_dir, image_subdir),
                        output_base_dir=os.path.join(test_dir, predict_out_subdir),
                        input_mask_dir=os.path.join(test_dir, mask_subdir),
                        batch_size=batch_size_pred,
                        evaluate_mode=True)

if __name__ == '__main__':
    None
    
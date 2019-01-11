# -*- coding: utf-8 -*-
"""
Module to run the prediction for greenhouses.

@author: Pieter Roggemans
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import keras as kr

import log_helper
import segment
import vector.vector_helper as vh
import models.model_helper as mh

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Initialisations specific for the segmentation subject...
    #
    # Remark: they need to be changed for different segmentations
    # -------------------------------------------------------------------------
    # General initialisations for the segmentation project
    segment_subject = "greenhouses"
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"
    project_dir = os.path.join(base_dir, segment_subject)
    
    # Model we will use for this segmentation
    train_data_version = 13
    model_architecture = "inceptionresnetv2+linknet"

    # The batch size to use. Depends on available hardware and model used.
    batch_size_512px = 20
    batch_size_1024px = 12
    
    '''
    batch_size_512px = 1
    batch_size_1024px = 1
    '''
    
    # General initialisations...
    #
    # Remark: they don't need to be changed if implied conventions are OK
    # -------------------------------------------------------------------------    
    # Main initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info(f"Start {segment_subject} prediction script")

    # Model architecture file to use   
    model_json_filepath = f"{project_dir}{os.sep}{model_architecture}.json"

    # Create base filename of model to use
    model_base_filename = mh.model_base_filename(segment_subject,
                                                 train_data_version,
                                                 model_architecture)

    # Get a list of all relevant models
    model_info_df = mh.get_models(model_dir=project_dir,
                                  model_base_filename=model_base_filename)

    # Get the model with the highest combined accuracy
    model_to_use = model_info_df.loc[model_info_df['acc_combined'].values.argmax()]
    model_weights_filepath = model_to_use['filepath']
    logger.info(f"Best model found: {model_to_use['filename']}")
        
    # Train and evaluate settings
    # The subdirs where the images and masks can be found by convention for training and validation
    image_subdir = "image"
    mask_subdir = "mask"
        
    # Seperate validation dataset for during training...
    train_dir = os.path.join(project_dir, "train")
    validation_dir = os.path.join(project_dir, "validation")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir = os.path.splitext(model_to_use['filename'])[0]
    
    # Create the output dir's if they don't exist yet...
    for dir in [project_dir, train_dir, log_dir]:
        if dir and not os.path.exists(dir):
            os.mkdir(dir)
    
    # Load prediction model...
    logger.info(f"Load model from {model_json_filepath}")
    with open(model_json_filepath, 'r') as src:
        model_json = src.read()
        model = kr.models.model_from_json(model_json)
    logger.info(f"Load weights from {model_weights_filepath}")                
    model.load_weights(model_weights_filepath)
    logger.info("Model weights loaded")
    
    '''
    # Predict training dataset
    segment.predict(model=model,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_base_dir=os.path.join(train_dir, predict_out_subdir),
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    batch_size=batch_size_512px,
                    evaluate_mode=True)
    
    # Predict validation dataset
    segment.predict(model=model,
                    input_image_dir=os.path.join(validation_dir, image_subdir),
                    output_base_dir=os.path.join(validation_dir, predict_out_subdir),
                    input_mask_dir=os.path.join(validation_dir, mask_subdir),
                    batch_size=batch_size_512px,
                    evaluate_mode=True)
    
    # Predict extra test dataset with random images in the roi, to add to 
    # TRAIN dataset if inaccuracies are found
    # -> this is very useful to find false positives to improve the 
    #    current train dataset
    test_for_train_dir = os.path.join(project_dir, "test_for_train")
    test_for_train_input_dir = os.path.join(test_for_train_dir, "image")
    segment.predict(model=model,
                    input_image_dir=test_for_train_input_dir,
                    output_base_dir=f"{test_for_train_input_dir}_{predict_out_subdir}",
                    batch_size=batch_size_512px,
                    evaluate_mode=True)
    
    # Predict extra test dataset with random images in the roi, to add to 
    # VALIDATION dataset if inaccuracies are found
    # -> this is very useful to find false positives to improve the 
    #    current validation dataset
    test_for_validation_dir = os.path.join(project_dir, "test_for_validation")
    test_for_validation_input_dir = os.path.join(test_for_validation_dir, "image")
    segment.predict(model=model,
                    input_image_dir=test_for_validation_input_dir,
                    output_base_dir=f"{test_for_validation_input_dir}_{predict_out_subdir}",
                    batch_size=batch_size_512px,
                    evaluate_mode=True)
    '''
    # Predict for test dataset with all known input data we have
    test_all_dir = os.path.join(project_dir, "test_all_known_input")
    '''
    if not os.path.exists(test_all_dir):
        logger.info('Prepare test_dir data')
        prep.prepare_training_data(
                input_vector_label_filepath=input_labels_filepath,
                wms_server_url=WMS_SERVER_URL,
                wms_server_layer='ofw',
                output_image_dir=os.path.join(test_all_dir, image_subdir),
                output_mask_dir=os.path.join(test_all_dir, mask_subdir),
                image_pixel_width=512,
                image_pixel_height=512,
                max_samples=None,
                force=False)
    '''
    '''    
    segment.predict(model=model,
                    input_image_dir=os.path.join(test_all_dir, image_subdir),
                    output_base_dir=os.path.join(test_all_dir, predict_out_subdir),
                    input_mask_dir=os.path.join(test_all_dir, mask_subdir),
                    batch_size=batch_size_512px,
                    evaluate_mode=True)
    '''
    
    # Predict for entire dataset
    image_pixel_width = 1024
    image_pixel_height = image_pixel_width
    border_pixels_to_ignore = 64
    pixels_overlap = border_pixels_to_ignore

    to_predict_input_dir = f"X:\\GIS\\GIS DATA\\_SegmentCache\\Ortho_2018\\{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"
    #to_predict_input_dir = f"X:\\GIS\\GIS DATA\\_Tmp\\Ortho_2018_autosegment_cache\\{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"
    #to_predict_input_dir = f"Y:\\Image_data\\Ortho2018\\{image_pixel_width}x{image_pixel_height}_{border_pixels_to_ignore}pxOverlap"
    
    if to_predict_input_dir:
        
        segment.predict(model=model,
                        input_image_dir=to_predict_input_dir,
                        output_base_dir=f"{to_predict_input_dir}_{predict_out_subdir}",
                        border_pixels_to_ignore=border_pixels_to_ignore,
                        input_mask_dir=None,
                        batch_size=batch_size_1024px,
                        evaluate_mode=False)
        
        vh.union_vectors(base_dir=f"{to_predict_input_dir}_{predict_out_subdir}",
                         evaluate_mode=False,
                         force=False)
    
if __name__ == '__main__':
    main()

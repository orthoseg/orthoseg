# -*- coding: utf-8 -*-
"""
Module to run the segmentation of horse tracks.

@author: Pieter Roggemans
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import keras as kr

import log_helper
import segment
import preprocess as prep
import vector_helper as vh

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():

    # Initialisations specific for the segmentation subject...
    #
    # Remark: they need to be changed for different segmentations
    # -------------------------------------------------------------------------
    # General initialisations for the segmentation project
    segment_subject = "horsetracks"
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"
    project_dir = os.path.join(base_dir, segment_subject)
    
    # WMS server we can use to get the image data
    WMS_SERVER_URL = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'
    
    # Model we will use for this segmentation
    model_architecture = "inceptionresnetv2+linknet"
#    model_weights_to_use_base = "107_0.95300_0.72116"
#    model_weights_to_use_base = "107_0.94359_0.73477"
#    model_weights_to_use_base = "0.92715_0.92715_0.66716_128"
#    model_weights_to_use_base = "0.93287_0.93287_0.73276_158"
#    model_weights_to_use_base = "0.76182_0.92349_0.60016_123"
#    model_weights_to_use = "horsetrack_08_inceptionresnetv2+linknet_0.83188_0.91937_0.74439_103"
#    model_weights_to_use = "horsetracks_09_inceptionresnetv2+linknet_0.87258_0.92380_0.82135_258"
    model_weights_to_use = "horsetracks_10_inceptionresnetv2+linknet_0.88096_0.92985_0.83206_149"

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
    # Model and weights to use   
    model_json_filepath = f"{project_dir}{os.sep}{model_architecture}.json"
    model_weights_filepath = f"{project_dir}{os.sep}{model_weights_to_use}.hdf5"
    
    # Main initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info(f"Start {segment_subject} prediction script")

    # Input label data
    input_labels_dir = os.path.join(project_dir, 'input_labels')
    input_labels_filename = f"{segment_subject}_groundtruth.geojson"
    input_labels_filepath = os.path.join(input_labels_dir, 
                                         input_labels_filename)
    
    # Train and evaluate settings
    # The subdirs where the images and masks can be found by convention for training and validation
    image_subdir = "image"
    mask_subdir = "mask"
        
    # Seperate validation dataset for during training...
    train_dir = os.path.join(project_dir, "train")
    validation_dir = os.path.join(project_dir, "validation")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir = f"{model_weights_to_use}"
    
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
    
    # Predict for test dataset with all known input data we have
    '''
    test_all_dir = os.path.join(project_dir, "test_all_known_input")
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
        '''
        segment.predict(model=model,
                        input_image_dir=to_predict_input_dir,
                        output_base_dir=f"{to_predict_input_dir}_{predict_out_subdir}",
                        border_pixels_to_ignore=border_pixels_to_ignore,
                        input_mask_dir=None,
                        batch_size=batch_size_1024px,
                        evaluate_mode=False)
        '''
        vh.union_vectors(base_dir=f"{to_predict_input_dir}_{predict_out_subdir}",
                         evaluate_mode=False,
                         force=False)
    
if __name__ == '__main__':
    main()

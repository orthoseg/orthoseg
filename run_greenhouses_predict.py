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

    # General initialisations for the segmentation project
    # -------------------------------------------------------------------------    
    segment_subject = "greenhouses"
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"
    
    # Model we will use for this segmentation
    # TODO: autodetect highest data version!!!
    train_data_version = 20
    model_architecture = "inceptionresnetv2+linknet"

    # The batch size to use. Depends on available hardware and model used.
    batch_size = 12
    
    # Info about the source images that need to be predicted
    image_pixel_width = 1024
    image_pixel_height = image_pixel_width
    border_pixels_to_ignore = 64
    pixels_overlap = border_pixels_to_ignore

    to_predict_input_dir = f"X:\\GIS\\GIS DATA\\_SegmentCache\\Ortho_2018\\{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"
    #to_predict_input_dir = f"X:\\GIS\\GIS DATA\\_Tmp\\Ortho_2018_autosegment_cache\\{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"
    #to_predict_input_dir = f"Y:\\Image_data\\Ortho2018\\{image_pixel_width}x{image_pixel_height}_{border_pixels_to_ignore}pxOverlap"
        
    # The real work...
    # -------------------------------------------------------------------------    
    project_dir = os.path.join(base_dir, segment_subject)
    
    # Main initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info(f"Start {segment_subject} prediction script")

    # Model architecture file to use   
    model_dir = os.path.join(project_dir, "models")
    model_json_filepath = f"{model_dir}{os.sep}{model_architecture}.json"

    # Create base filename of model to use
    model_base_filename = mh.model_base_filename(segment_subject,
                                                 train_data_version,
                                                 model_architecture)

    # Get the best model that already exists for this train dataset
    best_model = mh.get_best_model(model_dir=model_dir,
                                   model_base_filename=model_base_filename)
    model_weights_filepath = best_model['filepath']
    logger.info(f"Best model found: {best_model['filename']}")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir = os.path.splitext(best_model['filename'])[0]
    
    # Create the output dir's if they don't exist yet...
    for dir in [log_dir]:
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
    
    # Predict for entire dataset
    segment.predict(model=model,
                    input_image_dir=to_predict_input_dir,
                    output_base_dir=f"{to_predict_input_dir}_{predict_out_subdir}",
                    border_pixels_to_ignore=border_pixels_to_ignore,
                    input_mask_dir=None,
                    batch_size=batch_size,
                    evaluate_mode=False)
    
    vh.union_vectors(base_dir=f"{to_predict_input_dir}_{predict_out_subdir}",
                     evaluate_mode=False,
                     force=False)
    
if __name__ == '__main__':
    main()

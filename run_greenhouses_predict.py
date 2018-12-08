# -*- coding: utf-8 -*-
"""
Module to run the prediction for greenhouses.

@author: Pieter Roggemans
"""

import os

import log_helper
import segment
import preprocess as prep

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

print('Start greenhouse prediction script')

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
project_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\greenhouses"

# Vector data
input_image_dir = os.path.join(project_dir, 'input_image')
input_mask_dir = os.path.join(project_dir, 'input_mask')
input_vector_dir = os.path.join(project_dir, 'input_vector')
input_vector_labels_filepath = os.path.join(input_vector_dir, "Prc_SER_SGM.shp")

# WMS server we can use to get the image data
WMS_SERVER_URL = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'

# Train and evaluate settings
# The subdirs where the images and masks can be found by convention for training and validation
image_subdir = "image"
mask_subdir = "mask"

segmentation_model = 'linknet'
backbone_name = 'inceptionresnetv2'

# Model to use for prediction 
#model_to_use = "greenhouse_unet_vgg16_v1_077_0.02528_0.02648"
#model_to_use = "greenhouse_unet_vgg16_v2_103_0.04760_0.02614_2"
#model_to_use = "greenhouse_02_linknet_inceptionresnetv2_01_110_0.94266_0.95073"
#model_to_use = "greenhouse_02_linknet_inceptionresnetv2_02_008_0.71574_0.90055"
#model_to_use = "greenhouse_02_linknet_inceptionresnetv2_03_011_0.78744_0.86762"
#model_to_use = "greenhouse_02_linknet_inceptionresnetv2_03_011_0.78744_0.86762"
model_basename = f"greenhouse_07_{segmentation_model}_{backbone_name}_01"
#model_to_use = f"{model_basename}_037_0.93795_0.96904"
model_weights = f"{model_basename}_168_0.95860_0.95886"

#model_json_filepath = f"{project_dir}{os.sep}greenhouse_linknet_inceptionresnetv2_01.json"
model_json_filepath = f"{project_dir}{os.sep}{model_basename}.json"
model_weights_filepath = os.path.join(project_dir, f"{model_weights}.hdf5")

# Seperate validation dataset for during training...
train_dir = os.path.join(project_dir, "train")
validation_dir = os.path.join(project_dir, "validation")

# Prediction settings
prediction_eval_subdir = f"prediction_{model_weights}_eval"

border_pixels_to_ignore = 64

# Real prediction dir
batch_size_512px = 24
batch_size_1024px = 12

image_pixel_width = 1024
image_pixel_height = image_pixel_width
pixels_overlap = border_pixels_to_ignore
to_predict_input_dir = f"X:\\GIS\\GIS DATA\\_Tmp\\Ortho_2018_autosegment_cache\\{image_pixel_width}x{image_pixel_height}_{pixels_overlap}pxOverlap"
#to_predict_input_dir = None

# Log dir
log_dir = os.path.join(project_dir, "log")

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():

    # Main initialisation of the logging
    logger = log_helper.main_log_init(log_dir, __name__)

    # Create the output dir's if they don't exist yet...
    for dir in [project_dir, train_dir, log_dir]:
        if dir and not os.path.exists(dir):
            os.mkdir(dir)

    # Predict for training dataset
    '''
    segment.predict(model_json_filepath=model_json_filepath,
                    model_weights_filepath=model_weights_filepath,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_predict_dir=os.path.join(train_dir, prediction_eval_subdir),
                    border_pixels_to_ignore=border_pixels_to_ignore,
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    batch_size=batch_size_512px,
                    evaluate_mode=True)
    '''
    '''
    # Predict for validation dataset
    segment.predict(model_json_filepath=model_json_filepath,
                    model_weights_filepath=model_weights_filepath,
                    input_image_dir=os.path.join(validation_dir, image_subdir),
                    output_predict_dir=os.path.join(validation_dir, prediction_eval_subdir),
                    border_pixels_to_ignore=border_pixels_to_ignore,
                    input_mask_dir=os.path.join(validation_dir, mask_subdir),
                    batch_size=batch_size,
                    evaluate_mode=True)
    '''
    
    # Predict for test dataset with all greenhouses
    test_dir = os.path.join(project_dir, "test_all")
    '''
    if not os.path.exists(test_dir):
        logger.info('Prepare test_dir data')
        prep.prepare_training_data(
                input_vector_label_filepath=input_vector_labels_filepath,
                wms_server_url=WMS_SERVER_URL,
                wms_server_layer='ofw',
                output_image_dir=os.path.join(test_dir, image_subdir),
                output_mask_dir=os.path.join(test_dir, mask_subdir),
                max_samples=None,
                force=True)
    
    segment.predict(model_json_filepath=model_json_filepath,
                    model_weights_filepath=model_weights_filepath,
                    input_image_dir=os.path.join(test_dir, image_subdir),
                    output_predict_dir=os.path.join(test_dir, prediction_eval_subdir),
                    border_pixels_to_ignore=border_pixels_to_ignore,
                    input_mask_dir=os.path.join(test_dir, mask_subdir),
                    batch_size=batch_size_1024px,
                    evaluate_mode=True)
    '''
    
    
    # Predict for test dataset with andom images in the roi
    test_random_dir = os.path.join(project_dir, "test_random")
    test_random_input_dir = os.path.join(test_random_dir, "image512x512_0pxOverlap")
    segment.predict(model_json_filepath=model_json_filepath,
                    model_weights_filepath=model_weights_filepath,
                    input_image_dir=test_random_input_dir,
                    output_predict_dir=f"{test_random_input_dir}_{prediction_eval_subdir}",
                    border_pixels_to_ignore=0,
                    input_mask_dir=None,
                    batch_size=batch_size_512px,
                    evaluate_mode=True)
    
    '''
    # Predict for entire dataset
    if to_predict_input_dir:
        segment.predict(model_json_filepath=model_json_filepath,
                        model_weights_filepath=model_weights_filepath,
                        input_image_dir=to_predict_input_dir,
                        output_predict_dir=f"{to_predict_input_dir}_{prediction_eval_subdir}",
                        border_pixels_to_ignore=border_pixels_to_ignore,
                        input_mask_dir=None,
                        batch_size=batch_size_1024px,
                        evaluate_mode=True)
    '''
    
if __name__ == '__main__':
    main()

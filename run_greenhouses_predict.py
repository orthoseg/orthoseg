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
#model_to_use = "greenhouse_unet_vgg16_v5_832_0.15442_0.07930_2"
model_to_use = "greenhouse_unet_vgg16_v2_103_0.04760_0.02614_2"
model_to_use = "greenhouse_unet_inceptionresnetv2_v1_118_0.02477_0.01867_2"
model_to_use = "greenhouse_linknet_inceptionresnetv2_v2_627_0.02546_0.01443_2"
#model_to_use = 'greenhouse_unet_zhix_v6'
model_to_use_filepath = os.path.join(project_dir, f"{model_to_use}.hdf5")

batch_size = 4

# Seperate validation dataset for during training...
train_dir = os.path.join(project_dir, "train")
validation_dir = os.path.join(project_dir, "validation")

# Prediction settings
prediction_eval_subdir = f"prediction_{model_to_use}_eval"

# Real prediction dir
to_predict_input_dir = "X:\GIS\GIS DATA\_Tmp\Ortho_2018_autosegment_cache\1024x1024_50pxOverlap"
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

    '''
    # Predict for training dataset
    segment.predict(segmentation_model=segmentation_model,
                    backbone_name=backbone_name,
                    model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_predict_dir=os.path.join(train_dir, prediction_eval_subdir),
                    input_ext=['.tif'],
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    prefix_with_similarity=True)
    # There are 2 different sizes of tifs, 
    # TODO: add support in predict for different image sizes...
    segment.predict(segmentation_model=segmentation_model,
                    backbone_name=backbone_name,
                    model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_predict_dir=os.path.join(train_dir, prediction_eval_subdir),
                    input_ext=['.tif'],
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    prefix_with_similarity=True)
    segment.predict(segmentation_model=segmentation_model,
                    backbone_name=backbone_name,
                    model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_predict_dir=os.path.join(train_dir, prediction_eval_subdir),
                    input_ext=['.jpg'],
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    prefix_with_similarity=True)

    # Predict for validation dataset
    segment.predict(segmentation_model=segmentation_model,
                    backbone_name=backbone_name,
                    model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(validation_dir, image_subdir),
                    output_predict_dir=os.path.join(validation_dir, prediction_eval_subdir),
                    input_ext=['.jpg'],
                    input_mask_dir=os.path.join(validation_dir, mask_subdir),
                    prefix_with_similarity=True)
    segment.predict(segmentation_model=segmentation_model,
                    backbone_name=backbone_name,
                    model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(validation_dir, image_subdir),
                    output_predict_dir=os.path.join(validation_dir, prediction_eval_subdir),
                    input_ext=['.tif'],
                    input_mask_dir=os.path.join(validation_dir, mask_subdir),
                    prefix_with_similarity=True)

    # Predict for test dataset
    test_dir = os.path.join(project_dir, "test_all")
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

    segment.predict(segmentation_model=segmentation_model,
                    backbone_name=backbone_name,
                    model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(test_dir, image_subdir),
                    output_predict_dir=os.path.join(test_dir, prediction_eval_subdir),
                    input_ext=['.tif'],
                    input_mask_dir=os.path.join(test_dir, mask_subdir),
                    prefix_with_similarity=True)

    '''
    
    # Predict for entire dataset
    if to_predict_input_dir:
        segment.predict(segmentation_model=segmentation_model,
                        backbone_name=backbone_name,
                        model_to_use_filepath=model_to_use_filepath,
                        input_image_dir=to_predict_input_dir,
                        output_predict_dir=f"{to_predict_input_dir}_{prediction_eval_subdir}",
                        input_ext=['.jpg', '.tif'],
                        input_mask_dir=None,
                        prefix_with_similarity=True)

if __name__ == '__main__':
    main()

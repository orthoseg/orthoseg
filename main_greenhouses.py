# -*- coding: utf-8 -*-
"""
Module with helper functions to preprocess the data to use for the classification.

@author: Pieter Roggemans
"""

import os

import log_helper
import segment
import preprocess as prep

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

print('Start greenhouse script')

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

model_train_dir = project_dir
#model_train_dir = None
model_train_basename = 'unet_vgg16_greenhouse_v1'
model_train_best_name = model_train_basename + '_best'
model_train_preload_filepath = os.path.join(project_dir, "unet_vgg16_greenhouse_v1_069_0.02740_0.028812.hdf5")
#model_train_preload_filepath = None
batch_size = 4
nb_epoch = 100
train_dir = os.path.join(project_dir, "train")
train_augmented_dir = None #os.path.join(project_dir, "train_augmented")

# Prediction settings
# Model settings
# If we are training a model, use the best model for prediction...
if model_train_dir:
    model_to_use = model_train_best_name
    model_to_use_filepath = os.path.join(project_dir, model_train_best_name + ".hdf5")
else:
    #model_to_use = "unet_greenhouse_loss_0.15832"
    #model_to_use = "unet_vgg16_greenhouse_0.13600"
    #model_to_use = "unet_vgg16_greenhouse_0.11282"
    #model_to_use = "unet_vgg16_greenhouse_v4_019_0.58730"
    #model_to_use = "unet_vgg16_greenhouse_v1_004_0.69023"
#    model_to_use = "unet_vgg16_greenhouse03_01_0.05179"
    #model_to_use = "unet_vgg16_greenhouse_v1_009_0.05235"
    #model_to_use = "unet_vgg16_greenhouse_v1_026_0.03927_0.03259"
    #model_to_use = "unet_vgg16_greenhouse_v1_040_0.03077_0.03341"
    model_to_use = "unet_vgg16_greenhouse_v1_042_0.02450_0.02486"
    model_to_use = "unet_vgg16_greenhouse_v1_069_0.02740_0.028812"
#    model_to_use = model_train_best_name
    model_to_use_filepath = os.path.join(project_dir, f"{model_to_use}.hdf5")
prediction_eval_subdir = f"prediction_{model_to_use}_eval"

# Validation dir
validation_dir = os.path.join(project_dir, "validation")

# Real prediction dir
to_predict_input_dir = "X:\\GIS\\GIS DATA\_Tmp\\Ortho_2018_autosegment_cache\\1024x1024"
to_predict_input_dir = None

# Log dir
log_dir = os.path.join(project_dir, "log")

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():

    # Main initialisation of the logging
    logger = log_helper.main_log_init(log_dir, __name__)

    # Create the output dir's if they don't exist yet...
    for dir in [project_dir, train_dir, train_augmented_dir, log_dir]:
        if dir and not os.path.exists(dir):
            os.mkdir(dir)

    if model_train_dir:
        # If the training data doesn't exist yet, create it
        train_image_dir = os.path.join(train_dir, image_subdir)
        if not os.path.exists(train_image_dir):
            logger.info('Prepare training data')
            prep.prepare_training_data(
                    input_vector_label_filepath=input_vector_labels_filepath,
                    wms_server_url=WMS_SERVER_URL,
                    wms_server_layer='ofw',
                    output_image_dir=train_image_dir,
                    output_mask_dir=os.path.join(train_dir, mask_subdir),
                    max_samples=1,
                    force=True)

        logger.info('Start training')
        # TODO: enable validation dir again
        segment.train(traindata_dir=train_dir,
                      validationdata_dir=validation_dir,
                      image_subdir=image_subdir,
                      mask_subdir=mask_subdir,
                      model_dir=model_train_dir,
                      model_basename=model_train_basename,
                      model_preload_filepath=model_train_preload_filepath,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      train_augmented_dir=train_augmented_dir)

    # Predict for training dataset
    segment.predict(model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_predict_dir=os.path.join(train_dir, prediction_eval_subdir),
                    input_ext=['.tif'],
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    prefix_with_similarity=True)
    segment.predict(model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_predict_dir=os.path.join(train_dir, prediction_eval_subdir),
                    input_ext=['.jpg'],
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    prefix_with_similarity=True)

    # Predict for validation dataset
    segment.predict(model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(validation_dir, image_subdir),
                    output_predict_dir=os.path.join(validation_dir, prediction_eval_subdir),
                    input_ext=['.jpg'],
                    input_mask_dir=os.path.join(validation_dir, mask_subdir),
                    prefix_with_similarity=True)
    segment.predict(model_to_use_filepath=model_to_use_filepath,
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

    segment.predict(model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(test_dir, image_subdir),
                    output_predict_dir=os.path.join(test_dir, prediction_eval_subdir),
                    input_ext=['.tif'],
                    input_mask_dir=os.path.join(test_dir, mask_subdir),
                    prefix_with_similarity=True)

    # Predict for entire dataset
    if to_predict_input_dir:
        segment.predict(model_to_use_filepath=model_to_use_filepath,
                        input_image_dir=to_predict_input_dir,
                        output_predict_dir=os.path.join(to_predict_input_dir, prediction_eval_subdir),
                        input_ext=['.jpg', '.tif'],
                        input_mask_dir=None,
                        prefix_with_similarity=True)

if __name__ == '__main__':
    main()

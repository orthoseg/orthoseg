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

# Model settings
#Âmodel_to_use = "unet_greenhouse_loss_0.15832"
#model_to_use = "unet_vgg16_greenhouse_0.13600"
#model_to_use = "unet_vgg16_greenhouse_0.11282"
model_to_use = "unet_vgg16_greenhouse03_09_0.05235"
model_to_use_filepath = os.path.join(project_dir, f"{model_to_use}.hdf5")

# The subdirs where the images and masks can be found by convention for training and validation
prediction_eval_subdir = f"prediction_{model_to_use}_eval"

# Vector data
input_image_dir = os.path.join(project_dir, 'input_image')
input_mask_dir = os.path.join(project_dir, 'input_mask')
input_vector_dir = os.path.join(project_dir, 'input_vector')
input_vector_labels_filepath = os.path.join(input_vector_dir, "Prc_SER_SGM.shp")

# WMS server we can use to get the image data
WMS_SERVER_URL = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'

# Train and evaluate settings
model_train_dir = project_dir
model_train_basename = 'unet_vgg16_greenhouse03'
#model_train_filepath = None
model_train_preload_filepath = os.path.join(project_dir, model_train_basename + '_best.hdf5')
model_train_filepath = os.path.join(project_dir, 'unet_vgg16_greenhouse_best.hdf5')
model_train_filepath = None
model_train_preload_filepath = model_train_filepath
batch_size = 4
train_dir = os.path.join(project_dir, "train")
train_augmented_dir = None #os.path.join(project_dir, "train_augmented")
image_subdir = "images_labeled"
mask_subdir = "masks"

# Validation settings
validation_dir = os.path.join(project_dir, "validation")

# All images settings
to_predict_input_dir = None #os.path.join(project_dir, "input_images")
#to_predict_input_dir = '\\\\dg3.be\\alp\\Datagis\\Ortho_AGIV_2018_ofw'
to_predict_input_dir = "X:\\GIS\\GIS DATA\_Tmp\\Ortho_2018_autosegment_cache\\1024x1024"

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

        segment.train(traindata_dir=train_dir,
                      validationdata_dir=validation_dir,
                      image_subdir=image_subdir,
                      mask_subdir=mask_subdir,
                      model_dir=model_train_dir,
                      model_basename=model_train_basename,
                      model_preload_filepath=model_train_preload_filepath,
                      batch_size=batch_size,
                      train_augmented_dir=train_augmented_dir)

    # Predict for training dataset
    segment.predict(model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_predict_dir=os.path.join(train_dir, prediction_eval_subdir),
<<<<<<< HEAD
                    input_ext=['.jpg', '.tif'],
                    input_mask_dir=os.path.join(train_dir, mask_subdir))
                    input_ext=['.tif'],
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    prefix_with_jaccard=True)
    segment.predict(model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(train_dir, image_subdir),
                    output_predict_dir=os.path.join(train_dir, prediction_eval_subdir),
                    input_ext=['.jpg'],
                    input_mask_dir=os.path.join(train_dir, mask_subdir),
                    prefix_with_jaccard=True)

    # Predict for validation dataset
    segment.predict(model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(validation_dir, image_subdir),
                    output_predict_dir=os.path.join(validation_dir, prediction_eval_subdir),
                    input_ext=['.jpg', '.tif'],
                    input_mask_dir=os.path.join(validation_dir, mask_subdir))
    '''
                    input_ext=['.jpg'],
                    input_mask_dir=os.path.join(validation_dir, mask_subdir),
                    prefix_with_jaccard=True)
    segment.predict(model_to_use_filepath=model_to_use_filepath,
                    input_image_dir=os.path.join(validation_dir, image_subdir),
                    output_predict_dir=os.path.join(validation_dir, prediction_eval_subdir),
                    input_ext=['.tif'],
                    input_mask_dir=os.path.join(validation_dir, mask_subdir),
                    prefix_with_jaccard=True)
>>>>>>> 96344d75b661a27ff71e2e36c49c2f27d38c311a

    '''
    # Predict for entire dataset
    if to_predict_input_dir:
        segment.predict(model_to_use_filepath=model_to_use_filepath,
                        input_image_dir=to_predict_input_dir,
                        output_predict_dir=os.path.join(to_predict_input_dir, prediction_eval_subdir),
                        input_ext=['.jpg', '.tif'],
                        input_mask_dir=None)
    '''
if __name__ == '__main__':
    main()

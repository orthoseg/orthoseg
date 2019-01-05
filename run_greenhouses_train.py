# -*- coding: utf-8 -*-
"""
Module with helper functions to preprocess the data to use for the classification.

@author: Pieter Roggemans
"""

import os

import log_helper
import segment
import prepare_traindatasets as prep

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
    
    # WMS server we can use to get the image data
    WMS_SERVER_URL = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'
    
    # More detailed info about the model and train data (version)
    train_data_version = 13
    encoder = 'inceptionresnetv2'
    decoder = 'linknet'
    model_architecture = f"{encoder}+{decoder}"
    model_weights_to_use = None
    
    # The batch size to use depends on the model architecture, the size of the 
    # training images and the available (GPU) memory
    batch_size = 8
    
    nb_epoch = 1000
        
    # General initialisations...
    #
    # Remark: they don't need to be changed if implied conventions are OK
    # -------------------------------------------------------------------------    
    # Main initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)

    # Create file names of model, weight,... to use
    model_basename = f"{segment_subject}_{train_data_version:02}_{model_architecture}"
    logger.info(f"model_basename: {model_basename}")
        
    model_train_preload_filepath = None
    if model_weights_to_use:
        model_train_preload_filepath = os.path.join(project_dir, f"{model_weights_to_use}.hdf5")

    # Input label data
    input_labels_dir = os.path.join(project_dir, 'input_labels')
    input_labels_filename = f"{segment_subject}_groundtruth.geojson"
    input_labels_filepath = os.path.join(input_labels_dir, 
                                         input_labels_filename)
    
    # The subdirs where the images and masks can be found by convention for training and validation
    image_subdir = "image"
    mask_subdir = "mask"
    
    model_save_dir = project_dir
   
    train_dir = os.path.join(project_dir, "train")
    train_augmented_dir = None #os.path.join(project_dir, "train_augmented")
    
    # Seperate validation dataset for during training...
    validation_dir = os.path.join(project_dir, "validation")
    
    # Create the output dir's if they don't exist yet...
    for dir in [project_dir, train_dir, train_augmented_dir, log_dir]:
        if dir and not os.path.exists(dir):
            os.mkdir(dir)

    # If the training data doesn't exist yet, create it
    train_image_dir = os.path.join(train_dir, image_subdir)
    force_create_train_data = False 
    if(force_create_train_data 
       or not os.path.exists(train_image_dir)):
        logger.info('Prepare train and validation data')
        prep.prepare_traindatsets(
                input_vector_label_filepath=input_labels_filepath,
                wms_server_url=WMS_SERVER_URL,
                wms_server_layer='ofw',
                output_image_dir=train_image_dir,
                output_mask_dir=os.path.join(train_dir, mask_subdir),
                force=force_create_train_data)

    logger.info('Start training')
    segment.train(traindata_dir=train_dir,
                  validationdata_dir=validation_dir,
                  image_subdir=image_subdir,
                  mask_subdir=mask_subdir,
                  encoder=encoder,
                  decoder=decoder,
                  model_save_dir=model_save_dir,
                  model_save_basename=model_basename,
                  model_preload_filepath=model_train_preload_filepath,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  train_augmented_dir=train_augmented_dir)
    
if __name__ == '__main__':
    main()

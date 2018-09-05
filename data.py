# -*- coding: utf-8 -*-
"""
Module with helper functions to preprocess the data to use for the classification.

@author: Pieter Roggemans
"""

from __future__ import print_function
import logging
import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def create_train_generator(input_data_dir, image_subdir, mask_subdir,
                           aug_dict, batch_size=32,
                           image_color_mode="rgb", mask_color_mode="grayscale",
                           save_to_dir=None, image_save_prefix="image", mask_save_prefix="mask",
                           flag_multi_class=False, num_class=2,
                           target_size=(256,256), seed=1, class_mode=None):
    '''
    Can generate image and mask at the same time

    Remarks: * use the same seed for image_datagen and mask_datagen to ensure the
               transformation for image and mask is the same
             * if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        directory=input_data_dir,
        classes=[image_subdir],
        class_mode=class_mode,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        directory=input_data_dir,
        classes=[mask_subdir],
        class_mode=class_mode,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    return train_generator

def predictGenerator2(input_data_dir, image_subdir,
                   aug_dict, batch_size,
                   image_color_mode="rgb", mask_color_mode="grayscale",
                   save_to_dir=None, image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2,
                   target_size=(256,256), seed=1, class_mode=None):
    '''
    Generates images for testing
    if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        directory=input_data_dir,
        classes=[image_subdir],
        class_mode=class_mode,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        shuffle=False)
    return image_generator
    '''
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)
    '''

def predictGenerator(test_path, num_image=30, target_size=(256,256),
                  flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png"%i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img
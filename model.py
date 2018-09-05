# -*- coding: utf-8 -*-
"""
Unet definition in Keras.

Based on code from:
    * unet: https://github.com/zhixuhao/unet
    * preload first layer with VGG16 weights:
      https://www.microsoft.com/developerblog/2018/07/05/satellite-images-segmentation-sustainable-farming/

@author: Pieter Roggemans
"""

import logging
import keras as kr
import numpy as np

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def get_unet(input_width=256, input_height=256, n_channels=3,
             n_classes=1, loss_mode='binary_crossentropy',
             init_with_vgg16: bool = None, pretrained_weights_filepath: str = None):

    inputs = kr.layers.Input((input_width, input_height, n_channels))
    conv1 = kr.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="block1_conv1")(inputs)
    conv1 = kr.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="block1_conv2")(conv1)
    pool1 = kr.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = kr.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = kr.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = kr.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = kr.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = kr.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = kr.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = kr.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = kr.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = kr.layers.Dropout(0.5)(conv4)
    pool4 = kr.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = kr.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = kr.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = kr.layers.Dropout(0.5)(conv5)

    up6 = kr.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(kr.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = kr.layers.Concatenate(axis=3)([drop4, up6])
    conv6 = kr.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = kr.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = kr.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(kr.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = kr.layers.Concatenate(axis=3)([conv3, up7])
    conv7 = kr.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = kr.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = kr.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(kr.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = kr.layers.Concatenate(axis=3)([conv2, up8])
    conv8 = kr.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = kr.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = kr.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(kr.layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = kr.layers.Concatenate(axis=3)([conv1, up9])
    conv9 = kr.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = kr.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = kr.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = kr.layers.Conv2D(n_classes, 1, activation='sigmoid')(conv9)

    model = kr.models.Model(inputs=inputs, outputs=conv10)

    if loss_mode == "bcedice":
        loss_func = dice_coef_loss_bce
    elif loss_mode == "binary_crossentropy":
        loss_func = 'binary_crossentropy'
    else:
        raise Exception(f"Unknown loss function: {loss_mode}")

    model.compile(optimizer=kr.optimizers.Adam(lr=1e-4), loss=loss_func,
                  metrics=[jaccard_coef, jacard_coef_flat,
                           jaccard_coef_int, dice_coef, 'accuracy'])

    if(init_with_vgg16):
        # Load the VGG16 model with pretrained weights
        # Remark: this only has 3 channels. If the input images have more channels, those channels
        #         won't be pre-loaded
        vgg = kr.applications.vgg16.VGG16(include_top=False, input_shape=(input_width, input_height, 3), weights='imagenet')

        # Create empty array with the dimensions of the first convolution layer
        conv1_weights = np.zeros((3, 3, n_channels, 64), dtype="float32")

        conv1_weights[:, :, :3, :] = vgg.get_layer("block1_conv1").get_weights()[0][:, :, :, :]
        bias = vgg.get_layer("block1_conv1").get_weights()[1]
        model.get_layer('block1_conv1').set_weights((conv1_weights, bias))

    #model.summary()

    if(pretrained_weights_filepath):
    	model.load_weights(pretrained_weights_filepath)

    return model

#------------------------------------------
# Loss functions
#------------------------------------------

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = kr.backend.tensorflow_backend._to_tensor(kr.backend.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = kr.backend.tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = kr.backend.tf.log(prediction_tensor / (1 - prediction_tensor))

    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * kr.backend.tf.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * kr.backend.tf.cast(
            kr.backend.tf.sigmoid(prediction_tensor) > 0.5, kr.backend.tf.float32)
    return kr.backend.mean(kr.backend.tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))

def dice_coef_loss_bce(y_true, y_pred):
    dice = 0.5
    bce = 0.5
    bootstrapping = 'hard'
    alpha = 1.
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice

#------------------------------------------
# Metrics functions
#------------------------------------------

SMOOTH_LOSS = 1e-12

def jaccard_coef(y_true, y_pred):
    intersection = kr.backend.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = kr.backend.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)

    return kr.backend.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = kr.backend.round(kr.backend.clip(y_pred, 0, 1))

    intersection = kr.backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = kr.backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return kr.backend.mean(jac)

def jacard_coef_flat(y_true, y_pred):
    y_true_f = kr.backend.flatten(y_true)
    y_pred_f = kr.backend.flatten(y_pred)
    intersection = kr.backend.sum(y_true_f * y_pred_f)
    return (intersection + SMOOTH_LOSS) / (kr.backend.sum(y_true_f) + kr.backend.sum(y_pred_f) - intersection + SMOOTH_LOSS)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = kr.backend.flatten(y_true)
    y_pred_f = kr.backend.flatten(y_pred)
    intersection = kr.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (kr.backend.sum(y_true_f) + kr.backend.sum(y_pred_f) + smooth)

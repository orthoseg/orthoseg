# -*- coding: utf-8 -*-
"""
Module with helper functions to create models.

Offers a common interface, regardless of the underlying model implementation 
and contains extra metrics, callbacks,...

Many models are supported by using this segmentation model zoo:
https://github.com/qubvel/segmentation_models
"""

import logging
from pathlib import Path

from tensorflow import keras as kr
#import keras as kr
import numpy as np
import tensorflow as tf
import segmentation_models as sm

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

'''
preprocessing_fn = get_preprocessing('resnet34')
x = preprocessing_fn(x)
'''

def get_model(
        architecture: str,
        input_width: int = None,
        input_height: int = None,
        nb_channels: int = 3,
        nb_classes: int = 1,
        activation: str = 'sigmoid',
        init_weights_with: str = 'imagenet'):
    """[summary]

    Arguments:
        architecture

    Keyword Arguments:
        input_width {int} -- [description] (default: {None})
        input_height {int} -- [description] (default: {None})
        nb_channels {int} -- [description] (default: {3})
        nb_classes {int} -- [description] (default: {1})
        activation {str} -- [description] (default: {'sigmoid'init_weights_with:str='imagenet'})

    Returns:
        [type] -- [description]
    """
    # Check architecture
    segment_architecture_parts = architecture.split('+')
    if len(segment_architecture_parts) < 2:
        raise Exception(f"Unsupported architecture: {architecture}")
    encoder = segment_architecture_parts[0]
    decoder = segment_architecture_parts[1]

    if decoder.lower() == 'deeplabv3plus':
        import models.model_deeplabv3plus as m
        return m.get_model(input_width=input_width, input_height=input_height,
                           nb_channels=nb_channels, nb_classes=nb_classes,
                           init_model_weights=init_weights_with)
    elif decoder.lower() == 'unet':
        # These two unet variants are implemented in a seperate module
        if encoder.lower() == 'standard':
            import models.model_unet_standard as m
            if init_weights_with is not None:
                init_weights = True
            else:
                init_weights = False
            return m.get_model(input_width=input_width, input_height=input_height,
                               nb_channels=nb_channels, nb_classes=nb_classes,
                               init_model_weights=init_weights)
        elif encoder.lower() == 'ternaus':
            import models.model_unet_ternaus as m
            if init_weights_with is not None:
                init_weights = True
            else:
                init_weights = False                
            return m.get_model(input_width=input_width, input_height=input_height,
                               nb_channels=nb_channels, nb_classes=nb_classes,
                               init_model_weights=init_weights)

        # Some other unet variants is implemented using the segmentation_models library
        from segmentation_models import Unet
        #from segmentation_models.backbones import get_preprocessing

        model = Unet(backbone_name=encoder.lower(),
                     input_shape=(input_width, input_height, nb_channels),
                     classes=nb_classes, activation=activation,
                     encoder_weights=init_weights_with)
        return model
    elif decoder.lower() == 'pspnet':
        from segmentation_models import PSPNet
        #from segmentation_models.backbones import get_preprocessing

        model = PSPNet(backbone_name=encoder.lower(),
                       input_shape=(input_width, input_height, nb_channels),
                       classes=nb_classes, activation=activation,
                       encoder_weights=init_weights_with)
        return model
    elif decoder.lower() == 'linknet':
        from segmentation_models import Linknet
        #from segmentation_models.backbones import get_preprocessing

        # First check if input size is compatible with linknet 
        if input_width is not None and input_height is not None:
            check_image_size(decoder, input_width, input_height)
            
        model = Linknet(backbone_name=encoder.lower(),
                        input_shape=(input_width, input_height, nb_channels),
                        classes=nb_classes, activation=activation,
                        encoder_weights=init_weights_with)
        return model
    else:
        raise Exception(f"Unknown decoder architecture: {decoder}")

def compile_model(
        model,
        optimizer,
        loss: str,
        metrics: list = None,
        sample_weight_mode: str = None,
        class_weights: list = None):

    # If no merics specified, use default ones
    if metrics is None:
        metrics = []
        if loss in ['categorical_crossentropy', 'weighted_categorical_crossentropy']:
            metrics.append('categorical_accuracy')
        elif loss == 'sparse_categorical_crossentropy':
            #metrics.append('sparse_categorical_accuracy')
            None
        elif loss == 'binary_crossentropy':
            metrics.append('binary_accuracy')

        iou_score = sm.metrics.IOUScore()
        metrics.append(iou_score)
        f1_score = sm.metrics.FScore()
        metrics.append(f1_score)
        #metrics.append(jaccard_coef_round)
        # metrics=[jaccard_coef, jaccard_coef_flat,
        #          jaccard_coef_int, dice_coef, 'accuracy', 'binary_accuracy']

    # Check loss function
    if loss == 'bcedice':
        loss_func = dice_coef_loss_bce
    elif loss == 'dice_loss':
        loss_func = sm.losses.DiceLoss()
    elif loss == 'jaccard_loss':
        loss_func = sm.losses.JaccardLoss()
    elif loss == 'weighted_categorical_crossentropy':
        if class_weights is None: 
            raise Exception(f"With loss == {loss}, class_weights cannot be None!")
        loss_func = weighted_categorical_crossentropy(class_weights)
    else:
        loss_func = loss

    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics,
            sample_weight_mode=sample_weight_mode)

    return model

def load_model(
        model_to_use_filepath: Path, 
        compile: bool = True):
    # If it is a file with only weights
    if model_to_use_filepath.stem.endswith('_weights'):
        model_json_filename = model_to_use_filepath.stem.replace('_weights', '') + '.json'
        model_json_filepath = model_to_use_filepath.parent / model_json_filename
        with model_json_filepath.open('r') as src:
            model_json = src.read()
            model = kr.models.model_from_json(model_json)
        model.load_weights(str(model_to_use_filepath))
    else:
        iou_score = sm.metrics.IOUScore()
        f1_score = sm.metrics.FScore()
        model = kr.models.load_model(
                str(model_to_use_filepath),
                custom_objects={'jaccard_coef': jaccard_coef,
                                'jaccard_coef_flat': jaccard_coef_flat,
                                'jaccard_coef_round': jaccard_coef_round,
                                'dice_coef': dice_coef,
                                'iou_score': iou_score,
                                'f1_score': f1_score,
                                'weighted_categorical_crossentropy': weighted_categorical_crossentropy},
                compile=compile)

    return model

def check_image_size(
        decoder: str,
        input_width: int, 
        input_height: int):
    if decoder.lower() == 'linknet':
        if((input_width and (input_width % 16) != 0) 
           or (input_height and (input_height % 16) != 0)):
            message = f"STOP: input_width ({input_width} and input_height ({input_height}) should be divisable by 16!"
            logger.error(message)
            raise Exception(message)
    else:
        logger.info(f"check_image_size is not implemented for this model: {decoder}")
        
#------------------------------------------
# Loss functions
#------------------------------------------

def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    if isinstance(weights,list) or isinstance(weights, np.ndarray):
        weights=kr.backend.variable(weights)

    def loss(target,output,from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(kr.backend.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.math.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1)
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')
    return loss

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

def jaccard_coef_round(y_true, y_pred):
    y_pred_pos = kr.backend.round(kr.backend.clip(y_pred, 0, 1))

    intersection = kr.backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = kr.backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return kr.backend.mean(jac)

def jaccard_coef_flat(y_true, y_pred):
    y_true_f = kr.backend.flatten(y_true)
    y_pred_f = kr.backend.flatten(y_pred)
    intersection = kr.backend.sum(y_true_f * y_pred_f)
    return (intersection + SMOOTH_LOSS) / (kr.backend.sum(y_true_f) + kr.backend.sum(y_pred_f) - intersection + SMOOTH_LOSS)

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = kr.backend.flatten(y_true)
    y_pred_f = kr.backend.flatten(y_pred)
    intersection = kr.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (kr.backend.sum(y_true_f) + kr.backend.sum(y_pred_f) + smooth)

def pct_wrong(y_true, y_pred):
    y_pred_pos = kr.backend.round(kr.backend.clip(y_pred, 0, 1))

    intersection = kr.backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = kr.backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return kr.backend.mean(jac)

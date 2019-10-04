# -*- coding: utf-8 -*-
"""
Module with helper functions to create models.

Offers a common interface, regardless of the underlying model implementation 
and contains extra metrics, callbacks,...

Many models are supported by using this segmentation model zoo:
https://github.com/qubvel/segmentation_models
"""

import logging

import keras as kr
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
        encoder: str,
        decoder: str,
        input_width: int = None,
        input_height: int = None,
        nb_channels: int = 3,
        nb_classes: int = 1,
        activation: str = 'sigmoid',
        init_weights_with: str = 'imagenet'):
    """[summary]

    Arguments:
        encoder {str} -- [description]
        decoder {str} -- [description]

    Keyword Arguments:
        input_width {int} -- [description] (default: {None})
        input_height {int} -- [description] (default: {None})
        nb_channels {int} -- [description] (default: {3})
        nb_classes {int} -- [description] (default: {1})
        activation {str} -- [description] (default: {'sigmoid'init_weights_with:str='imagenet'})

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        [type] -- [description]
    """
    if decoder.lower() == 'deeplabv3plus':
        import models.model_deeplabv3plus as m
        return m.get_model(input_width=input_width, input_height=input_height,
                           nb_channels=nb_channels, nb_classes=nb_classes,
                           init_model_weights=init_weights_with)
    elif decoder.lower() == 'unet':
        # These two unet variants are implemented in a seperate module
        if encoder.lower() == 'standard':
            import models.model_unet_standard as m
            if init_weights_with:
                init_weights = True
            return m.get_model(input_width=input_width, input_height=input_height,
                               nb_channels=nb_channels, nb_classes=nb_classes,
                               init_model_weights=init_weights)
        elif encoder.lower() == 'ternaus':
            import models.model_unet_ternaus as m
            if init_weights_with:
                init_weights = True
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
        metrics: [] = None):

    # If no merics specified, use default ones
    if metrics is None:
        metrics = []
        if loss == 'categorical_crossentropy':
            metrics.append('categorical_accuracy')
        elif loss == 'sparse_categorical_crossentropy':
            #metrics.append('sparse_categorical_accuracy')
            None
        elif loss == 'binary_crossentropy':
            metrics.append('binary_accuracy')

        iou_score = sm.metrics.iou_score
        metrics.append(iou_score)
        #metrics.append(jaccard_coef_round)
        # metrics=[jaccard_coef, jaccard_coef_flat,
        #          jaccard_coef_int, dice_coef, 'accuracy', 'binary_accuracy']

    # Check loss function
    if loss == 'bcedice':
        loss_func = dice_coef_loss_bce
    elif loss == 'jaccard':
        loss_func = sm.losses.jaccard_loss
    else:
        loss_func = loss

    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)

    return model

def load_model(model_to_use_filepath: str):
    iou_score = sm.metrics.iou_score
    
    model = kr.models.load_model(
            model_to_use_filepath,
            custom_objects={'jaccard_coef': jaccard_coef,
                            'jaccard_coef_flat': jaccard_coef_flat,
                            'jaccard_coef_round': jaccard_coef_round,
                            'dice_coef': dice_coef,
                            'iou_score': iou_score})

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

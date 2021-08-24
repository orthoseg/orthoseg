# -*- coding: utf-8 -*-
"""
Module with helper functions to create models.

Offers a common interface, regardless of the underlying model implementation 
and contains extra metrics, callbacks,...

Many models are supported by using this segmentation model zoo:
https://github.com/qubvel/segmentation_models
"""

import json
import logging
from pathlib import Path
from typing import Any, List

from tensorflow import keras as kr
import numpy as np
import tensorflow as tf
import segmentation_models as sm

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

# Force using the keras implementation in tensorflow, because the default 
# use of this implementation seems broken with tensorflow 2.5
sm.set_framework('tf.keras')

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
        activation: str = 'softmax',
        init_weights_with: str = 'imagenet') -> kr.models.Model:
    """
    Get a model.
    
    Args:
        architecture (str): Architecture of the network to create
        input_width (int, optional): Width of the input images. Defaults to None.
        input_height (int, optional): Height of the input images. Defaults to None.
        nb_channels (int, optional): Nb of channels/bands of the input images. Defaults to 3.
        nb_classes (int, optional): Nb of classes to be segmented to. Defaults to 1.
        activation (Activation, optional): Activation function of last layer. Defaults to 'softmax'.
        init_weights_with (str, optional): Weights to init the network with. Defaults to 'imagenet'.
    
    Raises:
        Exception: [description]
        Exception: [description]
    
    Returns:
        [type]: [description]
    """
    # Check architecture
    segment_architecture_parts = architecture.split('+')
    if len(segment_architecture_parts) < 2:
        raise Exception(f"Unsupported architecture: {architecture}")
    encoder = segment_architecture_parts[0]
    decoder = segment_architecture_parts[1]

    if decoder.lower() == 'unet':
        # These two unet variants are implemented in a seperate module
        if encoder.lower() == 'standard':
            logger.warn(f"Architecture {architecture} not tested in a long time, so use at own risk")
            import orthoseg.model.model_unet_standard as m
            if init_weights_with is not None:
                init_weights = True
            else:
                init_weights = False
            return m.get_model(input_width=input_width, input_height=input_height,
                               nb_channels=nb_channels, nb_classes=nb_classes,
                               init_model_weights=init_weights)
        elif encoder.lower() == 'ternaus':
            logger.warn(f"Architecture {architecture} not tested in a long time, so use at own risk")
            import orthoseg.model.model_unet_ternaus as m
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
        model: kr.models.Model,
        optimizer: str,
        optimizer_params: dict,
        loss: str,
        metrics: List[str] = None,
        sample_weight_mode: str = None,
        class_weights: list = None) -> kr.models.Model:
    """
    Compile the model for training.
    
    Args:
        model (kr.models.Model): The keras model to compile.
        optimizer (str): The optimizer to use.
        optimizer_params (dict): Paramters to use for optimizer.
        loss (str): The loss function to use. One of:
            * categorical_crossentropy
            * weighted_categorical_crossentropy: class_weights should be specified!

        metrics (List[Metric], optional): Metrics to use. Defaults to None. Possible values:
            * 
        sample_weight_mode (str, optional): Sample weight mode to use. Defaults to None.
        class_weights (list, optional): Class weigths to use. Defaults to None.
    """

    # If no merics specified, use default ones
    metric_funcs: List[Any] = []
    if metrics is None:
        if loss in ['categorical_crossentropy', 'weighted_categorical_crossentropy']:
            metric_funcs.append('categorical_accuracy')
        elif loss == 'sparse_categorical_crossentropy':
            #metrics.append('sparse_categorical_accuracy')
            None
        elif loss == 'binary_crossentropy':
            metric_funcs.append('binary_accuracy')

        iou_score = sm.metrics.IOUScore()
        metric_funcs.append(iou_score)
        f1_score = sm.metrics.FScore()
        metric_funcs.append(f1_score)
        #metric_funcs.append(jaccard_coef_round)
        # metric_funcs=[jaccard_coef, jaccard_coef_flat,
        #          jaccard_coef_int, dice_coef, 'accuracy', 'binary_accuracy']

    else:
        raise Exception("Specifying metrics not yet implemented")

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

    # Create optimizer
    if optimizer == 'adam':
        optimizer_func = kr.optimizers.Adam(**optimizer_params)
    else: 
        raise Exception(f"Error creating optimizer: {optimizer}, with params {optimizer_params}")

    logger.info(f"Compile model with optimizer: optimizer, loss: {loss}, class_weights: {class_weights}")
    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=metric_funcs,
            sample_weight_mode=sample_weight_mode)

    return model

def load_model(
        model_to_use_filepath: Path, 
        compile: bool = True) -> kr.models.Model:
    """
    Load an existing model from a file.

    If loading the architecture + model from the file doesn't work, tries 
    different things to load a model anyway:
    1) looks for a ..._model.json file to load the architecture from,  
       the weights are then loaded from the original file
    2) looks for a ..._hyperparams.json file to create the architecture from, 
       the weights are then loaded from the original file

    Args:
        model_to_use_filepath (Path): The file to load the model from.
        compile (bool, optional): True to get a compiled version back that is 
            ready to train. Defaults to True.

    Raises:
        Exception: [description]

    Returns:
        kr.models.Model: The loaded model.
    """
    errors = []
    model = None
    
    # If it is a file with the complete model, try loading it entirely...
    if not model_to_use_filepath.stem.endswith('_weights'):
        iou_score = sm.metrics.IOUScore()
        f1_score = sm.metrics.FScore()

        try:
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
        except Exception as ex:
            errors.append(f"Error loading model+weights from {model_to_use_filepath}: {ex}")
    
    # If no model returned yet, try loading loading architecture and weights seperately
    if model is None:
        # Load the architecture from a model.json file
        model_basestem = f"{'_'.join(model_to_use_filepath.stem.split('_')[0:2])}"

        # Check if there is a specific model.json file for these weights
        model_json_filepath = model_to_use_filepath.parent / (model_to_use_filepath.stem.replace('_weights', '') + '.json')
        if not model_json_filepath.exists():
            # If not, check if there is a model.json file for the training session
            model_json_filepath = model_to_use_filepath.parent / f"{model_basestem}_model.json"
        if model_json_filepath.exists():
            with model_json_filepath.open('r') as src:
                model_json = src.read()
            try:
                model = kr.models.model_from_json(model_json)
            except Exception as ex:
                errors.append(f"Error loading model architecture from {model_json_filepath}: {ex}")
        else:
            errors.append(f"No model.json file found to load model from: {model_json_filepath}")
        
        # If loading model.json not successfull, create model based on hyperparams.json
        if model is None:
            # Load the hyperparams file if available
            hyperparams_json_filepath = model_to_use_filepath.parent / f"{model_basestem}_hyperparams.json"
            if hyperparams_json_filepath.exists():
                with hyperparams_json_filepath.open('r') as src:
                    hyperparams = json.load(src)
                # Create the model we want to use
                try:
                    model = get_model(
                            architecture=hyperparams['architecture']['architecture'],
                            nb_channels=hyperparams['architecture']['nb_channels'], 
                            nb_classes=hyperparams['architecture']['nb_classes'], 
                            activation=hyperparams['architecture']['activation_function'])
                except Exception as ex:
                    errors.append(f"Exception trying get_model() with parameters from: {hyperparams_json_filepath}: {ex}")
            else:
                errors.append(f"No hyperparams.json file found to load model from: {hyperparams_json_filepath}")

        # Now load the weights
        if model is not None:
            try:
                model.load_weights(str(model_to_use_filepath))
            except Exception as ex:
                errors.append(f"Exception trying model.load_weights on: {model_to_use_filepath}: {ex}")

    # Check if a model got loaded...
    if model is not None:
        # Eager seems to be 50% slower, tested on tensorflow 2.5
        model.run_eagerly = False
    else:
        # If we still have not model... time to give up.
        errors_str = ""
        if len(errors) > 0:
            errors_str = (" The following errors occured while trying: \n    -> " + "\n    -> ".join(errors))
        raise Exception(f"Error loading model for {model_to_use_filepath}.{errors_str}")

    return model # type: ignore

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
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon) # type: ignore
            weighted_losses = target * tf.math.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1) # type: ignore
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')
    return loss

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = tf.convert_to_tensor(kr.backend.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon) # type: ignore
    prediction_tensor = kr.backend.log(prediction_tensor / (1 - prediction_tensor)) # type: ignore

    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.cast(
            tf.sigmoid(prediction_tensor) > 0.5, tf.float32)
    return kr.backend.mean(tf.nn.sigmoid_cross_entropy_with_logits(
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

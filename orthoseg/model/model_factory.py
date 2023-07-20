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
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf
import keras.models

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
os.environ['SM_FRAMEWORK'] = 'tf.keras'

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------

"""
preprocessing_fn = get_preprocessing('resnet34')
x = preprocessing_fn(x)
"""


def get_model(
    architecture: str,
    input_width: Optional[int] = None,
    input_height: Optional[int] = None,
    nb_channels: int = 3,
    nb_classes: int = 1,
    activation: str = "softmax",
    init_weights_with: str = "imagenet",
    freeze: bool = False,
) -> keras.models.Model:
    """
    Get a model.

    Args:
        architecture (str): Architecture of the network to create
        input_width (int, optional): Width of the input images. Defaults to None.
        input_height (int, optional): Height of the input images. Defaults to None.
        nb_channels (int, optional): Nb of channels/bands of the input images.
            Defaults to 3.
        nb_classes (int, optional): Nb of classes to be segmented to. Defaults to 1.
        activation (Activation, optional): Activation function of last layer.
            Defaults to 'softmax'.
        init_weights_with (str, optional): Weights to init the network with.
            Defaults to 'imagenet'.
        freeze (bool, optional): Freeze the final layer weights during
            training. It is usefull to use this option for the first few
            epochs get a more robust network. Defaults to False.

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        [type]: [description]
    """
    # Check architecture
    segment_architecture_parts = architecture.split("+")
    if len(segment_architecture_parts) < 2:
        raise Exception(f"Unsupported architecture: {architecture}")
    encoder = segment_architecture_parts[0]
    decoder = segment_architecture_parts[1]

    if decoder.lower() == "unet":
        # Architecture implemented using the segmentation_models library
        from segmentation_models import Unet

        model = Unet(
            backbone_name=encoder.lower(),
            input_shape=(input_width, input_height, nb_channels),
            classes=nb_classes,
            activation=activation,
            encoder_weights=init_weights_with,
            encoder_freeze=freeze,
        )
        return model

    elif decoder.lower() == "pspnet":
        # Architecture implemented using the segmentation_models library
        from segmentation_models import PSPNet

        model = PSPNet(
            backbone_name=encoder.lower(),
            input_shape=(input_width, input_height, nb_channels),
            classes=nb_classes,
            activation=activation,
            encoder_weights=init_weights_with,
            encoder_freeze=freeze,
        )
        return model

    elif decoder.lower() == "linknet":
        # Architecture implemented using the segmentation_models library
        from segmentation_models import Linknet

        # First check if input size is compatible with linknet
        if input_width is not None and input_height is not None:
            check_image_size(decoder, input_width, input_height)

        model = Linknet(
            backbone_name=encoder.lower(),
            input_shape=(input_width, input_height, nb_channels),
            classes=nb_classes,
            activation=activation,
            encoder_weights=init_weights_with,
            encoder_freeze=freeze,
        )
        return model

    else:
        raise ValueError(f"Unknown decoder architecture: {decoder}")


def compile_model(
    model: keras.models.Model,
    optimizer: str,
    optimizer_params: dict,
    loss: str,
    metrics: Optional[List[str]] = None,
    sample_weight_mode: Optional[str] = None,
    class_weights: Optional[list] = None,
) -> keras.models.Model:
    """
    Compile the model for training.

    Args:
        model (keras.models.Model): the keras model to compile.
        optimizer (str): the optimizer to use.
        optimizer_params (dict): parameters to use for optimizer.
        loss (str): the loss function to use. One of:
            * categorical_crossentropy
            * weighted_categorical_crossentropy: class_weights should be specified!

        metrics (List[Metric], optional): metrics to use. Defaults to None. One of:
            *
        sample_weight_mode (str, optional): sample weight mode to use. Defaults to None.
        class_weights (list, optional): class weigths to use. Defaults to None.
    """
    import segmentation_models

    # Get number classes of model
    nb_classes = model.output[-1].shape[-1]

    # If no metrics specified, use default ones
    metric_funcs: List[Any] = []
    if metrics is None:
        if loss in ["categorical_crossentropy", "weighted_categorical_crossentropy"]:
            metric_funcs.append("categorical_accuracy")
        elif loss == "sparse_categorical_crossentropy":
            # metrics.append('sparse_categorical_accuracy')
            logger.warning("loss == sparse_categorical_crossentropy not implementedd")
        elif loss == "binary_crossentropy":
            metric_funcs.append("binary_accuracy")

        onehot_mean_iou = tf.keras.metrics.OneHotMeanIoU(
            num_classes=nb_classes, name="one_hot_mean_iou"
        )
        metric_funcs.append(onehot_mean_iou)
    else:
        raise Exception("Specifying metrics not yet implemented")

    # Check loss function
    if loss == "bcedice":
        loss_func = dice_coef_loss_bce
    elif loss == "dice_loss":
        loss_func = segmentation_models.losses.DiceLoss()
    elif loss == "jaccard_loss":
        loss_func = segmentation_models.losses.JaccardLoss()
    elif loss == "weighted_categorical_crossentropy":
        if class_weights is None:
            raise Exception(f"With loss == {loss}, class_weights cannot be None!")
        loss_func = weighted_categorical_crossentropy(class_weights)
    else:
        loss_func = loss

    # Create optimizer
    if optimizer == "adam":
        optimizer_func = tf.keras.optimizers.Adam(**optimizer_params)
    else:
        raise Exception(
            f"Error creating optimizer: {optimizer}, with params {optimizer_params}"
        )

    logger.info(
        f"Compile model, optimizer: {optimizer}, loss: {loss}, "
        f"class_weights: {class_weights}"
    )
    model.compile(
        optimizer=optimizer_func,
        loss=loss_func,
        metrics=metric_funcs,
        sample_weight_mode=sample_weight_mode,
    )

    return model


def load_model(model_to_use_filepath: Path, compile: bool = True) -> keras.models.Model:
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
        tf.keras.models.Model: The loaded model.
    """
    import segmentation_models

    errors = []
    model = None
    model_basestem = f"{'_'.join(model_to_use_filepath.stem.split('_')[0:2])}"

    # If it is a file with the complete model, try loading it entirely...
    if not model_to_use_filepath.stem.endswith("_weights"):
        # Load the hyperparams file
        hyperparams_json_filepath = (
            model_to_use_filepath.parent / f"{model_basestem}_hyperparams.json"
        )
        nb_classes = 1
        if hyperparams_json_filepath.exists():
            with hyperparams_json_filepath.open("r") as src:
                hyperparams = json.load(src)
                nb_classes = len(hyperparams["architecture"]["classes"])

        iou_score = segmentation_models.metrics.IOUScore()
        f1_score = segmentation_models.metrics.FScore()
        onehot_mean_iou = tf.keras.metrics.OneHotMeanIoU(
            num_classes=nb_classes, name="one_hot_mean_iou"
        )

        try:
            model = tf.keras.models.load_model(
                str(model_to_use_filepath),
                custom_objects={
                    "jaccard_coef": jaccard_coef,
                    "jaccard_coef_flat": jaccard_coef_flat,
                    "jaccard_coef_round": jaccard_coef_round,
                    "dice_coef": dice_coef,
                    "iou_score": iou_score,
                    "f1_score": f1_score,
                    "one_hot_mean_iou": onehot_mean_iou,
                    "weighted_categorical_crossentropy": weighted_categorical_crossentropy,  # noqa: E501
                },
                compile=compile,
            )
        except Exception as ex:
            errors.append(
                f"Error loading model+weights from {model_to_use_filepath}: {ex}"
            )

    # If no model returned yet, try loading loading architecture and weights seperately
    if model is None:
        # Load the architecture from a model.json file
        # Check if there is a specific model.json file for these weights
        model_json_filepath = model_to_use_filepath.parent / (
            model_to_use_filepath.stem.replace("_weights", "") + ".json"
        )
        if not model_json_filepath.exists():
            # If not, check if there is a model.json file for the training session
            model_json_filepath = (
                model_to_use_filepath.parent / f"{model_basestem}_model.json"
            )
        if model_json_filepath.exists():
            with model_json_filepath.open("r") as src:
                model_json = src.read()
            try:
                model = tf.keras.models.model_from_json(model_json)
            except Exception as ex:
                errors.append(
                    f"Error loading model architecture from {model_json_filepath}: {ex}"
                )
        else:
            errors.append(
                f"No model.json file found to load model from: {model_json_filepath}"
            )

        # If loading model.json not successfull, create model based on hyperparams.json
        if model is None:
            # Load the hyperparams file if available
            hyperparams_json_filepath = (
                model_to_use_filepath.parent / f"{model_basestem}_hyperparams.json"
            )
            if hyperparams_json_filepath.exists():
                with hyperparams_json_filepath.open("r") as src:
                    hyperparams = json.load(src)
                # Create the model we want to use
                try:
                    model = get_model(
                        architecture=hyperparams["architecture"]["architecture"],
                        nb_channels=hyperparams["architecture"]["nb_channels"],
                        nb_classes=hyperparams["architecture"]["nb_classes"],
                        activation=hyperparams["architecture"]["activation_function"],
                    )
                except Exception as ex:
                    errors.append(
                        "Error in get_model() with params from: "
                        f"{hyperparams_json_filepath}: {ex}"
                    )
            else:
                errors.append(
                    f"No file found to load model from: {hyperparams_json_filepath}"
                )

        # Now load the weights
        if model is not None:
            try:
                model.load_weights(str(model_to_use_filepath))
            except Exception as ex:
                errors.append(
                    f"Error trying model.load_weights on: {model_to_use_filepath}: {ex}"
                )

    # Check if a model got loaded...
    if model is not None:
        # Eager seems to be 50% slower, tested on tensorflow 2.5
        model.run_eagerly = False
    else:
        # If we still have not model... time to give up.
        errors_str = ""
        if len(errors) > 0:
            errors_str = (
                " The following errors occured while trying: \n    -> "
                + "\n    -> ".join(errors)
            )
        raise Exception(f"Error loading model for {model_to_use_filepath}.{errors_str}")

    return model  # type: ignore


def set_trainable(model, recompile: bool = True):
    import segmentation_models
    segmentation_models.utils.set_trainable(
        model=model, recompile=recompile
    )  # doesn't seem to work, so save and load model


def check_image_size(decoder: str, input_width: int, input_height: int):
    if decoder.lower() == "linknet":
        if (input_width and (input_width % 16) != 0) or (
            input_height and (input_height % 16) != 0
        ):
            message = (
                f"for decoder linknet, input_width ({input_width} and input_height "
                f"({input_height}) should be divisable by 16!"
            )
            logger.error(message)
            raise ValueError(message)
    else:
        logger.info(f"check_image_size is not implemented for this model: {decoder}")


# ------------------------------------------
# Loss functions
# ------------------------------------------


def weighted_categorical_crossentropy(weights):
    """weighted_categorical_crossentropy

    Args:
        * weights<ktensor|nparray|list>: crossentropy weights
    Returns:
        * weighted categorical crossentropy function
    """
    if isinstance(weights, list) or isinstance(weights, np.ndarray):
        weights = tf.keras.backend.variable(weights)

    def loss(target, output, from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output, len(output.get_shape()) - 1, True)
            _epsilon = tf.convert_to_tensor(
                tf.keras.backend.epsilon(), dtype=output.dtype.base_dtype
            )
            output = tf.clip_by_value(output, _epsilon, 1.0 - _epsilon)  # type: ignore
            weighted_losses = target * tf.math.log(output) * weights
            retval = -tf.reduce_sum(
                weighted_losses, len(output.get_shape()) - 1  # type: ignore
            )
            return retval
        else:
            raise ValueError("WeightedCategoricalCrossentropy: not valid with logits")

    return loss


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type="hard", alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = tf.convert_to_tensor(
        tf.keras.backend.epsilon(), prediction_tensor.dtype.base_dtype
    )
    prediction_tensor = tf.clip_by_value(
        prediction_tensor, _epsilon, 1 - _epsilon  # type: ignore
    )
    prediction_tensor = tf.keras.backend.log(
        prediction_tensor / (1 - prediction_tensor)  # type: ignore
    )

    if bootstrap_type == "soft":
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.sigmoid(
            prediction_tensor
        )
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.cast(
            tf.sigmoid(prediction_tensor) > 0.5, tf.float32
        )  # type: ignore
    return tf.keras.backend.mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=bootstrap_target_tensor, logits=prediction_tensor
        )
    )


def dice_coef_loss_bce(y_true, y_pred):
    dice = 0.5
    bce = 0.5
    bootstrapping = "hard"
    alpha = 1.0
    return (
        bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce
        + dice_coef_loss(y_true, y_pred) * dice
    )


# ------------------------------------------
# Metrics functions
# ------------------------------------------

SMOOTH_LOSS = 1e-12


def jaccard_coef(y_true, y_pred):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = tf.keras.backend.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)

    return tf.keras.backend.mean(jac)


def jaccard_coef_round(y_true, y_pred):
    y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))

    intersection = tf.keras.backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = tf.keras.backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return tf.keras.backend.mean(jac)


def jaccard_coef_flat(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + SMOOTH_LOSS) / (
        tf.keras.backend.sum(y_true_f)
        + tf.keras.backend.sum(y_pred_f)
        - intersection
        + SMOOTH_LOSS
    )


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def pct_wrong(y_true, y_pred):
    y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))

    intersection = tf.keras.backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = tf.keras.backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return tf.keras.backend.mean(jac)

"""Module with helper functions to create models.

Offers a common interface, regardless of the underlying model implementation
and contains extra metrics, callbacks,...

Many models are supported by using this segmentation model zoo:
https://github.com/qubvel/segmentation_models
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import keras
import keras.models
import segmodels_keras as smk
import tensorflow as tf
from segmodels_keras import Linknet, PSPNet, Unet

from orthoseg._compat import KERAS_GTE_3

if KERAS_GTE_3:
    import keras
    from keras import ops
else:
    from keras import backend as ops


# Get a logger...
logger = logging.getLogger(__name__)


def get_model(
    architecture: str,
    input_width: int | None = None,
    input_height: int | None = None,
    nb_channels: int = 3,
    nb_classes: int = 1,
    activation: str = "softmax",
    init_weights_with: str = "imagenet",
    freeze: bool = False,
) -> tuple[keras.models.Model, Callable | None]:
    """Get a model.

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

    Returns:
        Model: the model.
    """
    # Check architecture
    segment_architecture_parts = architecture.split("+")
    if len(segment_architecture_parts) < 2:
        raise Exception(f"Unsupported architecture: {architecture}")
    encoder = segment_architecture_parts[0]
    decoder = segment_architecture_parts[1]

    if decoder.lower() == "unet":
        # Architecture implemented using the segmentation_models library
        model = Unet(
            backbone_name=encoder.lower(),
            input_shape=(input_width, input_height, nb_channels),
            classes=nb_classes,
            activation=activation,
            encoder_weights=init_weights_with,
            encoder_freeze=freeze,
        )

    elif decoder.lower() == "pspnet":
        # Architecture implemented using the segmentation_models library
        model = PSPNet(
            backbone_name=encoder.lower(),
            input_shape=(input_width, input_height, nb_channels),
            classes=nb_classes,
            activation=activation,
            encoder_weights=init_weights_with,
            encoder_freeze=freeze,
        )

    elif decoder.lower() == "linknet":
        # Architecture implemented using the segmentation_models library
        # First check if input size is compatible with linknet
        if input_width is not None and input_height is not None:
            check_image_size(architecture, input_width, input_height)

        model = Linknet(
            backbone_name=encoder.lower(),
            input_shape=(input_width, input_height, nb_channels),
            classes=nb_classes,
            activation=activation,
            encoder_weights=init_weights_with,
            encoder_freeze=freeze,
        )

    else:
        raise ValueError(f"Unknown decoder architecture: {decoder}")

    model_preprocess_input = get_preprocess_input(architecture)
    return model, model_preprocess_input


def get_custom_objects(architecture: str) -> dict[str, Callable]:
    """Get the custom objects to use for loading models."""
    segment_architecture_parts = architecture.split("+")
    if len(segment_architecture_parts) < 2:
        raise ValueError(f"Unsupported architecture: {architecture}")

    encoder = segment_architecture_parts[0]
    return smk.Backbones.get_custom_objects(encoder.lower())


def get_preprocess_input(architecture: str) -> Callable:
    """Get the default input preprocess function for a given architecture.

    The preprocess_input function should be used to preprocess input images when using a
    model with the given architecture.

    Something typical that might be done in the preprocessing function is to rescale
    pixel values to the range [-1, 1].

    Args:
        architecture (str): the architecture to get the preprocessing function for.
    """
    # Check architecture
    segment_architecture_parts = architecture.split("+")
    if len(segment_architecture_parts) < 2:
        raise ValueError(f"Unsupported architecture: {architecture}")

    encoder = segment_architecture_parts[0]
    preprocess_input_func = smk.get_preprocessing(encoder.lower())

    return preprocess_input_func


def get_preprocess_input_rescale(rescale_factor: float) -> Callable:
    """Get a preprocessing function that applies the rescale factor provided.

    Args:
        rescale_factor (float): the factor to rescale pixel values with.
            E.g. 1/255 or 0.0039215686274509803921568627451 to rescale pixel values to
            the range [0, 1].

    Returns:
        Callable: the default preprocessing function.
    """
    return lambda x: _rescale(x, rescale_factor)


def _rescale(x, rescale_factor: float) -> Callable:
    """Get a preprocessing function that applies the rescale factor provided.

    Args:
        x: the input array, 3D or 4D, to preprocess.
        rescale_factor (float): the factor to rescale pixel values with.
            E.g. 1/255 or 0.0039215686274509803921568627451 to rescale pixel values to
            the range [0, 1].

    Returns:
        The rescaled input array.
    """
    return x * rescale_factor


def compile_model(
    model: keras.models.Model,
    optimizer: str,
    optimizer_params: dict,
    loss: str,
    metrics: list[str] | None = None,
    class_weights: list[float] | None = None,
) -> keras.models.Model:
    """Compile the model for training.

    Args:
        model (keras.models.Model): the keras model to compile.
        optimizer (str): the optimizer to use.
        optimizer_params (dict): parameters to use for optimizer.
        loss (str): the loss function to use. One of:
            * categorical_crossentropy
            * categorical_focal_crossentropy
            * weighted_categorical_crossentropy is deprecated as it gives issues with
              keras 3 combined with torch as backend.

        metrics (list[str], optional): metrics to use. Support to specify them is not
            implemented... so should be None. Defaults to None.
        class_weights (list[float], optional): class weights to use for the loss
            function. Defaults to None.
    """
    # Get number classes of model
    nb_classes = model.output[-1].shape[-1]

    # If no metrics specified, use default ones
    metric_funcs: list[Any] = []
    if metrics is None:
        if loss in [
            "categorical_crossentropy",
            "weighted_categorical_crossentropy",
            "categorical_focal_crossentropy",
        ]:
            metric_funcs.append("categorical_accuracy")
        elif loss == "sparse_categorical_crossentropy":
            # metrics.append('sparse_categorical_accuracy')
            logger.warning("loss == sparse_categorical_crossentropy not implementedd")
        elif loss == "binary_crossentropy":
            metric_funcs.append("binary_accuracy")

        onehot_mean_iou = keras.metrics.OneHotMeanIoU(
            num_classes=nb_classes, name="one_hot_mean_iou"
        )
        metric_funcs.append(onehot_mean_iou)
    else:
        raise ValueError("Specifying metrics not yet implemented")

    # Check loss function
    loss_func: Callable | str
    if loss == "bcedice":
        loss_func = dice_coef_loss_bce
    elif loss == "dice_loss":
        loss_func = smk.losses.DiceLoss()
    elif loss == "jaccard_loss":
        loss_func = smk.losses.JaccardLoss()
    elif loss == "weighted_categorical_crossentropy":
        # Remark: in keras it is possible to use a class_weight parameter of model.fit
        # to specify class weights. But, this option was implemented for timeseries data
        # and only supports 2D input data, otherwise the following error is given:
        # "class_weight not supported for 3+ dimensional targets". With computer vision
        # you typically have 3D+ input data, so it doesn't work.
        # Hence: use a custom weighted loss function!
        if class_weights is None:
            raise ValueError(f"With loss == {loss}, class_weights cannot be None!")
        loss_func = weighted_categorical_crossentropy(class_weights)
    elif loss == "categorical_focal_crossentropy":
        kwargs = {"alpha": class_weights} if class_weights is not None else {}
        loss_func = keras.losses.CategoricalFocalCrossentropy(**kwargs)
    else:
        loss_func = loss

    # Create optimizer
    if optimizer == "adam":
        optimizer_func = keras.optimizers.Adam(**optimizer_params)
    else:
        raise ValueError(
            f"Error creating optimizer: {optimizer}, with params {optimizer_params}"
        )

    logger.info(
        f"Compile model, optimizer: {optimizer}, loss: {loss}, "
        f"class_weights: {class_weights}"
    )
    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=metric_funcs)

    return model


def load_model(
    model_to_use_filepath: Path, compile_model: bool = True
) -> tuple[keras.models.Model, Callable | None]:
    """Load an existing model from a file.

    If loading the architecture + model from the file doesn't work, tries
    different things to load a model anyway:
    1) looks for a ..._model.json file to load the architecture from,
       the weights are then loaded from the original file
    2) looks for a ..._hyperparams.json file to create the architecture from,
       the weights are then loaded from the original file

    Args:
        model_to_use_filepath (Path): The file to load the model from.
        compile_model (bool, optional): True to get a compiled version back that is
            ready to train. Defaults to True.

    Raises:
        RuntimeError: If the hyperparams file is not found or if loading the model gave
            an error.

    Returns:
        A tuple of (keras.models.Model, Callable): The loaded model and the function to
        use to preprocess input images when using the model.
    """
    errors = []
    model = None

    model_basestem = f"{'_'.join(model_to_use_filepath.stem.split('_')[0:2])}"

    # Load the hyperparams file
    hyperparams_json_filepath = (
        model_to_use_filepath.parent / f"{model_basestem}_hyperparams.json"
    )
    nb_classes = 1
    if not hyperparams_json_filepath.exists():
        raise FileNotFoundError(
            f"No hyperparams file found for model: {hyperparams_json_filepath}"
        )

    with hyperparams_json_filepath.open("r") as src:
        hyperparams = json.load(src)
        nb_classes = len(hyperparams["architecture"]["classes"])
        train_rescale_factor = hyperparams["train"]["image_augmentations"].get(
            "rescale", None
        )

    # If it is a file with the complete model, try loading it entirely...
    if not model_to_use_filepath.stem.endswith("_weights"):
        # iou_score = segmentation_models.metrics.IOUScore()
        # f1_score = segmentation_models.metrics.FScore()
        iou_score = keras.metrics.IoU(
            num_classes=nb_classes,
            target_class_ids=list(range(nb_classes)),
            name="iou_score",
        )
        onehot_mean_iou = keras.metrics.OneHotMeanIoU(
            num_classes=nb_classes, name="one_hot_mean_iou"
        )

        load_model_kwargs = {}

        custom_objects = {
            "jaccard_coef": jaccard_coef,
            "jaccard_coef_flat": jaccard_coef_flat,
            "jaccard_coef_round": jaccard_coef_round,
            "dice_coef": dice_coef,
            "iou_score": iou_score,
            # "f1_score": f1_score,
            "one_hot_mean_iou": onehot_mean_iou,
            "weighted_categorical_crossentropy": weighted_categorical_crossentropy,
        }
        if KERAS_GTE_3:
            load_model_kwargs["safe_mode"] = False
            custom_objects["categorical_focal_crossentropy"] = (
                keras.losses.CategoricalFocalCrossentropy
            )

        # Add custom objects for the architecture used.
        custom_objects.update(
            get_custom_objects(hyperparams["architecture"]["architecture"])
        )
        try:
            model = keras.models.load_model(
                str(model_to_use_filepath),
                custom_objects=custom_objects,
                compile=compile_model,
                **load_model_kwargs,
            )
        except Exception as ex:
            errors.append(
                f"Error loading model+weights from {model_to_use_filepath}: {ex}"
            )

            # Try to make the saved model file keras 3 compatible and retry.
            if KERAS_GTE_3 and str(ex).startswith(
                "Error when deserializing class 'DepthwiseConv2D' using config"
            ):
                logger.warning(
                    "Error loading model file, try to make it keras 3 compatible and "
                    "retry."
                )

                # Ref: https://github.com/keras-team/keras/issues/19441
                # Hack to change model config from keras 2->3 compliant
                f = h5py.File(str(model_to_use_filepath), mode="r+")
                model_config_string = f.attrs.get("model_config")
                if model_config_string.find('"groups": 1,') != -1:
                    model_config_string = model_config_string.replace(
                        '"groups": 1,', ""
                    )
                    f.attrs.modify("model_config", model_config_string)
                    f.flush()
                    model_config_string = f.attrs.get("model_config")
                    assert model_config_string.find('"groups": 1,') == -1

                f.close()

                try:
                    model, _model_preprocess_input = keras.models.load_model(
                        str(model_to_use_filepath),
                        custom_objects=custom_objects,
                        compile=compile_model,
                        **load_model_kwargs,
                    )
                except Exception as ex2:
                    message = (
                        "Error loading model file, after trying to make it keras 3 "
                        f"compatible: {ex2=}"
                    )
                    logger.warning(message)
                    errors.append(message)

            if model is None and compile_model:
                logger.warning(
                    "Error loading model+weights from file. Will try loading "
                    "architecture and weights separately but this won't restore the "
                    f"optimizer state: {ex=}"
                )

    # If no model loaded yet, try loading loading architecture and weights separately
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
                model = keras.models.model_from_json(model_json)
            except Exception as ex:
                errors.append(
                    f"Error loading model architecture from {model_json_filepath}: {ex}"
                )
        else:
            errors.append(
                f"No model.json file found to load model from: {model_json_filepath}"
            )

        # If loading model.json not successful, create model based on hyperparams.json
        if model is None:
            # Create the model we want to use
            try:
                model, _model_preprocess_input = get_model(
                    architecture=hyperparams["architecture"]["architecture"],
                    nb_channels=hyperparams["architecture"]["nb_channels"],
                    nb_classes=len(hyperparams["architecture"]["classes"]),
                    activation=hyperparams["architecture"]["activation_function"],
                )
            except Exception as ex:
                errors.append(
                    "Error in get_model() with params from: "
                    f"{hyperparams_json_filepath}: {ex}"
                )

        # Now load the weights
        if model is not None:
            try:
                model.load_weights(str(model_to_use_filepath))
            except RuntimeError as ex:
                errors.append(
                    f"Error trying model.load_weights on: {model_to_use_filepath}: {ex}"
                )

    if model is None:
        # If we still have not model... time to give up.
        errors_str = ""
        if len(errors) > 0:
            errors_str = (
                " The following errors occured while trying: \n    -> "
                + "\n    -> ".join(errors)
            )
        raise RuntimeError(
            f"Error loading model for {model_to_use_filepath}.{errors_str}"
        )

    if train_rescale_factor is not None:
        # For backwards compatibility, if a rescale factor was still specified in the
        # augmentation hyperparams, use a custom preprocessing function that applies
        # this rescale factor.
        # Otherwise, use the default preprocessing function for the architecture.
        preprocess_input_func = get_preprocess_input_rescale(train_rescale_factor)
    else:
        preprocess_input_func = get_preprocess_input(
            hyperparams["architecture"]["architecture"]
        )

    # Eager seems slower: 50% slower on tf 2.5, 15% slower on tf 2.10.
    # model.run_eagerly = True

    return model, preprocess_input_func


def set_trainable(model, recompile: bool = True):
    """Set the model trainable.

    Args:
        model (_type_): model to set trainable.
        recompile (bool, optional): True to recompile the model so it is ready to train.
            Defaults to True.
    """
    # doesn't seem to work, so save and load model
    smk.utils.set_trainable(model=model, recompile=recompile)


def check_image_size(architecture: str, input_width: int, input_height: int):
    """Check if the image size is compatible with the architecture.

    A ValueError is raised if the architecture is not compatible with the size.

    Args:
        architecture (str): architecture to check compatibility for.
        input_width (int): image width.
        input_height (int): image height.
    """
    # Check architecture
    segment_architecture_parts = architecture.split("+")
    if len(segment_architecture_parts) < 2:
        raise ValueError(f"Unsupported architecture: {architecture}")
    # encoder = segment_architecture_parts[0]
    decoder = segment_architecture_parts[1]

    if decoder.lower() == "linknet":
        if (input_width and (input_width % 32) != 0) or (
            input_height and (input_height % 32) != 0
        ):
            message = (
                f"for decoder linknet, input image width ({input_width}) and "
                f"input image height ({input_height}) should be divisible by 32!"
            )
            logger.error(message)
            raise ValueError(message)


# ------------------------------------------
# Loss functions
# ------------------------------------------


def weighted_categorical_crossentropy(weights):
    """Loss function using weighted categorical crossentropy.

    Args:
        weights (ktensor|nparray|list): crossentropy weights
    Returns:
        weighted categorical crossentropy function
    """

    def loss(target, output, from_logits=False):
        if not from_logits:
            output /= ops.sum(output, len(output.shape) - 1, True)
            output = ops.clip(
                output, keras.backend.epsilon(), 1.0 - keras.backend.epsilon()
            )
            weighted_losses = target * ops.log(output) * weights
            retval = -ops.sum(weighted_losses, len(output.shape) - 1)
            return retval
        else:
            raise ValueError("WeightedCategoricalCrossentropy: not valid with logits")

    return loss


def dice_coef_loss(y_true, y_pred):
    """Loss function based of dice coefficient.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 - dice_coef(y_true, y_pred)


def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type="hard", alpha=0.95):
    """Loss function based on cross entropy with a bootstrap.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        bootstrap_type (str, optional): _description_. Defaults to "hard".
        alpha (float, optional): _description_. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = tf.convert_to_tensor(
        tf.keras.backend.epsilon(), prediction_tensor.dtype.base_dtype
    )
    prediction_tensor = tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = tf.keras.backend.log(
        prediction_tensor / (1 - prediction_tensor)
    )

    if bootstrap_type == "soft":
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.sigmoid(
            prediction_tensor
        )
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.cast(
            tf.sigmoid(prediction_tensor) > 0.5, tf.float32
        )
    return tf.keras.backend.mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=bootstrap_target_tensor, logits=prediction_tensor
        )
    )


def dice_coef_loss_bce(y_true, y_pred):
    """Loss function based on dice coefficient with bootstrapping.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """Metric jaccard coefficient aka intersection over union.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = tf.keras.backend.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)

    return tf.keras.backend.mean(jac)


def jaccard_coef_round(y_true, y_pred):
    """Metric jaccard coefficient aka intersection over union with rounding.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))

    intersection = tf.keras.backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = tf.keras.backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return tf.keras.backend.mean(jac)


def jaccard_coef_flat(y_true, y_pred):
    """Metric jaccard coefficient aka intersection over union with flattening.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """Metric dice coefficient.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        smooth (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def pct_wrong(y_true, y_pred):
    """Metric percentage wrong.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))

    intersection = tf.keras.backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = tf.keras.backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + SMOOTH_LOSS) / (sum_ - intersection + SMOOTH_LOSS)
    return tf.keras.backend.mean(jac)

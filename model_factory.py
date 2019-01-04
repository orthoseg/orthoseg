# -*- coding: utf-8 -*-
"""
Module with helper functions to create models.

Offers a common interface, regardless of the underlying model implementation 
and contains extra metrics, callbacks,...

Many models are supported by using the segmentation model zoo:
https://github.com/qubvel/segmentation_models

@author: Pieter Roggemans
"""

import os
import logging
import glob

import keras as kr

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

def get_model(encoder: str = 'inceptionresnetv2',
              decoder: str = 'linknet',
              input_width: int = None,
              input_height: int = None,
              n_channels: int = 3,
              n_classes: int = 1,
              init_weights_with: str = 'imagenet'):

    if decoder.lower() == 'deeplabv3plus':
        import model_deeplabv3plus as m
        return m.get_model(input_width=input_width, input_height=input_height,
                           n_channels=n_channels, n_classes=n_classes,
                           init_model_weights=init_weights_with)
    elif decoder.lower() == 'unet':
        # These two unet variants are implemented in a seperate module
        if encoder.lower() == 'standard':
            import model_unet_standard as m
            if init_weights_with:
                init_weights = True
            return m.get_model(input_width=input_width, input_height=input_height,
                               n_channels=n_channels, n_classes=n_classes,
                               init_model_weights=init_weights)
        elif encoder.lower() == 'ternaus':
            import model_unet_ternaus as m
            if init_weights_with:
                init_weights = True
            return m.get_model(input_width=input_width, input_height=input_height,
                               n_channels=n_channels, n_classes=n_classes,
                               init_model_weights=init_weights)

        # Some other unet variants is implemented using the segmentation_models library
        from segmentation_models import Unet
        #from segmentation_models.backbones import get_preprocessing

        model = Unet(backbone_name=encoder,
                     input_shape=(input_width, input_height, n_channels),
                     classes=n_classes,
                     encoder_weights=init_weights_with)
        return model
    elif decoder.lower() == 'pspnet':
        from segmentation_models import PSPNet
        #from segmentation_models.backbones import get_preprocessing

        model = PSPNet(backbone_name=encoder,
                       input_shape=(input_width, input_height, n_channels),
                       classes=n_classes,
                       encoder_weights=init_weights_with)
        return model
    elif decoder.lower() == 'linknet':
        from segmentation_models import Linknet
        #from segmentation_models.backbones import get_preprocessing

        # First check if input size is compatible with linknet 
        check_image_size(decoder, input_width, input_height)
            
        model = Linknet(backbone_name=encoder,
                        input_shape=(input_width, input_height, n_channels),
                        classes=n_classes,
                        encoder_weights=init_weights_with)
        return model
    else:
        raise Exception(f"Unknown decoder architecture: {decoder}")

def compile_model(model,
                  optimizer,
                  loss_mode='binary_crossentropy',
                  metrics=None):

    if loss_mode == "bcedice":
        loss_func = dice_coef_loss_bce
    elif loss_mode == "binary_crossentropy":
        loss_func = "binary_crossentropy"
    else:
        raise Exception(f"Unknown loss function: {loss_mode}")

    # TODO: implement option to specify metrics...
    model.compile(optimizer=optimizer, loss=loss_func,
                  metrics=[jaccard_coef, jaccard_coef_flat,
                           jaccard_coef_int, dice_coef, 'accuracy', 'binary_accuracy'])

    return model

def load_model(model_to_use_filepath: str):
    model = kr.models.load_model(model_to_use_filepath,
                                 custom_objects={'jaccard_coef': jaccard_coef,
                                                 'jaccard_coef_flat': jaccard_coef_flat,
                                                 'jaccard_coef_int': jaccard_coef_int,
                                                 'dice_coef': dice_coef})

    return model

def check_image_size(decoder: str,
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

def jaccard_coef_int(y_true, y_pred):
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

#------------------------------------------
# Custom callback functions
#------------------------------------------

# TODO: can be cleaned up, eg. now models are always saved, while they might be 
# deleted again a few lines further in the algo... (and in general is quite spaghetti :-(.
class ModelCheckpointExt(kr.callbacks.Callback):
    
    def __init__(self, 
                 model_save_dir: str, 
                 model_save_basename: str,
                 track_metric: str,
                 track_metric_validation: str,
                 only_report: bool = False):
        self.model_save_dir = model_save_dir
        self.model_save_basename = model_save_basename
        self.track_metric = track_metric
        self.track_metric_validation = track_metric_validation
        self.only_report = only_report
        
    # TODO: add verbose parameter
    def on_epoch_end(self, epoch, logs={}):
        logger.debug("Start in callback on_epoch_begin")
        
        track_metric_value = logs.get(self.track_metric)
        track_metric_val_value = logs.get(self.track_metric_validation)
        
        track_metric_avg = (track_metric_value + track_metric_val_value) / 2
        filepath = f"{self.model_save_dir}{os.sep}{self.model_save_basename}_{track_metric_avg:.5f}_{track_metric_value:.5f}_{track_metric_val_value:.5f}_{epoch}.hdf5"
        
        # Todo: is more efficient to only save is necessary...
        self.model.save_weights(filepath)
        
        model_weight_filepaths = glob.glob(f"{self.model_save_dir}{os.sep}{self.model_save_basename}_*.hdf5")
        '''
        if len(model_weight_filepaths) > 0:
            logger.info(f"models found: {model_weight_filepaths}")
        '''
        # Loop through all models to extract necessary info...
        param1_best_value = 0.0
        param2_best_value = 0.0
        avg_param_best_value = 0.0
        param1_for_best_avg = 0.0
        param2_for_best_avg = 0.0

        model_info_list = []
        # Loop through them in reversed order: we want to keep +- the best 10
        for filepath in sorted(model_weight_filepaths, reverse=True):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            param_value_string = filename.replace(self.model_save_basename + "_", "")
            param_values = param_value_string.split("_")
            epoch = param_values[0]
            param1_value = float(param_values[1])
            param2_value = float(param_values[2])
            avg_param_value = (param1_value + param2_value) / 2            
            
            model_info_list.append({'filepath': filepath,
                                    'filename': filename,
                                    'epoch': epoch,
                                    'param1_value': param1_value,
                                    'param2_value': param2_value,
                                    'avg_param_value': avg_param_value})
            
            # Search the best values for the params
            if param1_value > param1_best_value:
                param1_best_value = param1_value
            if param2_value > param2_best_value:
                param2_best_value = param2_value
            if avg_param_value > avg_param_best_value:
                avg_param_best_value = avg_param_value
                param1_for_best_avg = param1_value
                param2_for_best_avg = param2_value
        
        # Now go through them again and cleanup those that aren't great 
        nb_kept = 0
        nb_to_keep_max = 10
        for model_info in model_info_list:
            # Only keep the best models...
            message = ""
            keep = False
            best = False           
            if model_info['avg_param_value'] >= avg_param_best_value:
                best = True
                keep = True
                message += f"BEST AVG of params: {avg_param_best_value:0.4f}; "
            if model_info['param1_value'] >= param1_best_value:
                keep = True
                message += f"best param1: {model_info['param1_value']}; "
            if model_info['param2_value'] >= param2_best_value:
                keep = True
                message += f"best param2: {model_info['param2_value']}; "
            if model_info['param1_value'] > param1_for_best_avg:
                if nb_kept < nb_to_keep_max:
                    keep = True
                    message += f"param1>param1 of best avg: {param1_for_best_avg}; "
            if model_info['param2_value'] > param2_for_best_avg:
                if nb_kept < nb_to_keep_max:
                    keep = True
                    message += f"param2>param2 of best avg: {param2_for_best_avg}; "
            
            # If the model needs to be kept 
            if keep:
                nb_kept += 1
                if best:
                    logger.info(f"KEEP: {message}{model_info['filename']}")
            else:
                logger.info(f"DELETE: model isn't very good and/or max reached; {model_info['filename']}")
                if not self.only_report:
                    os.remove(model_info['filepath'])
            
# -*- coding: utf-8 -*-
"""
High-level API to run a segmentation.
"""

import logging
from pathlib import Path

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import tensorflow as tf
from tensorflow import keras as kr
#import keras as kr      

from orthoseg.helpers import config_helper as conf
import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh
from orthoseg import segment_predict

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def run_prediction():
    """
    Run a prediction of the input dir given.
    """
    
    # TODO: add something to delete old data, predictions???

    # Create base filename of model to use
    # TODO: is force data version the most logical, or rather implement 
    #       force weights file or ?
    force_model_traindata_version = conf.train.getint('force_model_traindata_version')
    if force_model_traindata_version > -1:
        traindata_version = force_model_traindata_version 
    else:
        traindata_version = mh.get_max_data_version(model_dir=conf.dirs.getpath('model_dir'))
        #logger.info(f"max model_traindata_version found: {model_traindata_version}")
    
    # Get the best model that already exists for this train dataset
    best_model = mh.get_best_model(
            model_dir=conf.dirs.getpath('model_dir'), 
            segment_subject=conf.general['segment_subject'],
            model_architecture=conf.model['architecture'],
            train_data_version=traindata_version)
    
    # Check if a model was found
    if best_model is False:
        message = f"No model found in model_dir: {conf.dirs.getpath('model_dir')} for traindata_version: {traindata_version}"
        logger.critical(message)
        raise Exception(message)
    else:    
        model_weights_filepath = best_model['filepath']
        logger.info(f"Best model found: {model_weights_filepath}")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir = f"{best_model['segment_subject']}_{best_model['train_data_version']}_{best_model['model_architecture']}_{best_model['epoch']}"
    
    # Try optimizing model with tensorrt
    try:
        # Try import
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        # Import didn't fail, so optimize model
        logger.info('Tensorrt is available, so use optimized model')
        savedmodel_optim_dir = best_model['filepath'].parent / best_model['filepath'].stem + "_optim"
        if not savedmodel_optim_dir.exists():
            # If base model not yet in savedmodel format
            savedmodel_dir = best_model['filepath'].parent / best_model['filepath'].stem
            if not savedmodel_dir.exists():
                logger.info(f"SavedModel format not yet available, so load model + weights from {best_model['filepath']}")
                model = mf.load_model(best_model['filepath'], compile=False)
                logger.info(f"Now save again as savedmodel to {savedmodel_dir}")
                tf.saved_model.save(model, str(savedmodel_dir))
                del model

            # Now optimize model
            logger.info(f"Optimize + save model to {savedmodel_optim_dir}")
            converter = trt.TrtGraphConverterV2(
                    input_saved_model_dir=savedmodel_dir,
                    is_dynamic_op=True,
                    precision_mode='FP16')
            converter.convert()
            converter.save(savedmodel_optim_dir)
        
        logger.info(f"Load optimized model + weights from {savedmodel_optim_dir}")
        model = tf.keras.models.load_model(savedmodel_optim_dir)

    except ImportError as e:
        logger.info('Tensorrt is not available, so load unoptimized model')
        model = mf.load_model(best_model['filepath'], compile=False)

    # Prepare the model for predicting
    nb_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    batch_size = conf.predict.getint('batch_size')
    # TODO: because of bug in tensorflow 1.14, multi GPU doesn't work (this way), 
    # so always use one
    if nb_gpu <= 1:
        model_for_predict = model
        logger.info(f"Predict using single GPU or CPU, with nb_gpu: {nb_gpu}")
    else:
        # If multiple GPU's available, create multi_gpu_model
        try:
            model_for_predict = kr.utils.multi_gpu_model(model, gpus=nb_gpu, cpu_relocation=True)
            logger.info(f"Predict using multiple GPUs: {nb_gpu}, batch size becomes: {batch_size*nb_gpu}")
            batch_size *= nb_gpu
        except ValueError:
            logger.info("Predict using single GPU or CPU")
            model_for_predict = model

    # Predict for entire dataset
    image_layer = conf.image_layers[conf.predict['image_layer']]
    predict_output_dir = Path(f"{conf.dirs['predict_image_output_basedir']}_{predict_out_subdir}")
    segment_predict.predict_dir(
            model=model_for_predict,
            input_image_dir=conf.dirs.getpath('predict_image_input_dir'),
            output_base_dir=predict_output_dir,
            border_pixels_to_ignore=conf.predict.getint('image_pixels_overlap'),
            projection_if_missing=image_layer['projection'],
            input_mask_dir=None,
            batch_size=batch_size,
            evaluate_mode=False)

# If the script is ran directly...
if __name__ == '__main__':
    raise Exception("Not implemented")
    
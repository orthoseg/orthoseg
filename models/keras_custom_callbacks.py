# -*- coding: utf-8 -*-
"""
Keras callbacks.

@author: Pieter Roggemans
"""

import os
import glob
import logging

import keras as kr

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

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
            
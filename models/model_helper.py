# -*- coding: utf-8 -*-
"""
Module with helper functions regarding (keras) models.

@author: Pieter Roggemans
"""

import os
import glob
import logging

import pandas as pd 
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

def model_base_filename(segment_subject: str,
                        train_data_version: int,
                        model_architecture: str) -> str:
    return f"{segment_subject}_{train_data_version:02}_{model_architecture}"

def model_filename(segment_subject: str,
                   train_data_version: int,
                   model_architecture: str,
                   acc_train: float,
                   acc_val: float,
                   acc_combined: float,
                   epoch: int) -> str:
    base_filename = model_base_filename(segment_subject=segment_subject,
                                        train_data_version=train_data_version,
                                        model_architecture=model_architecture)
    return model_filename2(model_base_filename=base_filename,
                           acc_train=acc_train,
                           acc_val=acc_val,
                           acc_combined=acc_combined,
                           epoch=epoch)

def model_filename2(model_base_filename: str,
                    acc_train: float,
                    acc_val: float,
                    acc_combined: float,
                    epoch: int) -> str:
    return f"{model_base_filename}_{acc_combined:.5f}_{acc_train:.5f}_{acc_val:.5f}_{epoch}.hdf5"

def get_models(model_dir: str,
               model_base_filename: str) -> pd.DataFrame:

    # glob search string    
    model_weight_filepaths = glob.glob(f"{model_dir}{os.sep}{model_base_filename}_*.hdf5")

    # Loop through all models and extract necessary info...
    model_info_list = []
    for filepath in model_weight_filepaths:
        # Prepare filepath to extract info
        filename = os.path.splitext(os.path.basename(filepath))[0]
        param_value_string = filename.replace(model_base_filename + "_", "")
        param_values = param_value_string.split("_")
        
        # Now extract fields we are interested in
        acc_combined = float(param_values[0])
        acc_train = float(param_values[1])
        acc_val = float(param_values[2])
        epoch = int(param_values[3])
        
        model_info_list.append({'filepath': filepath,
                                'filename': filename,
                                'acc_combined': acc_combined,
                                'acc_train': acc_train,
                                'acc_val': acc_val,
                                'epoch': epoch})
   
    return pd.DataFrame.from_dict(model_info_list)

def get_best_model(model_dir,
                   model_base_filename) -> dict:
    # Get list of existing models for this train dataset
    model_info_df = get_models(model_dir=model_dir,
                               model_base_filename=model_base_filename)
    
    if len(model_info_df) > 0:
        return model_info_df.loc[model_info_df['acc_combined'].values.argmax()]
    else :
        return None
    
def save_and_clean_models(model_save_dir: str,
                          model_save_base_filename: str,
                          new_model=None,
                          new_model_acc_train: float = None,
                          new_model_acc_val: float = None,
                          new_model_epoch: int = None,
                          verbose: bool = True,
                          debug: bool = False,
                          only_report: bool = False):

    # Get a list of all existing models
    model_info_df = get_models(model_dir=model_save_dir,
                               model_base_filename=model_save_base_filename)

    # If there is a new model passed as param, add it to the list
    new_model_filepath = None
    if new_model is not None:            
        # Calculate combined accuracy
        new_model_acc_combined = (new_model_acc_train+new_model_acc_val)/2
        
        # Build save filepath
        new_model_filename = model_filename2(
                model_base_filename=model_save_base_filename,
                acc_combined=new_model_acc_combined,
                acc_train=new_model_acc_train, acc_val=new_model_acc_val, 
                epoch=new_model_epoch)
        new_model_filepath = os.path.join(model_save_dir, 
                                          new_model_filename)
        
        # Append model to the retrieved models...
        model_info_df = model_info_df.append({'filepath': new_model_filepath,
                                              'filename': new_model_filename,
                                              'acc_combined': new_model_acc_combined,
                                              'acc_train': new_model_acc_train,
                                              'acc_val': new_model_acc_val,
                                              'epoch': new_model_epoch}, 
                                             ignore_index=True)
                            
    # For each model found, check if there is one with ALL parameters 
    # higher than itself. If one is found: delete model
    # Remark: the list is sorted before iterating it, this way the logging
    # on sorted from "worst to best"
    model_info_sorted_df = model_info_df.sort_values(by='acc_combined')
    for index, model_info in model_info_sorted_df.iterrows():
        better_ones_df = model_info_df[
                (model_info_df['filepath'] != model_info['filepath']) 
                & (model_info_df['acc_combined'] >= model_info['acc_combined']) 
                & (model_info_df['acc_train'] >= model_info['acc_train'])
                & (model_info_df['acc_val'] >= model_info['acc_val'])] 
        
        # If one or more better ones are found, no use in keeping it...
        if len(better_ones_df) > 0:
            if only_report is True:
                logger.debug(f"DELETE {model_info['filename']}")
            elif os.path.exists(model_info['filepath']):
                logger.debug(f"DELETE {model_info['filename']}")
                os.remove(model_info['filepath'])
                
            if debug is True:
                print(f"Better one(s) found for{model_info['filename']}:")
                for index, better_one in better_ones_df.iterrows():
                    print(f"  {better_one['filename']}")
        else:
            # No better one found, so keep it
            logger.debug(f"KEEP {model_info['filename']}")

            # If it is the new model that needs to be kept, save to disk
            if(new_model_filepath is not None 
               and only_report is not True
               and model_info['filepath'] == new_model_filepath
               and not os.path.exists(new_model_filepath)):
                new_model.save_weights(new_model_filepath)

    if verbose is True or debug is True:
        best_model = get_best_model(model_save_dir, model_save_base_filename)
        print(f"BEST MODEL: acc_combined: {best_model['acc_combined']}, acc_train: {best_model['acc_train']}, acc_val: {best_model['acc_val']}, epoch: {best_model['epoch']}")
            
class ModelCheckpointExt(kr.callbacks.Callback):
    
    def __init__(self, 
                 model_save_dir: str, 
                 model_save_base_filename: str,
                 acc_metric_train: str,
                 acc_metric_validation: str,
                 verbose: bool = True,
                 only_report: bool = False):
        self.model_save_dir = model_save_dir
        self.model_save_base_filename = model_save_base_filename
        self.acc_metric_train = acc_metric_train
        self.acc_metric_validation = acc_metric_validation
        self.verbose = verbose
        self.only_report = only_report
        
    def on_epoch_end(self, epoch, logs={}):
        logger.debug("Start in callback on_epoch_begin")
        
        save_and_clean_models(
                model_save_dir=self.model_save_dir,
                model_save_base_filename=self.model_save_base_filename,
                new_model=self.model,
                new_model_acc_train=logs.get(self.acc_metric_train),
                new_model_acc_val=logs.get(self.acc_metric_validation),
                new_model_epoch=epoch,
                verbose=self.verbose,
                only_report=self.only_report)
        
if __name__ == '__main__':
    
    #raise Exception("Not implemented")

    # General inits
    segment_subject = 'greenhouses'
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"        
    traindata_version = 17
    model_architecture = "inceptionresnetv2+linknet"
    
    project_dir = os.path.join(base_dir, segment_subject)
    
    # Init logging
    import log_helper
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)

    '''
    print(get_models(model_dir="",
                     model_basename=""))
    '''
    # Test the clean_models function (without new model)
    # Build save dir and model base filename 
    model_save_dir = os.path.join(project_dir, "models")
    model_save_base_filename = model_base_filename(
            segment_subject, traindata_version, model_architecture)
    
    # Clean the models (only report)
    save_and_clean_models(model_save_dir=model_save_dir,
                          model_save_base_filename=model_save_base_filename,
                          only_report=True)
    
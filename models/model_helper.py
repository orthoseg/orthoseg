# -*- coding: utf-8 -*-
"""
Module with helper functions regarding (keras) models.

@author: Pieter Roggemans
"""

import os
import glob
import logging

import pandas as pd 

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
    
if __name__ == '__main__':
    
    print(get_models(model_dir="",
                     model_basename=""))
    
# -*- coding: utf-8 -*-
"""
Module to test model_helper.
"""

import os

from orthoseg.helpers import log_helper
import orthoseg.model.model_helper as mh

if __name__ == '__main__':
    
    # General inits
    segment_subject = 'greenhouses'
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"        
    traindata_version = 17
    model_architecture = "inceptionresnetv2+linknet"
    
    project_dir = os.path.join(base_dir, segment_subject)
    
    # Init logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)

    '''
    print(get_models(model_dir="",
                     model_basename=""))
    '''
    # Test the clean_models function (without new model)
    # Build save dir and model base filename 
    model_save_dir = os.path.join(project_dir, "models")
    model_save_base_filename = mh.format_model_base_filename(
            segment_subject, traindata_version, model_architecture)
    
    # Clean the models (only report)
    mh.save_and_clean_models(
            model_save_dir=model_save_dir,
            model_save_base_filename=model_save_base_filename,
            monitor_metric_mode='max',
            only_report=True)

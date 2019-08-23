# -*- coding: utf-8 -*-
"""
Module with functions for post-processing prediction masks towards polygons.
"""

import os

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, '.')

import orthoseg.predict_postprocess as postp
import orthoseg.helpers.config as conf
import orthoseg.helpers.log as log_helper

def test_postprocess_vectors():
    
    # Read the configuration
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    segment_config_filepaths=[os.path.join(scriptdir, 'general_test.ini'), 
                              os.path.join(scriptdir, 'sealedsurfaces_test.ini'), 
                              os.path.join(scriptdir, 'local_overrule_test.ini')]
    conf.read_config(segment_config_filepaths)

    # Init logger
    global logger
    logger = log_helper.main_log_init(conf.dirs['log_dir'], __name__)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Input and output dir
    input_dir = (conf.dirs['predict_image_output_basedir'] 
                 + "_sealedsurfaces_08_inceptionresnetv2+linknet_0.94311_0.92964_0")
    output_dir = os.path.join(conf.dirs['output_vector_dir'], 
                              f"{conf.general['segment_subject']}_test")
    output_filepath = os.path.join(output_dir, 
                                   f"{conf.general['segment_subject']}_test.gpkg")
    
    postp.postprocess_vectors(input_dir=input_dir,
                              output_filepath=output_filepath,
                              evaluate_mode=True,
                              force=True)

#-------------------------------------------------------------
# If the script is ran directly...
#-------------------------------------------------------------

if __name__ == '__main__':
    test_postprocess_vectors()

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
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_filepaths=[os.path.join(config_dir, 'general_test.ini'), 
                              os.path.join(config_dir, 'sealedsurfaces_test.ini'), 
                              os.path.join(config_dir, 'local_overrule_test.ini')]
    layer_config_filepath = os.path.join(config_dir, 'image_layers.ini')
    conf.read_config(config_filepaths, layer_config_filepath)

    # Init logger
    global logger
    logger = log_helper.main_log_init(conf.dirs['log_dir'], __name__)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Input and output dir
    input_dir = (conf.dirs['predict_image_output_basedir'] 
                 + "_sealedsurfaces_16_inceptionresnetv2+unet_348")
    output_vector_name = f"{conf.general['segment_subject']}_16_{conf.predict['image_layer']}"
    output_dir = os.path.join(
            conf.dirs['output_vector_dir'], output_vector_name)
    output_filepath = os.path.join(
            output_dir, f"{output_vector_name}.gpkg")
    
    postp.postprocess_vectors(input_dir=input_dir,
                              output_filepath=output_filepath,
                              evaluate_mode=True,
                              force=True)

#-------------------------------------------------------------
# If the script is ran directly...
#-------------------------------------------------------------

if __name__ == '__main__':
    test_postprocess_vectors()

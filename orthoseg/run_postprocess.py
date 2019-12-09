# -*- coding: utf-8 -*-
"""
Module with functions for post-processing prediction masks towards polygons.
"""

import glob
import os

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, '.')

import orthoseg.postprocess_predictions as postp
from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper

def postprocess_predictions():
    
    ##### Init #####
    # Input dir = the "most recent" prediction result dir for this subject 
    prediction_basedir = f"{conf.dirs['predict_image_output_basedir']}_{conf.general['segment_subject']}_"
    prediction_dirs = sorted(glob.glob(f"{prediction_basedir}*{os.sep}"), reverse=True)
    input_dir = prediction_dirs[0]
	
    # Format output dir, partly based on input dir
    train_version = input_dir.replace(prediction_basedir, "").split('_')[0]
    output_vector_name = f"{conf.general['segment_subject']}_{train_version:02}_{conf.predict['image_layer']}"
    output_dir = os.path.join(
            conf.dirs['output_vector_dir'], output_vector_name)
    output_filepath = os.path.join(
            output_dir, f"{output_vector_name}.gpkg")
    
    ##### Go! #####
    border_pixels_to_ignore = conf.predict.getint('image_pixels_overlap')
    postp.postprocess_predictions(
            input_dir=input_dir,
            output_filepath=output_filepath,
            input_ext='.tif',
            border_pixels_to_ignore=border_pixels_to_ignore,
            evaluate_mode=False,
            force=True)

#-------------------------------------------------------------
# If the script is ran directly...
#-------------------------------------------------------------

if __name__ == '__main__':
    raise Exception("Not implemented")

# -*- coding: utf-8 -*-
"""
Module with functions for post-processing prediction masks towards polygons.
"""

import argparse
from pathlib import Path
import shlex
import sys

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper
from orthoseg.lib import postprocess_predictions as postp

def postprocess_argstr(argstr):
    args = shlex.split(argstr)
    postprocess_args(args)

def postprocess_args(args):

    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument("--config_dir", type=str, required=True,
            help="The config dir to use")
    required.add_argument("--config_filename", type=str, required=True,
            help="The config file to use")
    
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    # Add back help 
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')

    # Interprete arguments
    args = parser.parse_args(args)

    ##### Run! #####
    postprocess(config_dir=Path(args.config_dir),
            config_filename=args.config_filename)

def postprocess(
        config_dir: Path,
        config_filename: str):

    ##### Init #####   
    # Load config
    config_filepaths = conf.get_needed_config_files(
            config_dir=config_dir, 
            config_filename=config_filename)
    layer_config_filepath = config_dir / 'image_layers.ini'
    conf.read_config(config_filepaths, layer_config_filepath)
    
    # Main initialisation of the logging
    global logger
    logger = log_helper.main_log_init(conf.dirs.getpath('log_training_dir'), __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Input dir = the "most recent" prediction result dir for this subject 
    prediction_basedir = Path(f"{conf.dirs['predict_image_output_basedir']}_{conf.general['segment_subject']}_")
    prediction_dirs = sorted(prediction_basedir.parent.glob(f"{prediction_basedir.name}*/"), reverse=True)
    input_dir = prediction_dirs[0]
	
    # Format output dir, partly based on input dir
    # Remove first 2 field from the input dir to get the model info
    input_dir_splitted = input_dir.name.split('_')
    model_info = []
    for i, input_dir_field in enumerate(input_dir_splitted):
        if i >= 2:
            model_info.append(input_dir_field)
    
    output_dir = conf.dirs.getpath('output_vector_dir') / conf.predict['image_layer']
    output_vector_name = f"{'_'.join(model_info)}_{conf.predict['image_layer']}"
    output_filepath = output_dir / f"{output_vector_name}.gpkg"
    
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
    postprocess_args(sys.argv[1:])
    
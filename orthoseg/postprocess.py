# -*- coding: utf-8 -*-
"""
Module with functions for post-processing prediction masks towards polygons.
"""

import argparse
from pathlib import Path
import shlex
import sys

from geofileops import geofileops

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper
from orthoseg.lib import postprocess_predictions as postp
import orthoseg.model.model_helper as mh

# Define global variables
logger = None

def postprocess_argstr(argstr):
    args = shlex.split(argstr)
    postprocess_args(args)

def postprocess_args(args):

    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument("--configfile", type=str, required=True,
            help="The config file to use")
    
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    # Add back help 
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')

    # Interprete arguments
    args = parser.parse_args(args)

    ##### Run! #####
    postprocess(projectconfig_path=Path(args.configfile))

def postprocess(
        projectconfig_path: Path,
        imagelayerconfig_path: Path = None):
    """
    Postprocess.

    Args:
        projectconfig_path (Path): Path to the projects config file.
        imagelayerconfig_path (Path, optional): Path to the imagelayer config file. If not specified, 
            the path specified in files.image_layers_config_filepath in the project config will be used. 
            Defaults to None.
    """

    ##### Init #####   
    # Load config
    config_filepaths = conf.search_projectconfig_files(projectconfig_path=projectconfig_path)
    conf.read_project_config(config_filepaths, imagelayerconfig_path)
    
    # Main initialisation of the logging
    global logger
    logger = log_helper.main_log_init(conf.dirs.getpath('log_dir'), __name__)      
    logger.debug(f"Config used: \n{conf.pformat_config()}")

    # Create base filename of model to use
    # TODO: is force data version the most logical, or rather implement 
    #       force weights file or ?
    traindata_id = None
    force_model_traindata_id = conf.train.getint('force_model_traindata_id')
    if force_model_traindata_id is not None and force_model_traindata_id > -1:
        traindata_id = force_model_traindata_id 
    
    # Get the best model that already exists for this train dataset
    trainparams_id = conf.train.getint('trainparams_id')
    best_model = mh.get_best_model(
            model_dir=conf.dirs.getpath('model_dir'), 
            segment_subject=conf.general['segment_subject'],
            traindata_id=traindata_id,
            trainparams_id=trainparams_id)
    
    # Input file  the "most recent" prediction result dir for this subject 
    output_vector_dir = conf.dirs.getpath('output_vector_dir') / conf.predict['image_layer']
    output_vector_name = f"{best_model['basefilename']}_{best_model['epoch']}_{conf.predict['image_layer']}"
    output_vector_path = output_vector_dir / f"{output_vector_name}.gpkg"
    	    
    # Prepare some parameters for the postprocessing
    dissolve = conf.postprocess.getboolean('dissolve')
    dissolve_tiles_path = conf.postprocess.getpath('dissolve_tiles_path')
    simplify_algorithm = conf.postprocess.get('simplify_algorithm')
    if simplify_algorithm is not None:
        simplify_algorithm = geofileops.SimplifyAlgorithm[simplify_algorithm]
    simplify_tolerance = conf.postprocess.geteval('simplify_tolerance')
    simplify_lookahead = conf.postprocess.get('simplify_lookahead')
    if simplify_lookahead is not None:
        simplify_lookahead = int(simplify_lookahead)

    ##### Go! #####
    postp.postprocess_predictions(
            input_path=output_vector_path,
            output_path=output_vector_path,
            dissolve=dissolve,
            dissolve_tiles_path=dissolve_tiles_path,
            simplify_algorithm=simplify_algorithm,
            simplify_tolerance=simplify_tolerance,
            simplify_lookahead=simplify_lookahead,
            force=False)

#-------------------------------------------------------------
# If the script is ran directly...
#-------------------------------------------------------------

if __name__ == '__main__':
    postprocess_args(sys.argv[1:])
    
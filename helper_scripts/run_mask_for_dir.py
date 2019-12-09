# -*- coding: utf-8 -*-
"""
Script to load images from a WMS server.
"""

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, '.')

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper

import orthoseg.prepare_traindatasets as prep

if __name__ == '__main__':
    ##### Init #####
    # Main initialisation of the logging
    '''
    layer_config_filepath = os.path.join(config_dir, 'image_layers.ini')
    conf.read_config(config_filepaths, layer_config_filepath)
    logger = log_helper.main_log_init(conf.dirs['log_dir'], __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")
    '''
    prep.create_masks_for_images(
            input_vector_label_filepath=r"X:\Monitoring\OrthoSeg\topobuildings\input_labels\topobuildings_trainlabels_4326.shp",
            input_image_dir=r"X:\Monitoring\OrthoSeg\topobuildings\training\input_images_train",
            output_basedir=r"X:\Monitoring\OrthoSeg\topobuildings\training\train")

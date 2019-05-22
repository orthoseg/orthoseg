# -*- coding: utf-8 -*-
"""
Module to run a prediction.

@author: Pieter Roggemans
"""

import os

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, '.')

import orthoseg.predict as pred

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():

    scriptdir = os.path.dirname(os.path.abspath(__file__))
    pred.run_prediction(segment_config_filepaths=[os.path.join(scriptdir, 'general.ini'), 
                                                  os.path.join(scriptdir, 'fruit.ini'),
                                                  os.path.join(scriptdir, 'local_overrule.ini')], 
                        force_model_traindata_version=None)
    
if __name__ == '__main__':
    main()

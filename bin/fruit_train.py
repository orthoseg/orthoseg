# -*- coding: utf-8 -*-
"""
Script with commands to run a training or prediction session.
"""

import os

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
sys.path.insert(0, '.')

import orthoseg.train as train
import orthoseg.predict as pred

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():
        
    # Start the training session
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    train.run_training_session(segment_config_filepaths=[os.path.join(scriptdir, 'general.ini'), 
                                                         os.path.join(scriptdir, 'fruit.ini'),
                                                         os.path.join(scriptdir, 'local_overrule.ini')])
    pred.run_prediction(segment_config_filepaths=[os.path.join(scriptdir, 'general.ini'), 
                                                  os.path.join(scriptdir, 'fruit.ini'),
                                                  os.path.join(scriptdir, 'local_overrule.ini')])

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Script with commands to run a training or prediction session.

@author: Pieter Roggemans
"""

import training_helper as th

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():
        
    # Start the training session
    th.run_training_session(segment_config_filepath=['general.ini', 
                                                     'sealedsurfaces.ini'],
                            force_traindata_version=None,
                            resume_train=False)

if __name__ == '__main__':
    main()

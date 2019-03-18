# -*- coding: utf-8 -*-
"""
Script to run a training session for horsetrack segmentation.

@author: Pieter Roggemans
"""

import training_helper as th

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():
        
    # Start the training session
    th.run_training_session(segment_config_filepaths=['general.ini', 
                                                      'horsetracks.ini',
                                                      'local_overrule.ini'],
                            force_traindata_version=None,
                            resume_train=False)
    
if __name__ == '__main__':
    main()

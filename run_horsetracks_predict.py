# -*- coding: utf-8 -*-
"""
Module to run the prediction for horsetracks.

@author: Pieter Roggemans
"""

import predict_helper as ph

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():

    ph.run_prediction(segment_config_filepaths=['general.ini', 
                                                'horsetracks.ini',
                                                'local_overrule.ini'], 
                      force_model_traindata_version=None)
    
if __name__ == '__main__':
    main()

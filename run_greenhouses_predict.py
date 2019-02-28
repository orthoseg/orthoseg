# -*- coding: utf-8 -*-
"""
Module to run the prediction for greenhouses.

@author: Pieter Roggemans
"""

import predict_helper as ph

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():

    ph.run_prediction(segment_config_filepath=['general.ini', 
                                               'greenhouses.ini'], 
                      force_model_traindata_version=None)
    
if __name__ == '__main__':
    main()

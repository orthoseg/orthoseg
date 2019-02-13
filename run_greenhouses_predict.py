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

    # TODO: autodetect highest data version!!!        
    ph.run_prediction(segment_config_filepath='greenhouses.ini', 
                      force_traindata_version=20)
    
if __name__ == '__main__':
    main()

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
    
    # General initialisations for the segmentation job
    segment_subject = "horsetracks"
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation"
    
    # WMS server we can use to get the image data
    WMS_SERVER_URL = 'http://geoservices.informatievlaanderen.be/raadpleegdiensten/ofw/wms?'
    
    # Start the training session    
    th.run_training_session(segment_subject=segment_subject,
                            base_dir=base_dir,
                            wms_server_url=WMS_SERVER_URL,
                            model_encoder='inceptionresnetv2',
                            model_decoder='linknet',
                            batch_size_train=8,
                            batch_size_pred=20,
                            force_traindata_version=None,
                            preload_existing_model=False)
    
if __name__ == '__main__':
    main()

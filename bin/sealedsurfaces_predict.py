# -*- coding: utf-8 -*-
"""
Module to run a prediction.
"""

import os   
import sys

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
#sys.path.insert(0, '.')
[sys.path.append(i) for i in ['.', '..']]

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"

import orthoseg.predict as pred

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():
    ##### List the needed config files #####
    scriptdir = os.path.dirname(os.path.abspath(__file__))

    # General settings first
    config_filepaths = [os.path.join(scriptdir, 'general.ini')]

    # Then specific settings depending on the OS
    if os.name == 'posix':
        config_filepaths.append(os.path.join(scriptdir, 'general_posix.ini'))
    elif os.name == 'nt':
        None
    else: 
        raise Exception(f"Unsupported os.name: {os.name}")

    # Finally specific setting for subject or local overule
    config_filepaths.append(os.path.join(scriptdir, 'sealedsurfaces.ini'))
    config_filepaths.append(os.path.join(scriptdir, 'local_overrule.ini'))

    ##### Run! #####
    pred.run_prediction(segment_config_filepaths=config_filepaths)
    
if __name__ == '__main__':
    main()

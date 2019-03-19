# -*- coding: utf-8 -*-
"""
Module with helper functions to work with geo file.

@author: Pieter Roggemans
"""

import os
import filecmp
import shutil
import logging

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def cmp(filepath1, filepath2):
    
    # For a shapefile, multiple files need to be compared
    filepath1_noext, file1ext = os.path.splitext(filepath1)
    if file1ext.lower() == '.shp':
        filepath2_noext, file2ext = os.path.splitext(filepath2)
        shapefile_extentions = [".shp", ".dbf", ".shx"]
        for ext in shapefile_extentions:
            if not filecmp.cmp(filepath1_noext + ext, filepath2_noext + ext):
                logger.info(f"File {filepath1_noext}{ext} is differnet from {filepath2_noext}{ext}")
                return False
        return True
    else:
        return filecmp.cmp(filepath1, filepath2)
    
def copy(filepath_src, dest):

    # For a shapefile, multiple files need to be copied
    filepath_src_noext, fileext_src = os.path.splitext(filepath_src)
    if fileext_src.lower() == '.shp':
        shapefile_extentions = [".shp", ".dbf", ".shx", ".prj"]

        # If dest is a dir, just use copy. Otherwise concat dest filepaths
        if os.path.isdir(dest):
            for ext in shapefile_extentions:
                shutil.copy(filepath_src_noext + ext, dest)
        else:
            filepath_dest_noext, fileext_dest = os.path.splitext(dest)
            for ext in shapefile_extentions:
                shutil.copy(filepath_src_noext + ext, filepath_dest_noext + ext)                
    else:
        return shutil.copy(filepath_src, dest)
    

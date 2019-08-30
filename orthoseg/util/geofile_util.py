# -*- coding: utf-8 -*-
"""
Module with generic usable utility functions to work with geo files.
"""

import os
import filecmp
import shutil
import logging

import fiona
import geopandas as gpd

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def read_file(filepath: str,
              layer: str = None,
              columns: [] = None) -> gpd.GeoDataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    # If no layer name specified, use the filename (without extension)
    filepath_noext, ext = os.path.splitext(filepath)
    if layer is None:
        _, layer = os.path.split(filepath_noext)

    # Depending on the extension... different implementations
    ext_lower = ext.lower()
    if ext_lower == '.shp':
        return gpd.read_file(filepath)
    elif ext_lower == '.geojson':
        return gpd.read_file(filepath)        
    elif ext_lower == '.gpkg':
        return gpd.read_file(filepath, layer=layer)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")

def to_file(gdf: gpd.GeoDataFrame,
            filepath: str,
            layer: str = None,
            index: bool = True):
    """
    Reads a pandas dataframe to file. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    # If no layer name specified, use the filename (without extension)
    filepath_noext, ext = os.path.splitext(filepath)
    if layer is None:
        _, layer = os.path.split(filepath_noext)

    # Depending on the extension... different implementations
    ext_lower = ext.lower()
    if ext_lower == '.shp':
        gdf.to_file(filepath)
    elif ext_lower == '.geojson':
        gdf.to_file(filepath, driver='GeoJSON')
    elif ext_lower == '.gpkg':
        gdf.to_file(filepath, layer=layer, driver='GPKG')
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")
        
def get_crs(filepath):
    with fiona.open(filepath, 'r') as geofile:
        return geofile.crs

def cmp(filepath1, filepath2):
    
    # For a shapefile, multiple files need to be compared
    filepath1_noext, file1ext = os.path.splitext(filepath1)
    if file1ext.lower() == '.shp':
        filepath2_noext, _ = os.path.splitext(filepath2)
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
            filepath_dest_noext, _ = os.path.splitext(dest)
            for ext in shapefile_extentions:
                shutil.copy(filepath_src_noext + ext, filepath_dest_noext + ext)                
    else:
        return shutil.copy(filepath_src, dest)

def get_driver(filepath) -> str:
    """
    Get the driver to use for the file extension of this filepath.
    """
    _, file_ext = os.path.splitext(filepath)

    return get_driver_for_ext(file_ext)

def get_driver_for_ext(file_ext) -> str:
    """
    Get the driver to use for this file extension.
    """
    file_ext_lower = file_ext.lower()
    if file_ext_lower == '.shp':
        return 'ESRI Shapefile'
    elif file_ext_lower == '.geojson':
        return 'GeoJSON'
    elif file_ext_lower == '.gpkg':
        return 'GPKG'
    else:
        raise Exception(f"Not implemented for extension {file_ext_lower}")        

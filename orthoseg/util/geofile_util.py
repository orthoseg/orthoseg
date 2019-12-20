# -*- coding: utf-8 -*-
"""
Module with generic usable utility functions to work with geo files.
"""

import filecmp
import logging
import os
from pathlib import Path
import shutil
from typing import List

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

def read_file(filepath: Path,
              layer: str = None,
              columns: List[str] = None) -> gpd.GeoDataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    # If no layer name specified, use the filename (without extension)
    if layer is None:
        layer = filepath.stem

    # Depending on the extension... different implementations
    ext_lower = filepath.suffix.lower()
    if ext_lower == '.shp':
        return gpd.read_file(str(filepath))
    elif ext_lower == '.geojson':
        return gpd.read_file(str(filepath))
    elif ext_lower == '.gpkg':
        return gpd.read_file(str(filepath), layer=layer)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")

def to_file(gdf: gpd.GeoDataFrame,
            filepath: Path,
            layer: str = None,
            index: bool = True):
    """
    Reads a pandas dataframe to file. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    # If no layer name specified, use the filename (without extension)
    if layer is None:
        layer = filepath.stem
    # If the dataframe is empty, log warning and return
    if len(gdf) <= 0:
        logger.warn(f"Cannot write an empty dataframe to {filepath}.{layer}")
        return

    # Depending on the extension... different implementations
    ext_lower = filepath.suffix.lower()
    if ext_lower == '.shp':
        gdf.to_file(str(filepath))
    elif ext_lower == '.geojson':
        gdf.to_file(str(filepath), driver='GeoJSON')
    elif ext_lower == '.gpkg':
        gdf.to_file(str(filepath), layer=layer, driver='GPKG')
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")
        
def get_crs(filepath: Path):
    with fiona.open(str(filepath), 'r') as geofile:
        return geofile.crs

def cmp(filepath1: Path, filepath2: Path):
    
    # For a shapefile, multiple files need to be compared
    if filepath1.suffix.lower() == '.shp':
        filepath2_noext, _ = os.path.splitext(filepath2)
        shapefile_extentions = [".shp", ".dbf", ".shx"]
        filepath1_noext = filepath1.parent / filepath1.stem
        filepath2_noext = filepath2.parent / filepath2.stem
        for ext in shapefile_extentions:
            if not filecmp.cmp(str(filepath1_noext) + ext, str(filepath2_noext) + ext):
                logger.info(f"File {filepath1_noext}{ext} is differnet from {filepath2_noext}{ext}")
                return False
        return True
    else:
        return filecmp.cmp(str(filepath1), str(filepath2))
    
def copy(filepath_src: Path, dest: Path):

    # For a shapefile, multiple files need to be copied
    if filepath_src.suffix.lower() == '.shp':
        shapefile_extentions = [".shp", ".dbf", ".shx", ".prj"]

        # If dest is a dir, just use copy. Otherwise concat dest filepaths
        filepath_src_noext = filepath_src.parent / filepath_src.stem
        if dest.is_dir():
            for ext in shapefile_extentions:
                shutil.copy(str(filepath_src_noext) + ext, dest)
        else:
            filepath_dest_noext = dest.parent / dest.stem
            for ext in shapefile_extentions:
                shutil.copy(str(filepath_src_noext) + ext, str(filepath_dest_noext) + ext)                
    else:
        return shutil.copy(filepath_src, dest)

def get_driver(filepath: Path) -> str:
    """
    Get the driver to use for the file extension of this filepath.
    """
    return get_driver_for_ext(filepath.suffix)

def get_driver_for_ext(file_ext: str) -> str:
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

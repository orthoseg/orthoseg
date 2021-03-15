# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.train.
"""

from pathlib import Path
import sys

import owslib
import owslib.wms
import owslib.util

# Add path so the local orthoseg packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.util import ows_util
from tests import test_helper

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_getmap_to_file(tmpdir):
    # Init some stuff 
    tmpdir = Path(tmpdir)
    projection = 'epsg:31370'
    pixelsize_x = 0.25
    pixelsize_y = pixelsize_x
    image_pixel_width = 512
    image_pixel_height = image_pixel_width
    xmin = 160000
    ymin = 170000
    bbox = (xmin, ymin, xmin + image_pixel_width*pixelsize_x, ymin + image_pixel_height*pixelsize_y)
    
    ### Test getting a standard 3 band image from a WMS layer ### 
    wms_server_url = 'https://geoservices.informatievlaanderen.be/raadpleegdiensten/omw/wms?'
    wms_version = '1.3.0'
    auth = owslib.util.Authentication()
    wms_service = owslib.wms.WebMapService(wms_server_url, version=wms_version, auth=auth)

    # Two different layers, just use the 1st band from each of them
    layersources = []
    layersources.append(
            ows_util.LayerSource(
                    wms_service = wms_service,
                    layernames = ['OMWRGB20VL'],
                    layerstyles = ['default']))
    
    # Now get the image...
    image_filepath = ows_util.getmap_to_file(
            layersources=layersources,
            output_dir=tmpdir,
            crs=projection,
            bbox=bbox,
            size=(image_pixel_width, image_pixel_height),
            image_format=ows_util.FORMAT_PNG,
            #image_format_save=ows_util.FORMAT_TIFF,
            image_pixels_ignore_border=0,
            transparent=False,
            layername_in_filename=True)

    assert image_filepath.exists() is True

    ### Test combining bands of 3 different WMS layers ### 
    wms_version = '1.3.0'
    auth = owslib.util.Authentication()
    # WMS server for skyview and hillshade
    wms_server_url_dhm = 'https://geoservices.informatievlaanderen.be/raadpleegdiensten/DHMV/wms?'
    wms_service_dhm = owslib.wms.WebMapService(wms_server_url_dhm, version=wms_version, auth=auth)
    # WMS server for rgb images taken together with dhm2
    wms_server_url_dhm_ortho = 'https://geoservices.informatievlaanderen.be/raadpleegdiensten/ogw/wms?'
    wms_service_dhm_ortho = owslib.wms.WebMapService(wms_server_url_dhm_ortho, version=wms_version, auth=auth)

    # Three different layers, just use one band from each of them
    layersources = []
    layersources.append(
            ows_util.LayerSource(
                    wms_service=wms_service_dhm,
                    layernames=['DHMV_II_SVF_25cm'],
                    bands=[1]))
    layersources.append(
            ows_util.LayerSource(
                    wms_service=wms_service_dhm,
                    layernames=['DHMV_II_HILL_25cm'],
                    bands=[1]))
    layersources.append(
            ows_util.LayerSource(
                    wms_service=wms_service_dhm_ortho,
                    layernames=['OGWRGB13_15VL'],
                    bands=[-1]))

    # Now get the image...
    image_filepath = ows_util.getmap_to_file(
            layersources=layersources,
            output_dir=tmpdir,
            crs=projection,
            bbox=bbox,
            size=(image_pixel_width, image_pixel_height),
            image_format=ows_util.FORMAT_PNG,
            #image_format_save=ows_util.FORMAT_TIFF,
            image_pixels_ignore_border=0,
            transparent=False,
            layername_in_filename=True)

    assert image_filepath.exists() is True

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)
    
    test_getmap_to_file(tmpdir)
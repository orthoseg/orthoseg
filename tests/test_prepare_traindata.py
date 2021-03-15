# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.lib.postprocess_predictions.
"""
import os
from pathlib import Path
import sys

from shapely import geometry as sh_geom

# Make hdf5 version warning non-blocking
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

import geopandas as gpd
from geofileops import geofile

# Add path so the local orthoseg packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.lib import prepare_traindatasets as prep_traindata
from orthoseg.helpers import config_helper
from tests import test_helper

def test_prepare_traindata(tmpdir):
    # Prepare test data
    tmpdir=Path(tmpdir)
    
    locations_path = tmpdir / 'locations.gpkg'
    locations_data = {
            'geometry': [sh_geom.Polygon([(150000, 170000), (150128, 170000), (150128, 170128), (150000, 170128), (150000, 170000)]),
                         sh_geom.Polygon([(150000, 180000), (150128, 180000), (150128, 180128), (150000, 180128), (150000, 180000)]),
                         sh_geom.Polygon([(150000, 190000), (150128, 190000), (150128, 190128), (150000, 190128), (150000, 190000)])],
            'traindata_type': ['train', 'train', 'validation']
        }
    locations_gdf = gpd.GeoDataFrame(locations_data, crs='epsg:31370')
    geofile.to_file(locations_gdf, locations_path)
    polygons_path = tmpdir / 'polygons.gpkg'
    polygons_data = {
            'geometry': [sh_geom.Polygon([(150030, 170030), (15060, 170030), (150060, 170060), (150030, 170060), (150030, 170030)]),
                         sh_geom.Polygon([(150030, 180030), (150060, 180030), (150060, 180060), (150030, 180060), (150030, 180030)])],
            'label_name': ['testlabel', 'testlabel']
        }

    polygons_gdf = gpd.GeoDataFrame(polygons_data, crs='epsg:31370')
    geofile.to_file(polygons_gdf, polygons_path)
    label_infos = []
    label_infos.append(prep_traindata.LabelInfo(
            locations_path=locations_path,
            polygons_path=polygons_path,
            image_layer='BEFL-2019'))
    classes = { "background": {
                    "labelnames": ["ignore_for_train", "background"],
                    "weight": 1,
                    "burn_value": 0
                },
                "test": {
                    "labelnames": ["testlabel"],
                    "weight": 1,
                    "burn_value": 1
                }
            }
    image_layers_config_path = test_helper.get_testprojects_dir() / 'imagelayers.ini'
    image_layers = config_helper.read_layer_config(image_layers_config_path) 
    
    # Test with the default data... 
    training_dir = tmpdir / 'training_dir'
    training_imagedata_dir = tmpdir / 'training_imagedata_dir'
    training_dir, traindata_id = prep_traindata.prepare_traindatasets(
            label_infos=label_infos,
            classes=classes,
            image_layers=image_layers,
            training_dir=training_dir,
            training_imagedata_dir=training_imagedata_dir,
            labelname_column='label_name',
            image_pixel_x_size=0.25,
            image_pixel_y_size=0.25,
            image_pixel_width=512,
            image_pixel_height=512)

    assert training_dir.exists() is True

    # Test with None and incorrect label names
    polygons_data = {
            'geometry': [sh_geom.Polygon([(150030, 170030), (15060, 170030), (150060, 170060), (150030, 170060), (150030, 170030)]),
                         sh_geom.Polygon([(150030, 180030), (150060, 180030), (150060, 180060), (150030, 180060), (150030, 180030)])],
            'label_name': ['testlabelwrong', None]
        }
    polygons_gdf = gpd.GeoDataFrame(polygons_data, crs='epsg:31370')
    geofile.to_file(polygons_gdf, polygons_path)

    training_dir = tmpdir / 'training_dir'
    training_imagedata_dir = tmpdir / 'training_imagedata_dir'
    try:
        training_dir, traindata_id = prep_traindata.prepare_traindatasets(
                label_infos=label_infos,
                classes=classes,
                image_layers=image_layers,
                training_dir=training_dir,
                training_imagedata_dir=training_imagedata_dir,
                labelname_column='label_name',
                image_pixel_x_size=0.25,
                image_pixel_y_size=0.25,
                image_pixel_width=512,
                image_pixel_height=512)
        run_ok = True
    except Exception as ex:
        print(str(ex))
        run_ok = False
    # There should have been an error!
    assert run_ok is False

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)
    
    # Run!
    test_prepare_traindata(tmpdir)

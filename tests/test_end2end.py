# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.train.
"""

from pathlib import Path
import shutil
import sys

from geofileops import geofile

# Add path so the local orthoseg packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg import orthoseg
from orthoseg.helpers import config_helper as conf
import orthoseg.model.model_helper as mh

# ----------------------------------------------------
# Tests
# ----------------------------------------------------

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def get_test_projects_dir() -> Path:
    return Path(__file__).resolve().parent / 'test_projects'

def test_load_images():
    # Load project config to init some vars.
    tasks_path = get_test_projects_dir() / 'tasks_load_images.csv'
    project_config_path = get_test_projects_dir() / 'footballfields' / 'footballfields_BEFL-2019.ini'
    config_filepaths = conf.search_projectconfig_files(project_config_path)
    conf.read_project_config(config_filepaths)
    image_cache_dir = conf.dirs.getpath('predict_image_input_dir')

    # Clean result if it isn't empty yet
    if image_cache_dir.exists():
        shutil.rmtree(image_cache_dir)
        # Make sure is is deleted now!
        assert image_cache_dir.exists() is False

    # Run task to load images
    orthoseg.run_tasks(tasks_path=tasks_path)

    # Check if the right number of files was loaded
    assert image_cache_dir.exists() is True
    files = list(image_cache_dir.glob("**/*.*"))
    assert len(files) == 16

def test_train():
    # Load project config to init some vars.
    tasks_path = get_test_projects_dir() / 'tasks_load_images.csv'
    project_config_path = get_test_projects_dir() / 'footballfields' / 'footballfields_BEFL-2019.ini'
    config_filepaths = conf.search_projectconfig_files(project_config_path)
    conf.read_project_config(config_filepaths)

    # Init + cleanup result dirs
    traindata_id_result = 2
    training_dir = conf.dirs.getpath('training_dir')
    training_id_dir = training_dir / f"{traindata_id_result:02d}"
    if training_id_dir.exists():
        shutil.rmtree(training_id_dir)
    training_imagedata_dir = conf.dirs.getpath('training_imagedata_dir')
    training_imagedata_id_dir = training_imagedata_dir / f"{traindata_id_result:02d}"
    if training_imagedata_id_dir.exists():
        shutil.rmtree(training_imagedata_id_dir)

    # Run task to train
    # The label files are newer than the ones used to train the current model, 
    # so a new model will be trained. 
    tasks_path = get_test_projects_dir() / 'tasks_train.csv'
    orthoseg.run_tasks(tasks_path=tasks_path)

    # Check if the training (image) data was created
    assert training_id_dir.exists() is True
    assert training_imagedata_id_dir.exists() is True
    
    # Check if the new model was created
    best_model = mh.get_best_model(
            model_dir=conf.dirs.getpath('model_dir'), 
            segment_subject=conf.general['segment_subject'])
    
    assert best_model['traindata_id'] == traindata_id_result
    assert best_model['epoch'] == 0

def test_predict():
    # Load project config to init some vars.
    tasks_path = get_test_projects_dir() / 'tasks_load_images.csv'
    project_config_path = get_test_projects_dir() / 'footballfields' / 'footballfields_BEFL-2019.ini'
    config_filepaths = conf.search_projectconfig_files(project_config_path)
    conf.read_project_config(config_filepaths)

    # Cleanup result if it isn't empty yet
    predict_image_output_basedir = conf.dirs.getpath('predict_image_output_basedir')
    predict_image_output_dir = predict_image_output_basedir.parent / f"{predict_image_output_basedir.name}_footballfields_02_0"
    if predict_image_output_dir.exists():
        shutil.rmtree(predict_image_output_dir)
        # Make sure it is deleted now!
        assert predict_image_output_dir.exists() is False
    result_vector_dir = conf.dirs.getpath('output_vector_dir')
    if result_vector_dir.exists():
        shutil.rmtree(result_vector_dir)
        # Make sure is is deleted now!
        assert result_vector_dir.exists() is False

    # Run task to predict
    tasks_path = get_test_projects_dir() / 'tasks_predict.csv'
    orthoseg.run_tasks(tasks_path=tasks_path)

    # Check results
    result_vector_path = result_vector_dir / "footballfields_01_242_BEFL-2019.gpkg"
    assert result_vector_path.exists() is True
    result_gdf = geofile.read_file(result_vector_path)
    assert len(result_gdf) == 7

def test_postprocess():
    # Load project config, to read destination dir's of load_images.
    tasks_path = get_test_projects_dir() / 'tasks_load_images.csv'
    project_config_path = get_test_projects_dir() / 'footballfields' / 'footballfields_BEFL-2019.ini'
    config_filepaths = conf.search_projectconfig_files(project_config_path)
    conf.read_project_config(config_filepaths)
    result_vector_dir = conf.dirs.getpath('output_vector_dir')

    # Cleanup result if it isn't empty yet
    result_diss_path = result_vector_dir / 'footballfields_01_242_BEFL-2019_dissolve.gpkg'
    result_simpl_path = result_vector_dir / 'footballfields_01_242_BEFL-2019_dissolve_simpl.gpkg'
    if result_diss_path.exists():
        geofile.remove(result_diss_path)
    if result_simpl_path.exists():
        geofile.remove(result_simpl_path)

    # Run task to postprocess
    tasks_path = get_test_projects_dir() / 'tasks_postprocess.csv'
    orthoseg.run_tasks(tasks_path=tasks_path)

    # Check results
    assert result_diss_path.exists() is True
    result_gdf = geofile.read_file(result_diss_path)
    assert len(result_gdf) == 6

    assert result_simpl_path.exists() is True
    result_gdf = geofile.read_file(result_simpl_path)
    assert len(result_gdf) == 6

if __name__ == '__main__':
    #test_load_images()
    #test_train()
    test_predict()
    test_postprocess()
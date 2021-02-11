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

# ----------------------------------------------------
# Tests
# ----------------------------------------------------

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def get_test_projects_dir() -> Path:
    return Path(__file__).resolve().parent / 'test_projects'

def test_load_images():
    # Cleanup result if it isn't empty yet
    image_cache_dir = get_test_projects_dir() / '_image_cache' / 'BEFL-2019'
    if image_cache_dir.exists():
        shutil.rmtree(image_cache_dir, ignore_errors=True)
        # Make sure is is deleted now!
        if image_cache_dir.exists():
            assert True is False

    # Run task to load images
    tasks_path = get_test_projects_dir() / 'tasks_load_images.csv'
    orthoseg.run_tasks(tasks_path=tasks_path)

    # Check if the right number of files was loaded
    files = list(image_cache_dir.glob("**/*.*"))
    assert len(files) == 16

def test_train():
    # Cleanup/rename current model to start training ???
    # Run task to train
    tasks_path = get_test_projects_dir() / 'tasks_train.csv'
    orthoseg.run_tasks(tasks_path=tasks_path)

def test_predict():
    # Cleanup result if it isn't empty yet
    result_vector_dir = get_test_projects_dir() / 'footballfields' / 'output_vector'
    if result_vector_dir.exists():
        shutil.rmtree(result_vector_dir, ignore_errors=True)
        # Make sure is is deleted now!
        if result_vector_dir.exists():
            assert True is False

    # Run task to predict
    tasks_path = get_test_projects_dir() / 'tasks_predict.csv'
    orthoseg.run_tasks(tasks_path=tasks_path)

    # Check results
    result_vector_path = result_vector_dir / 'BEFL-2019' / "footballfields_01_242_BEFL-2019.gpkg"
    assert result_vector_path.exists() is True
    result_gdf = geofile.read_file(result_vector_path)
    assert len(result_gdf) == 6

def test_postprocess():
    # Cleanup result if it isn't empty yet
    result_dir = get_test_projects_dir() / 'footballfields' / 'output_vector' / 'BEFL-2019'
    result_diss_path = result_dir / 'footballfields_01_242_BEFL-2019_dissolve.gpkg'
    result_simpl_path = result_dir / 'footballfields_01_242_BEFL-2019_dissolve_simpl.gpkg'
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
    assert len(result_gdf) == 5

    assert result_simpl_path.exists() is True
    result_gdf = geofile.read_file(result_simpl_path)
    assert len(result_gdf) == 5

if __name__ == '__main__':
    #test_load_images()
    #test_train()
    test_predict()
    #test_postprocess()
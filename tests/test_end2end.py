"""
Tests for functionalities in orthoseg.train.
"""

from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile

import gdown
import geofileops as gfo
import orthoseg
from orthoseg.helpers import config_helper as conf
import orthoseg.model.model_helper as mh
import pytest

from tests import test_helper

testprojects_dir = Path(tempfile.gettempdir()) / "orthoseg_test_end2end/sample_projects"
footballfields_dir = testprojects_dir / "footballfields"
projecttemplate_dir = testprojects_dir / "project_template"


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def test_1_init_testproject():
    # Use footballfields sample project for these end to end tests
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)


@pytest.mark.order(after="test_1_init_testproject")
def test_2_load_images():
    # Load project config to init some vars.
    config_path = footballfields_dir / "footballfields_BEFL-2019_test.ini"
    conf.read_orthoseg_config(config_path)
    image_cache_dir = conf.dirs.getpath("predict_image_input_dir")

    # Clean result if it isn't empty yet
    if image_cache_dir.exists():
        shutil.rmtree(image_cache_dir)
        # Make sure is is deleted now!
        assert not image_cache_dir.exists()

    # Run task to load images
    orthoseg.load_images(config_path)

    # Check if the right number of files was loaded
    assert image_cache_dir.exists()
    files = list(image_cache_dir.glob("**/*.jpg"))
    assert len(files) == 6


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
)
@pytest.mark.order(after="test_1_init_testproject")
def test_3_train():
    # Load project config to init some vars.
    config_path = footballfields_dir / "footballfields_train_test.ini"
    conf.read_orthoseg_config(config_path)

    # Init + cleanup result dirs
    traindata_id_result = 2
    training_dir = conf.dirs.getpath("training_dir")
    training_id_dir = training_dir / f"{traindata_id_result:02d}"
    if training_id_dir.exists():
        shutil.rmtree(training_id_dir)
    model_dir = conf.dirs.getpath("model_dir")
    if model_dir.exists():
        modelfile_paths = model_dir.glob(f"footballfields_{traindata_id_result:02d}_*")
        for modelfile_path in modelfile_paths:
            modelfile_path.unlink()

    # Make sure the label files in version 01 are older than those in the label dir
    # so a new model will be trained
    label_01_path = training_dir / "01/footballfields_BEFL-2019_polygons.gpkg"
    timestamp_old = datetime(year=2020, month=1, day=1).timestamp()
    os.utime(label_01_path, (timestamp_old, timestamp_old))

    # Run train session
    orthoseg.train(config_path)

    # Check if the training (image) data was created
    assert training_id_dir.exists()

    # Check if the new model was created
    best_model = mh.get_best_model(
        model_dir=conf.dirs.getpath("model_dir"),
        segment_subject=conf.general["segment_subject"],
        traindata_id=2,
    )

    assert best_model is not None
    assert best_model["traindata_id"] == traindata_id_result
    assert best_model["epoch"] == 0


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
)
@pytest.mark.order(after="test_2_load_images")
def test_4_predict():
    # Load project config to init some vars.
    config_path = footballfields_dir / "footballfields_BEFL-2019_test.ini"
    conf.read_orthoseg_config(config_path)

    # Cleanup result if it isn't empty yet
    predict_image_output_basedir = conf.dirs.getpath("predict_image_output_basedir")
    predict_image_output_dir = (
        predict_image_output_basedir.parent
        / f"{predict_image_output_basedir.name}_footballfields_02_0"
    )
    if predict_image_output_dir.exists():
        shutil.rmtree(predict_image_output_dir)
        # Make sure it is deleted now!
        assert not predict_image_output_dir.exists()
    result_vector_dir = conf.dirs.getpath("output_vector_dir")
    if result_vector_dir.exists():
        shutil.rmtree(result_vector_dir)
        # Make sure is is deleted now!
        assert not result_vector_dir.exists()

    # Download the version 01 model
    model_dir = conf.dirs.getpath("model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_hdf5_path = model_dir / "footballfields_01_0.97392_201.hdf5"
    if not model_hdf5_path.exists():
        gdown.download(
            id="1UlNorZ74ADCr3pL4MCJ_tnKRNoeZX79g", output=str(model_hdf5_path)
        )
    model_hyperparams_path = model_dir / "footballfields_01_hyperparams.json"
    if not model_hyperparams_path.exists():
        gdown.download(
            id="1NwrVVjx9IsjvaioQ4-bkPMrq7S6HeWIo", output=str(model_hyperparams_path)
        )
    model_modeljson_path = model_dir / "footballfields_01_model.json"
    if not model_modeljson_path.exists():
        gdown.download(
            id="1LNPLypM5in3aZngBKK_U4Si47Oe97ZWN", output=str(model_modeljson_path)
        )

    # Run task to predict
    orthoseg.predict(config_path)

    # Check results
    result_vector_path = result_vector_dir / "footballfields_01_201_BEFL-2019.gpkg"
    assert result_vector_path.exists()
    result_gdf = gfo.read_file(result_vector_path)
    if os.name == "nt":
        assert len(result_gdf) == 211
    else:
        assert len(result_gdf) == 211


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ and os.name == "nt",
    reason="crashes on github CI on windows",
)
@pytest.mark.order(after="test_5_predict")
def test_5_postprocess():
    # Load project config to init some vars.
    config_path = footballfields_dir / "footballfields_BEFL-2019_test.ini"
    conf.read_orthoseg_config(config_path)

    # Cleanup result if it isn't empty yet
    result_vector_dir = conf.dirs.getpath("output_vector_dir")
    result_diss_path = (
        result_vector_dir / "footballfields_01_201_BEFL-2019_dissolve.gpkg"
    )
    if result_diss_path.exists():
        gfo.remove(result_diss_path)

    # Run task to postprocess
    orthoseg.postprocess(config_path)

    # Check results
    assert result_diss_path.exists()
    result_gdf = gfo.read_file(result_diss_path)
    if os.name == "nt":
        assert len(result_gdf) == 207
    else:
        assert len(result_gdf) == 207

import os
from pathlib import Path
import shutil
import tempfile

import pytest

from orthoseg.lib import cleanup
from orthoseg.helpers import config_helper as conf
from tests import test_helper

testprojects_dir = Path(tempfile.gettempdir()) / "orthoseg_test_cleanup"
sampleprojects_dir = testprojects_dir / "sample_projects"
footballfields_dir = sampleprojects_dir / "footballfields"


def init_project():
    shutil.rmtree(path=testprojects_dir, ignore_errors=True)
    footballfields_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "project_template/projectfile.ini",
        dst=footballfields_dir / "footballfields.ini",
    )
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "imagelayers.ini",
        dst=sampleprojects_dir / "imagelayers.ini",
    )


def load_project_config():
    conf.read_orthoseg_config(
        config_path=footballfields_dir / "footballfields.ini",
        overrules=[
            "general.segment_subject=footballfields",
            "predict.image_layer=BEFL_2019",
        ],
    )


@pytest.mark.parametrize(
    "simulate, versions_to_retain, expected_files_in_dir",
    [
        (True, 3, 25),
        (False, 3, 15),
    ],
)
def test_cleanup_models(
    simulate: bool, versions_to_retain: int, expected_files_in_dir: int
):
    # Init test project
    init_project()
    models_dir = footballfields_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    models = [
        "0.44293_0.hdf5",
        "hyperparams.json",
        "log.csv",
        "model.json",
        "report.pdf",
    ]
    for x in range(1, 6):
        for model in models:
            with open(models_dir / f"footballfields_0{x}_{model}", "w"):
                pass

    # Load project config to init some vars.
    load_project_config()

    cleanup.clean_models(
        model_dir=conf.dirs.getpath("model_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    models = os.listdir(footballfields_dir / "models")
    assert len(models) == expected_files_in_dir


@pytest.mark.parametrize(
    "simulate, versions_to_retain, expected_files_in_dir",
    [
        (True, 2, 5),
        (False, 2, 2),
    ],
)
def test_cleanup_training(
    simulate: bool, versions_to_retain: int, expected_files_in_dir: int
):
    # Init test project
    init_project()
    training_dir = footballfields_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    for x in range(1, 6):
        dir = training_dir / f"0{x}"
        dir.mkdir(parents=True, exist_ok=True)
        with open(training_dir / dir / "footballfields_BEFL-2019_locations.gpkg", "w"):
            pass
        with open(training_dir / dir / "footballfields_BEFL-2019_polygons.gpkg", "w"):
            pass

    # Load project config to init some vars.
    load_project_config()

    cleanup.clean_training_data_directories(
        model_dir=conf.dirs.getpath("training_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    training_dirs = os.listdir(footballfields_dir / "training")
    assert len(training_dirs) == expected_files_in_dir


@pytest.mark.parametrize(
    "simulate, versions_to_retain, expected_files_in_dir",
    [
        (True, 1, 10),
        (False, 1, 2),
    ],
)
def test_cleanup_predictions(
    simulate: bool, versions_to_retain: int, expected_files_in_dir: int
):
    # Init test project
    init_project()
    predictions_dir = footballfields_dir / "output_vector"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for x in range(1, 6):
        dir = predictions_dir / f"BEFL_20{18 + x}"
        dir.mkdir(parents=True, exist_ok=True)
        for p in range(1, 6):
            with open(
                predictions_dir / dir / f"footballfields_0{p}_201_BEFL-20{18 + x}.gpkg",
                "w",
            ):
                pass
            with open(
                predictions_dir
                / dir
                / f"footballfields_0{p}_201_BEFL-20{18 + x}_dissolve.gpkg",
                "w",
            ):
                pass

    # Load project config to init some vars.
    load_project_config()

    cleanup.clean_predictions(
        model_dir=conf.dirs.getpath("output_vector_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    prediction_dirs = os.listdir(footballfields_dir / "output_vector")
    for prediction_dir in prediction_dirs:
        predictions = os.listdir(footballfields_dir / "output_vector" / prediction_dir)
        assert len(predictions) == expected_files_in_dir

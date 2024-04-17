import logging
import os
from pathlib import Path
import shutil

import pytest

from orthoseg.lib import cleanup
from orthoseg.helpers import config_helper as conf
from tests import test_helper


def create_projects_dir(tmp_path: Path) -> Path:
    testproject_dir = tmp_path / "orthoseg_test_cleanup"
    project_dir = testproject_dir / "footballfields"

    shutil.rmtree(path=testproject_dir, ignore_errors=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "project_template/projectfile.ini",
        dst=project_dir / "footballfields.ini",
    )
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "imagelayers.ini",
        dst=testproject_dir / "imagelayers.ini",
    )
    return project_dir


def load_project_config(path: Path):
    conf.read_orthoseg_config(
        config_path=path / "footballfields.ini",
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
    tmp_path, simulate: bool, versions_to_retain: int, expected_files_in_dir: int
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    models_dir = project_dir / "models"
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
            (models_dir / f"footballfields_0{x}_{model}").touch()

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanup.clean_models(
        model_dir=conf.dirs.getpath("model_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    models = os.listdir(project_dir / "models")
    assert len(models) == expected_files_in_dir


@pytest.mark.parametrize(
    "simulate, versions_to_retain, expected_files_in_dir",
    [
        (True, 2, 5),
        (False, 2, 2),
    ],
)
def test_cleanup_training(
    tmp_path, simulate: bool, versions_to_retain: int, expected_files_in_dir: int
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    training_dir = project_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    for x in range(1, 6):
        dir = training_dir / f"0{x}"
        dir.mkdir(parents=True, exist_ok=True)
        (training_dir / dir / "footballfields_BEFL-2019_locations.gpkg").touch()
        (training_dir / dir / "footballfields_BEFL-2019_polygons.gpkg").touch()

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanup.clean_training_data_directories(
        training_dir=conf.dirs.getpath("training_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    training_dirs = os.listdir(project_dir / "training")
    assert len(training_dirs) == expected_files_in_dir


@pytest.mark.parametrize(
    "simulate, versions_to_retain, expected_files_in_dir",
    [
        (True, 1, 10),
        (False, 1, 2),
    ],
)
def test_cleanup_predictions(
    tmp_path, simulate: bool, versions_to_retain: int, expected_files_in_dir: int
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    predictions_dir = project_dir / "output_vector"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for x in range(1, 6):
        dir = predictions_dir / f"BEFL_20{18 + x}"
        dir.mkdir(parents=True, exist_ok=True)
        for p in range(1, 6):
            (
                predictions_dir / dir / f"footballfields_0{p}_201_BEFL-20{18 + x}.gpkg"
            ).touch()
            (
                predictions_dir
                / dir
                / f"footballfields_0{p}_201_BEFL-20{18 + x}_dissolve.gpkg"
            ).touch()

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanup.clean_predictions(
        output_vector_dir=conf.dirs.getpath("output_vector_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    prediction_dirs = os.listdir(project_dir / "output_vector")
    for prediction_dir in prediction_dirs:
        predictions = os.listdir(project_dir / "output_vector" / prediction_dir)
        assert len(predictions) == expected_files_in_dir


def test_cleanup_non_existing_dir(caplog):
    caplog.set_level(logging.INFO)
    caplog.clear()
    path = Path("not_existing")
    cleanup.clean_models(model_dir=path, versions_to_retain=3, simulate=True)
    assert f"ERROR|Directory {path.name} doesn't exist" in caplog.text

    caplog.clear()
    cleanup.clean_training_data_directories(
        training_dir=path, versions_to_retain=3, simulate=True
    )
    assert f"ERROR|Directory {path.name} doesn't exist" in caplog.text

    caplog.clear()
    path = Path("not_existing/not_existing")
    cleanup.clean_predictions(
        output_vector_dir=path, versions_to_retain=3, simulate=True
    )
    assert f"ERROR|Directory {path.name} doesn't exist" in caplog.text

import logging
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


def create_model_files(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    files = [
        {"path": "footballfields_01", "min_versions_to_retain_to_keep": 4},
        {"path": "footballfields_02", "min_versions_to_retain_to_keep": 3},
        {"path": "footballfields_03", "min_versions_to_retain_to_keep": 2},
        {"path": "footballfields_04", "min_versions_to_retain_to_keep": 1},
    ]
    filetypes = [
        "0.44293_0.hdf5",
        "hyperparams.json",
        "log.csv",
        "model.json",
        "report.pdf",
    ]
    for file in files:
        for file_type in filetypes:
            (path / f"{file['path']}_{file_type}").touch()
    return files, filetypes


def create_training_files(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    dirs = [
        {"path": "01", "min_versions_to_retain_to_keep": 4},
        {"path": "02", "min_versions_to_retain_to_keep": 3},
        {"path": "03", "min_versions_to_retain_to_keep": 2},
        {"path": "04", "min_versions_to_retain_to_keep": 1},
    ]
    for dir in dirs:
        sub_dir = path / dir["path"]
        sub_dir.mkdir(parents=True, exist_ok=True)
    return dirs


def create_prediction_files(path: Path, imagelayer: str):
    path.mkdir(parents=True, exist_ok=True)
    files = [
        {
            "path": f"footballfields_01_201_{imagelayer}.gpkg",
            "min_versions_to_retain_to_keep": 4,
        },
        {
            "path": f"footballfields_02_201_{imagelayer}.gpkg",
            "min_versions_to_retain_to_keep": 3,
        },
        {
            "path": f"footballfields_03_201_{imagelayer}.gpkg",
            "min_versions_to_retain_to_keep": 2,
        },
        {
            "path": f"footballfields_04_201_{imagelayer}.gpkg",
            "min_versions_to_retain_to_keep": 1,
        },
        {
            "path": f"footballfields_01_201_{imagelayer}_dissolve.gpkg",
            "min_versions_to_retain_to_keep": 4,
        },
        {
            "path": f"footballfields_02_201_{imagelayer}_dissolve.gpkg",
            "min_versions_to_retain_to_keep": 3,
        },
        {
            "path": f"footballfields_03_201_{imagelayer}_dissolve.gpkg",
            "min_versions_to_retain_to_keep": 2,
        },
        {
            "path": f"footballfields_04_201_{imagelayer}_dissolve.gpkg",
            "min_versions_to_retain_to_keep": 1,
        },
    ]
    for file in files:
        (path / file["path"]).touch()
    return files


def load_project_config(path: Path):
    conf.read_orthoseg_config(
        config_path=path / "footballfields.ini",
        overrules=[
            "general.segment_subject=footballfields",
            "predict.image_layer=BEFL-2019",
        ],
    )


# Test parameters
# path : path of the filename that must be checked
# min_versions_to_retain_to_keep : if this version is less then versions_to_retain
#   the file (path) must still exist in the directory


def test_cleanup_non_existing_dir(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    caplog.clear()
    path = Path("not_existing")
    cleanup.clean_models(model_dir=path, versions_to_retain=3, simulate=True)
    assert f"Directory {path.name} doesn't exist" in caplog.text

    caplog.clear()
    cleanup.clean_training_data_directories(
        training_dir=path, versions_to_retain=3, simulate=True
    )
    assert f"Directory {path.name} doesn't exist" in caplog.text

    caplog.clear()
    path = Path("not_existing/not_existing")
    cleanup.clean_predictions(
        output_vector_dir=path, versions_to_retain=3, simulate=True
    )
    assert f"Directory {path.name} doesn't exist" in caplog.text


@pytest.mark.parametrize("simulate", [False, True])
@pytest.mark.parametrize("versions_to_retain", [4, 2, 1, 0])
def test_cleanup_models(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    models_dir = project_dir / "models"
    files, filetypes = create_model_files(path=models_dir)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanedup_models = cleanup.clean_models(
        model_dir=conf.dirs.getpath("model_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )

    # Asserts
    for file in files:
        if file["min_versions_to_retain_to_keep"] <= versions_to_retain:
            if simulate:
                assert file["path"] not in cleanedup_models
            else:
                for file_type in filetypes:
                    assert (models_dir / f"{file['path']}_{file_type}").exists()
        else:
            if simulate:
                assert file["path"] in cleanedup_models
            else:
                for file_type in filetypes:
                    assert not (models_dir / f"{file['path']}_{file_type}").exists()


@pytest.mark.parametrize("simulate", [False, True])
@pytest.mark.parametrize("versions_to_retain", [4, 2, 1, 0])
def test_cleanup_training(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    training_dir = project_dir / "training"
    dirs = create_training_files(path=training_dir)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanedup_trainingdata_dirs = cleanup.clean_training_data_directories(
        training_dir=conf.dirs.getpath("training_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )

    # Asserts
    for dir in dirs:
        if dir["min_versions_to_retain_to_keep"] <= versions_to_retain:
            if simulate:
                assert dir["path"] not in cleanedup_trainingdata_dirs
            else:
                assert (training_dir / dir["path"]).exists()
        else:
            if simulate:
                assert dir["path"] in cleanedup_trainingdata_dirs
            else:
                assert not (training_dir / dir["path"]).exists()


@pytest.mark.parametrize("simulate", [False, True])
@pytest.mark.parametrize("versions_to_retain", [4, 2, 1, 0])
def test_cleanup_predictions(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    imagelayer = "BEFL-2019"
    output_vector_dir = project_dir / "output_vector" / imagelayer
    files = create_prediction_files(path=output_vector_dir, imagelayer=imagelayer)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanedup_predictions = cleanup.clean_predictions(
        output_vector_dir=conf.dirs.getpath("output_vector_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )

    # Asserts
    for file in files:
        if file["min_versions_to_retain_to_keep"] <= versions_to_retain:
            if simulate:
                assert file["path"] not in [x.path.name for x in cleanedup_predictions]
            else:
                assert (output_vector_dir / file["path"]).exists()
        else:
            if simulate:
                assert file["path"] in [x.path.name for x in cleanedup_predictions]
            else:
                assert not (output_vector_dir / file["path"]).exists()


@pytest.mark.parametrize("simulate", [False])
@pytest.mark.parametrize(
    "versions_to_retain, removed_models, removed_training_dirs, removed_predictions",
    [(4, 0, 0, 0), (2, 2, 2, 8), (1, 3, 3, 12), (0, 4, 4, 16)],
)
def test_cleanup_project_dir(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
    removed_models: int,
    removed_training_dirs: int,
    removed_predictions: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    model_dir = project_dir / "models"
    create_model_files(path=model_dir)
    training_dir = project_dir / "training"
    create_training_files(path=training_dir)
    imagelayer = "BEFL-2019"
    output_vector_dir = project_dir / "output_vector" / imagelayer
    create_prediction_files(path=output_vector_dir, imagelayer=imagelayer)
    imagelayer = "BEFL-2020"
    output_vector_dir = project_dir / "output_vector" / imagelayer
    create_prediction_files(path=output_vector_dir, imagelayer=imagelayer)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    removed = cleanup.clean_project_dir(
        model_dir=conf.dirs.getpath("model_dir"),
        model_versions_to_retain=versions_to_retain,
        training_dir=conf.dirs.getpath("training_dir"),
        training_versions_to_retain=versions_to_retain,
        output_vector_dir=conf.dirs.getpath("output_vector_dir"),
        prediction_versions_to_retain=versions_to_retain,
        simulate=simulate,
    )

    # Asserts
    assert len(removed["models"]) == removed_models
    assert len(removed["training_dirs"]) == removed_training_dirs
    assert len(removed["predictions"]) == removed_predictions

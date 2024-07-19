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


def create_model_files(path: Path) -> dict[Path, int]:
    """
    Creates test model files and returns a dict with info about them.

    Args:
        path (Path): _description_

    Returns:
        dict[Path, int]: dict with Path as key and the min_versions_to_retain_to_keep as
            value
    """
    path.mkdir(parents=True, exist_ok=True)
    models = [
        {"path": "footballfields_01", "min_versions_to_retain_to_keep": 4},
        {"path": "footballfields_02", "min_versions_to_retain_to_keep": 3},
        {"path": "footballfields_03", "min_versions_to_retain_to_keep": 2},
        {"path": "footballfields_04", "min_versions_to_retain_to_keep": 1},
    ]
    model_files = [
        "0.44293_0.hdf5",
        "hyperparams.json",
        "log.csv",
        "model.json",
        "report.pdf",
    ]

    files_dict = {}
    for model in models:
        for file_type in model_files:
            filepath = path / f"{model['path']}_{file_type}"
            filepath.touch()
            files_dict[filepath] = model["min_versions_to_retain_to_keep"]

    return files_dict


def create_training_dirs(path: Path) -> dict[Path, int]:
    path.mkdir(parents=True, exist_ok=True)
    dirs = [
        {"path": "01", "min_versions_to_retain_to_keep": 4},
        {"path": "02", "min_versions_to_retain_to_keep": 3},
        {"path": "03", "min_versions_to_retain_to_keep": 2},
        {"path": "04", "min_versions_to_retain_to_keep": 1},
    ]

    dirs_dict = {}
    for dir in dirs:
        sub_dir = path / dir["path"]
        sub_dir.mkdir(parents=True, exist_ok=True)
        dirs_dict[sub_dir] = dir["min_versions_to_retain_to_keep"]

    return dirs_dict


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

    files_dict = {}
    for file in files:
        file_path = path / file["path"]
        file_path.touch()
        files_dict[file_path] = file["min_versions_to_retain_to_keep"]

    return files_dict


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
    model_files = create_model_files(path=models_dir)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    files_removed = cleanup.clean_models(
        model_dir=conf.dirs.getpath("model_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )

    # Asserts
    for filepath in model_files:
        if model_files[filepath] <= versions_to_retain:
            # The file should be retained
            assert filepath not in files_removed
            assert filepath.exists()
        else:
            # The file should be cleaned up
            assert filepath in files_removed
            if simulate:
                assert filepath.exists()
            else:
                assert not filepath.exists()


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
    dirs = create_training_dirs(path=training_dir)

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
        if dirs[dir] <= versions_to_retain:
            # Directory should be kept
            assert dir not in cleanedup_trainingdata_dirs
            assert dir.exists()
        else:
            # Directory should be removed
            assert dir in cleanedup_trainingdata_dirs
            if simulate:
                assert dir.exists()
            else:
                assert not dir.exists()


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
        if files[file] <= versions_to_retain:
            # File should be kept
            assert file not in cleanedup_predictions
            if not simulate:
                assert file.exists()
        else:
            # File should be removed
            assert file in cleanedup_predictions
            if simulate:
                assert file.exists()
            else:
                assert not file.exists()


@pytest.mark.parametrize("simulate", [False])
@pytest.mark.parametrize(
    "versions_to_retain, removed_model_files, removed_training_dirs, "
    "removed_prediction_files",
    [(4, 0, 0, 0), (2, 2 * 5, 2, 8), (1, 3 * 5, 3, 12), (0, 4 * 5, 4, 16)],
)
def test_cleanup_project_dir(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
    removed_model_files: int,
    removed_training_dirs: int,
    removed_prediction_files: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    model_dir = project_dir / "models"
    create_model_files(path=model_dir)
    training_dir = project_dir / "training"
    create_training_dirs(path=training_dir)
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
    assert len(removed["models"]) == removed_model_files
    assert len(removed["training_dirs"]) == removed_training_dirs
    assert len(removed["predictions"]) == removed_prediction_files

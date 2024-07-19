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


def create_model_files(dir: Path) -> list[tuple[Path, int]]:
    """
    Creates test model files and returns information about them.

    Args:
        dir (Path): the directory where the files should be created

    Returns:
        list[tuple[Path, int]]: the list of files created. Each item of the list is a
            tuple with:
              - the Path to the file
              - the minimum value of versions_to_retain that should lead to this file
                being kept when cleaning up the models.
    """
    models = {
        "footballfields_01": 4,
        "footballfields_02": 3,
        "footballfields_03": 2,
        "footballfields_04": 1,
    }
    model_files = [
        "0.44293_0.hdf5",
        "hyperparams.json",
        "log.csv",
        "model.json",
        "report.pdf",
    ]

    files = []
    dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        for file_type in model_files:
            filepath = dir / f"{model}_{file_type}"
            filepath.touch()
            files.append((filepath, models[model]))

    return files


def create_training_dirs(dir: Path) -> list[tuple[Path, int]]:
    """
    Creates test training directories and returns information about them.

    Args:
        dir (Path): the directory where the directories should be created

    Returns:
        list[tuple[Path, int]]: the list of directories created. Each item of the list
            is a tuple with:
              - the Path to the directory
              - the minimum value of versions_to_retain that should lead to this file
                being kept when cleaning up the training directories.
    """
    # Directory paths with the minimum number of versions to retain to keep them
    dirs = [(dir / "01", 4), (dir / "02", 3), (dir / "03", 2), (dir / "04", 1)]

    # Create directories
    dir.mkdir(parents=True, exist_ok=True)
    for training_dir, _ in dirs:
        training_dir.mkdir(parents=True, exist_ok=True)

    return dirs


def create_prediction_files(dir: Path, imagelayer: str) -> list[tuple[Path, int]]:
    """
    Creates test prediction files and returns information about them.

    Args:
        dir (Path): the directory where the directories should be created

    Returns:
        list[tuple[Path, int]]: the list of files created. Each item of the list
            is a tuple with:
              - the Path to the file
              - the minimum value of versions_to_retain that should lead to this file
                being kept when cleaning up the prediction files.
    """
    # File paths with the minimum number of versions to retain to keep them
    files = [
        (dir / f"footballfields_01_201_{imagelayer}.gpkg", 4),
        (dir / f"footballfields_02_201_{imagelayer}.gpkg", 3),
        (dir / f"footballfields_03_201_{imagelayer}.gpkg", 2),
        (dir / f"footballfields_04_201_{imagelayer}.gpkg", 1),
        (dir / f"footballfields_01_201_{imagelayer}_dissolve.gpkg", 4),
        (dir / f"footballfields_02_201_{imagelayer}_dissolve.gpkg", 3),
        (dir / f"footballfields_03_201_{imagelayer}_dissolve.gpkg", 2),
        (dir / f"footballfields_04_201_{imagelayer}_dissolve.gpkg", 1),
    ]

    # Create the files
    dir.mkdir(parents=True, exist_ok=True)
    for file, _ in files:
        file.touch()

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
    model_files = create_model_files(dir=models_dir)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    files_removed = cleanup.clean_models(
        model_dir=conf.dirs.getpath("model_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )

    # Asserts
    for path, min_version_to_retain in model_files:
        if min_version_to_retain <= versions_to_retain:
            # The file should be retained
            assert path not in files_removed
            assert path.exists()
        else:
            # The file should be cleaned up
            assert path in files_removed
            if simulate:
                assert path.exists()
            else:
                assert not path.exists()


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
    dirs = create_training_dirs(dir=training_dir)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanedup_trainingdata_dirs = cleanup.clean_training_data_directories(
        training_dir=conf.dirs.getpath("training_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )

    # Asserts
    for dir, min_version_to_retain in dirs:
        if min_version_to_retain <= versions_to_retain:
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
    files = create_prediction_files(dir=output_vector_dir, imagelayer=imagelayer)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanedup_predictions = cleanup.clean_predictions(
        output_vector_dir=conf.dirs.getpath("output_vector_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )

    # Asserts
    for path, min_version_to_retain in files:
        if min_version_to_retain <= versions_to_retain:
            # File should be kept
            assert path not in cleanedup_predictions
            if not simulate:
                assert path.exists()
        else:
            # File should be removed
            assert path in cleanedup_predictions
            if simulate:
                assert path.exists()
            else:
                assert not path.exists()


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
    create_model_files(dir=model_dir)
    training_dir = project_dir / "training"
    create_training_dirs(dir=training_dir)
    imagelayer = "BEFL-2019"
    output_vector_dir = project_dir / "output_vector" / imagelayer
    create_prediction_files(dir=output_vector_dir, imagelayer=imagelayer)
    imagelayer = "BEFL-2020"
    output_vector_dir = project_dir / "output_vector" / imagelayer
    create_prediction_files(dir=output_vector_dir, imagelayer=imagelayer)

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

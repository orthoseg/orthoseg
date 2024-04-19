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


def create_model_files(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "0.44293_0.hdf5",
        "hyperparams.json",
        "log.csv",
        "model.json",
        "report.pdf",
    ]
    for x in range(1, 5):
        for file in files:
            (model_dir / f"footballfields_0{x}_{file}").touch()


def create_training_files(training_dir: Path):
    training_dir.mkdir(parents=True, exist_ok=True)
    for x in range(1, 5):
        dir = training_dir / f"0{x}"
        dir.mkdir(parents=True, exist_ok=True)
        (training_dir / dir / "footballfields_BEFL-2019_locations.gpkg").touch()
        (training_dir / dir / "footballfields_BEFL-2019_polygons.gpkg").touch()


def create_prediction_files(output_vector_dir: Path, imagelayer: str):
    output_vector_dir.mkdir(parents=True, exist_ok=True)
    for x in range(1, 5):
        (output_vector_dir / f"footballfields_0{x}_201_{imagelayer}.gpkg").touch()
        (
            output_vector_dir / f"footballfields_0{x}_201_{imagelayer}_dissolve.gpkg"
        ).touch()


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
@pytest.mark.parametrize(
    "path, min_versions_to_retain_to_keep",
    [
        ("footballfields_01", 4),
        ("footballfields_02", 3),
        ("footballfields_03", 2),
        ("footballfields_04", 1),
    ],
)
def test_cleanup_models(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
    path: Path,
    min_versions_to_retain_to_keep: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    models_dir = project_dir / "models"
    files = [
        "0.44293_0.hdf5",
        "hyperparams.json",
        "log.csv",
        "model.json",
        "report.pdf",
    ]
    create_model_files(model_dir=models_dir)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanedup_models = cleanup.clean_models(
        model_dir=conf.dirs.getpath("model_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    if min_versions_to_retain_to_keep <= versions_to_retain:
        if simulate:
            assert path not in cleanedup_models
        else:
            for file in files:
                assert (models_dir / f"{path}_{file}").exists()
    else:
        if simulate:
            assert path in cleanedup_models
        else:
            for file in files:
                assert not (models_dir / f"{path}_{file}").exists()


@pytest.mark.parametrize("simulate", [False, True])
@pytest.mark.parametrize("versions_to_retain", [4, 2, 1, 0])
@pytest.mark.parametrize(
    "path, min_versions_to_retain_to_keep",
    [
        ("01", 4),
        ("02", 3),
        ("03", 2),
        ("04", 1),
    ],
)
def test_cleanup_training(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
    path: Path,
    min_versions_to_retain_to_keep: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    training_dir = project_dir / "training"
    create_training_files(training_dir=training_dir)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanedup_trainingdata_dirs = cleanup.clean_training_data_directories(
        training_dir=conf.dirs.getpath("training_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    if min_versions_to_retain_to_keep <= versions_to_retain:
        if simulate:
            assert path not in cleanedup_trainingdata_dirs
        else:
            assert (training_dir / path).exists()
    else:
        if simulate:
            assert path in cleanedup_trainingdata_dirs
        else:
            assert not (training_dir / path).exists()


@pytest.mark.parametrize("simulate", [False, True])
@pytest.mark.parametrize("versions_to_retain", [4, 2, 1, 0])
@pytest.mark.parametrize(
    "path, min_versions_to_retain_to_keep",
    [
        ("footballfields_01_201_BEFL-2019.gpkg", 4),
        ("footballfields_02_201_BEFL-2019.gpkg", 3),
        ("footballfields_03_201_BEFL-2019.gpkg", 2),
        ("footballfields_04_201_BEFL-2019.gpkg", 1),
        ("footballfields_01_201_BEFL-2019_dissolve.gpkg", 4),
        ("footballfields_02_201_BEFL-2019_dissolve.gpkg", 3),
        ("footballfields_03_201_BEFL-2019_dissolve.gpkg", 2),
        ("footballfields_04_201_BEFL-2019_dissolve.gpkg", 1),
    ],
)
def test_cleanup_predictions(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
    path: Path,
    min_versions_to_retain_to_keep: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    imagelayer = "BEFL-2019"
    predictions_dir = project_dir / "output_vector" / imagelayer
    create_prediction_files(output_vector_dir=predictions_dir, imagelayer=imagelayer)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    # Cleanup
    cleanedup_predictions = cleanup.clean_predictions(
        output_vector_dir=conf.dirs.getpath("output_vector_dir"),
        versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    if min_versions_to_retain_to_keep <= versions_to_retain:
        if simulate:
            assert path not in [x.path.name for x in cleanedup_predictions]
        else:
            assert (predictions_dir / path).exists()
    else:
        if simulate:
            assert path in [x.path.name for x in cleanedup_predictions]
        else:
            assert not (predictions_dir / path).exists()


@pytest.mark.parametrize("simulate", [False])
@pytest.mark.parametrize(
    "versions_to_retain, expected_models, expected_training_dirs, expected_predictions",
    [
        (2, 2, 2, 8),
    ],
)
def test_cleanup_project_dir(
    tmp_path: Path,
    simulate: bool,
    versions_to_retain: int,
    expected_models: int,
    expected_training_dirs: int,
    expected_predictions: int,
):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)

    # Creating dummy files
    model_dir = project_dir / "models"
    create_model_files(model_dir=model_dir)
    training_dir = project_dir / "training"
    create_training_files(training_dir=training_dir)
    imagelayer = "BEFL-2019"
    output_vector_dir = project_dir / "output_vector" / imagelayer
    create_prediction_files(output_vector_dir=output_vector_dir, imagelayer=imagelayer)
    imagelayer = "BEFL-2020"
    output_vector_dir = project_dir / "output_vector" / imagelayer
    create_prediction_files(output_vector_dir=output_vector_dir, imagelayer=imagelayer)

    # Load project config to init some vars.
    load_project_config(path=project_dir)

    cleanedup_files = cleanup.clean_project_dir(
        model_dir=conf.dirs.getpath("model_dir"),
        model_versions_to_retain=versions_to_retain,
        training_dir=conf.dirs.getpath("training_dir"),
        training_versions_to_retain=versions_to_retain,
        output_vector_dir=conf.dirs.getpath("output_vector_dir"),
        prediction_versions_to_retain=versions_to_retain,
        simulate=simulate,
    )
    assert len(cleanedup_files["models"]) == expected_models
    assert len(cleanedup_files["training_dirs"]) == expected_training_dirs
    assert len(cleanedup_files["predictions"]) == expected_predictions

from pathlib import Path
import shutil
import pytest

import orthoseg
from orthoseg.validate import _validate_args
from tests import test_helper


@pytest.mark.parametrize(
    "args",
    [
        (
            [
                "--config",
                "X:/Monitoring/OrthoSeg/test/test.ini",
                "predict.image_layer=LT-2023",
            ]
        )
    ],
)
def test_validate_args(args):
    valid_args = _validate_args(args=args)
    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None


def test_validate_error(tmp_path):
    # Create test project
    project_dir = create_projects_dir(tmp_path=tmp_path)
    # Create dummy files
    training_dir = project_dir / "training"
    create_training_files(path=training_dir)
    overrules = [
        "general.segment_subject=footballfields",
        "predict.image_layer=BEFL-2019",
    ]

    with pytest.raises(
        Exception,
        match="ERROR while running validate for task footballfields",
    ):
        orthoseg.validate(
            config_path=project_dir / "footballfields.ini", config_overrules=overrules
        )


def create_projects_dir(tmp_path: Path) -> Path:
    testproject_dir = tmp_path / "orthoseg_test_validate"
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


def create_training_files(path: Path):
    sub_dir = path / "01"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "footballfields_BEFL-2019_locations.gpkg").touch()
    (sub_dir / "footballfields_BEFL-2019_polygons.gpkg").touch()

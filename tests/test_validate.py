import os
import shutil
from datetime import datetime

import pytest

import orthoseg
from orthoseg.helpers import config_helper as conf
from orthoseg.validate import _validate_args
from tests import test_helper
from tests.test_helper import SportsFields


def test_validate(tmp_path):
    # Copy sportsfields sample project for this test
    testprojects_dir = tmp_path / "sample_projects"
    sportsfields_dir = testprojects_dir / SportsFields.subject
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)

    # Load project config to init some vars.
    config_path = sportsfields_dir / SportsFields.config_path.name
    conf.read_orthoseg_config(config_path)

    # Init + cleanup result dirs
    traindata_id_result = 2
    training_dir = conf.dirs.getpath("training_dir")
    training_id_dir = training_dir / f"{traindata_id_result:02d}"
    if training_id_dir.exists():
        shutil.rmtree(training_id_dir)
    model_dir = conf.dirs.getpath("model_dir")
    if model_dir.exists():
        modelfile_paths = model_dir.glob(f"sportsfields_{traindata_id_result:02d}_*")
        for modelfile_path in modelfile_paths:
            modelfile_path.unlink()

    # Make sure the label files in version 01 are older than those in the label dir
    # so a new model will be trained
    label_01_path = training_dir / "01/sportsfields_BEFL-2025_polygons.gpkg"
    timestamp_old = datetime(year=2020, month=1, day=1).timestamp()
    os.utime(label_01_path, (timestamp_old, timestamp_old))

    # Run validate
    orthoseg.validate(config_path=config_path)


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
    # Create test project without label dir.
    # Validation should give an error that it should exist.
    tmp_project_dir = tmp_path / SportsFields.subject
    tmp_project_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_project_dir / SportsFields.config_path.name
    shutil.copy(src=SportsFields.config_path, dst=config_path)
    shutil.copy(src=test_helper.sampleprojects_dir / "imagelayers.ini", dst=tmp_path)
    (tmp_path / "project_defaults_overrule.ini").touch()

    with pytest.raises(
        RuntimeError,
        match="ERROR in validate for sportsfields: Label dir doesn't exist",
    ):
        orthoseg.validate(config_path=config_path)

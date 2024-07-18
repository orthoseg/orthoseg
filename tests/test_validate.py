from datetime import datetime
import os
import shutil

import pytest

import orthoseg
from orthoseg.helpers import config_helper as conf
from orthoseg.validate import _validate_args
from tests import test_helper
from tests.test_helper import SampleProjectFootball


def test_validate(tmp_path):
    # Copy footballfields sample project for this test
    testprojects_dir = tmp_path / "sample_projects"
    footballfields_dir = testprojects_dir / "footballfields"
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)

    # Load project config to init some vars.
    config_path = footballfields_dir / SampleProjectFootball.train_config_path.name
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
    # Create test project
    project_dir = tmp_path / "footballfields"
    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        src=test_helper.SampleProjectFootball.project_dir
        / "footballfields_BEFL-2019.ini",
        dst=project_dir / "footballfields_BEFL-2019.ini",
    )
    shutil.copyfile(
        src=test_helper.sampleprojects_dir / "imagelayers.ini",
        dst=tmp_path / "imagelayers.ini",
    )

    with pytest.raises(
        Exception,
        match="ERROR while running validate for task footballfields_BEFL-2019",
    ):
        orthoseg.validate(
            config_path=project_dir / "footballfields_BEFL-2019.ini",
        )

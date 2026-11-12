"""Tests for module postprocess."""

import shutil

import pytest

from orthoseg import postprocess
from orthoseg.helpers import config_helper as conf
from orthoseg.postprocess import _postprocess_args
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
def test_postprocess_args(args):
    valid_args = _postprocess_args(args=args)
    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None


def test_postprocess_backward_compat_existing_predict(tmp_path):
    # Test that postprocess reuses existing previous standard predict file
    testprojects_dir = tmp_path / "sample_projects"
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)
    project_dir = testprojects_dir / test_helper.SportsFields.subject
    config_path = project_dir / test_helper.SportsFields.config_path.name

    result_path = postprocess(config_path=config_path)
    assert result_path.exists()
    assert result_path.name == "sportsfields_01_276_BEFL-2025-sportsfields.gpkg"


def test_postprocess_custom_output_vector_names(tmp_path):
    testprojects_dir = tmp_path / "sample_projects"
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)
    project_dir = testprojects_dir / test_helper.SportsFields.subject
    config_path = project_dir / test_helper.SportsFields.config_path.name

    conf.read_orthoseg_config(config_path=config_path)
    output_vector_dir = conf.dirs.getpath("output_vector_dir")

    # Copy the existing prediction to the custom predict filename
    existing_predict = (
        output_vector_dir / "sportsfields_01_276_BEFL-2025-sportsfields_predict.gpkg"
    )
    custom_predict = output_vector_dir / "custom_sportsfields_predict.gpkg"
    shutil.copy(existing_predict, custom_predict)

    overrules = [
        "files.output_vector_predict_name=custom_{segment_subject}_predict.gpkg",
        "files.output_vector_postp_name=custom_{segment_subject}_postp.gpkg",
    ]

    result_path = postprocess(config_path=config_path, config_overrules=overrules)
    assert result_path.exists()
    assert result_path.name == "custom_sportsfields_postp.gpkg"


def test_postprocess_error_handling():
    """Force an error so the general error handler in postprocess is tested."""
    with pytest.raises(
        RuntimeError, match="ERROR in postprocess for sportsfields_01 on UNEXISTING"
    ):
        postprocess(
            config_path=test_helper.SportsFields.config_path,
            config_overrules=["predict.image_layer=UNEXISTING"],
        )

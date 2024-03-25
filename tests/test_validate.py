"""
Tests for functionalities in orthoseg.validate.
"""

from contextlib import nullcontext
from pathlib import Path
import shutil
import tempfile

import pytest
import orthoseg
from tests import test_helper

testprojects_dir = (
    Path(tempfile.gettempdir()) / "orthoseg_test_validate/sample_projects"
)
footballfields_dir = testprojects_dir / "footballfields"


def test_1_init_testproject():
    # Use footballfields sample project for these end to end tests
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)


@pytest.mark.parametrize("exp_error", [(False)])
@pytest.mark.order(after="test_1_init_testproject")
def test_2_validate(exp_error: bool):
    if exp_error:
        handler = pytest.raises(Exception)
    else:
        handler = nullcontext()
    with handler:
        # config_path = Path("X:/Monitoring/OrthoSeg/fields-arable/fields-arable.ini")
        config_path = footballfields_dir / "footballfields_train.ini"
        orthoseg.validate(config_path=config_path)

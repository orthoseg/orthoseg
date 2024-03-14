"""
Tests for functionalities in orthoseg.validate.
"""

from contextlib import nullcontext
from pathlib import Path
import tempfile

import pytest
import orthoseg

testprojects_dir = Path(tempfile.gettempdir()) / "orthoseg_test_end2end/sample_projects"


@pytest.mark.parametrize("exp_error", [(False), (True)])
def test_validate(exp_error: bool):
    if exp_error:
        handler = pytest.raises(Exception)
    else:
        handler = nullcontext()
    with handler:
        config_path = Path("X:/Monitoring/OrthoSeg/fields-arable/fields-arable.ini")
        orthoseg.validate(config_path=config_path)

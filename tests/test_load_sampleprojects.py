"""
Tests for functionalities in orthoseg.train.
"""

import os
import platform
import shutil
import sys
from pathlib import Path

import pytest

from orthoseg import load_sampleprojects


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ
    and not (
        platform.system() == "Linux"
        and sys.version_info.major == 3
        and sys.version_info.minor == 10
    ),
    reason="on github CI, run this only in one env to avoid rate limit exceeded",
)
def test_load_sampleprojects(tmp_path):
    sampleprojects_dir = tmp_path / "sample_projects"
    shutil.rmtree(sampleprojects_dir, ignore_errors=True)
    load_sampleprojects.load_sampleprojects(dest_dir=sampleprojects_dir.parent)

    # Check if the files were correctly loaded
    assert sampleprojects_dir.exists()
    assert (sampleprojects_dir / "imagelayers.ini").exists()
    assert (sampleprojects_dir / "project_defaults_overrule.ini").exists()
    assert (sampleprojects_dir / "run_footballfields.py").exists()

    footballfields_dir = sampleprojects_dir / "footballfields"
    assert footballfields_dir.exists()
    files = list((footballfields_dir).glob("**/*.ini"))
    assert len(files) > 0
    files = list((footballfields_dir / "labels").glob("**/*.gpkg"))
    assert len(files) == 2
    model_path = footballfields_dir / "models" / "footballfields_01_0.92512_242.hdf5"
    assert model_path.exists()
    # The model should be larger than 50 MB, otherwise not normal
    assert model_path.stat().st_size > 50 * 1024 * 1024

    projecttemplate_dir = sampleprojects_dir / "project_template"
    assert projecttemplate_dir.exists()
    files = list((projecttemplate_dir).glob("**/*.ini"))
    assert len(files) > 0
    files = list((projecttemplate_dir / "labels").glob("**/*.gpkg"))
    assert len(files) == 2

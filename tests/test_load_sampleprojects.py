# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.train.
"""

import os
from pathlib import Path
import platform
import shutil
import sys

import pytest

# Add path so the local orthoseg packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg import load_sampleprojects

# ----------------------------------------------------
# Tests
# ----------------------------------------------------


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

    # Check if the right number of files was loaded
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

    projecttemplate_dir = sampleprojects_dir / "project_template"
    assert projecttemplate_dir.exists()
    files = list((projecttemplate_dir).glob("**/*.ini"))
    assert len(files) > 0
    files = list((projecttemplate_dir / "labels").glob("**/*.gpkg"))
    assert len(files) == 2

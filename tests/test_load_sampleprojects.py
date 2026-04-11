"""Tests for module load_sampleprojects."""

import os
import platform
import shutil
import sys
from pathlib import Path

import pytest

from orthoseg import load_sampleprojects
from orthoseg.helpers import config_helper as conf
from orthoseg.load_sampleprojects import _parse_load_sampleprojects_args


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize(
    "args, exp_ssl_verify",
    [
        (["C:/Monitoring/OrthoSeg/test"], True),
        (["C:/Monitoring/OrthoSeg/test", "--ssl_verify", "FaLsE"], False),
        (["C:/Monitoring/OrthoSeg/test", "--ssl_verify", "TrUe"], True),
        (["C:/Monitoring/OrthoSeg/test", "--ssl_verify", "abc"], "abc"),
    ],
)
def test_load_images_args(args, exp_ssl_verify):
    valid_args = _parse_load_sampleprojects_args(args=args)
    assert valid_args is not None
    assert valid_args["dest_dir"] is not None
    if isinstance(exp_ssl_verify, bool):
        assert valid_args["ssl_verify"] is exp_ssl_verify
    else:
        assert valid_args["ssl_verify"] == exp_ssl_verify


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ
    and not (
        platform.system() == "Linux"
        and sys.version_info.major == 3
        and sys.version_info.minor == 10
    ),
    reason="on github CI, run this only in one env to avoid rate limit exceeded",
)
def test_load_sampleprojects():
    """Test loading the sample projects.

    Test this using ~ in the path to check if this is correctly handled as this is
    used in the documentation.
    """
    dest_dir = Path("~/orthoseg_test")

    # The path needs to be expanded to have exist, rmtree,... work, at least on windows
    shutil.rmtree(dest_dir.expanduser(), ignore_errors=True)

    # Load the sample projects. Has to handle ~ in the path correctly.
    load_sampleprojects.load_sampleprojects(dest_dir=dest_dir)

    # Try loading the config of the footballfields sample project
    # Use the unexpanded path here as read_orthoseg_config should handle this correctly
    config_path = dest_dir / "sample_projects/footballfields/footballfields.ini"
    conf.read_orthoseg_config(config_path)

    # Check if the files were correctly loaded
    sampleprojects_dir = dest_dir.expanduser() / "sample_projects"

    # The path needs to be expanded to get e.g. exist to work, at least on windows
    sampleprojects_exp_dir = sampleprojects_dir.expanduser()
    assert sampleprojects_exp_dir.exists()
    assert (sampleprojects_exp_dir / "imagelayers.ini").exists()
    assert (sampleprojects_exp_dir / "project_defaults_overrule.ini").exists()
    assert (sampleprojects_exp_dir / "run_footballfields.py").exists()

    footballfields_dir = sampleprojects_exp_dir / "footballfields"
    assert footballfields_dir.exists()
    files = list((footballfields_dir).glob("**/*.ini"))
    assert len(files) > 0
    files = list((footballfields_dir / "labels").glob("**/*.gpkg"))
    assert len(files) == 2
    model_path = footballfields_dir / "models" / "footballfields_01_0.97392_201.hdf5"
    assert model_path.exists()
    # The model should be larger than 40 MB, otherwise not normal
    assert model_path.stat().st_size > 40 * 1024 * 1024

    projecttemplate_dir = sampleprojects_exp_dir / "project_template"
    assert projecttemplate_dir.exists()
    files = list((projecttemplate_dir).glob("**/*.ini"))
    assert len(files) > 0
    files = list((projecttemplate_dir / "labels").glob("**/*.gpkg"))
    assert len(files) == 2

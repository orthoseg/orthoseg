import shutil

import pytest

import orthoseg
from orthoseg.helpers import config_helper as conf
from tests import test_helper


@pytest.mark.parametrize(
    "football_ini",
    [
        "footballfields_BEFL-2023-WMTS_test.ini",
        "footballfields_BEFL-2023-XYZ_test.ini",
    ],
)
def test_load_images_wmts(tmp_path, football_ini: str):
    # Use footballfields sample project for these end to end tests
    testprojects_dir = tmp_path / "sample_projects"
    footballfields_dir = testprojects_dir / "footballfields"
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)
    # Load project config to init some vars.
    config_path = footballfields_dir / football_ini
    conf.read_orthoseg_config(config_path)
    image_cache_dir = conf.dirs.getpath("predict_image_input_dir")

    # Run task to load images
    orthoseg.load_images(config_path)

    # Check if the right number of files was loaded
    assert image_cache_dir.exists()
    files = list(image_cache_dir.glob("**/*.jpg"))
    assert len(files) == 6

import shutil

import orthoseg
from orthoseg.helpers import config_helper as conf
from tests import test_helper


def test_load_images_wmts(tmp_path):
    # Use footballfields sample project for these end to end tests
    testprojects_dir = tmp_path / "orthoseg_test_end2end/sample_projects"
    footballfields_dir = testprojects_dir / "footballfields"
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)
    # Load project config to init some vars.
    config_path = footballfields_dir / "footballfields_BEFL-2023-WMTS_test.ini"
    conf.read_orthoseg_config(config_path)
    image_cache_dir = conf.dirs.getpath("predict_image_input_dir")

    # Run task to load images
    orthoseg.load_images(config_path)

    # Check if the right number of files was loaded
    assert image_cache_dir.exists()
    files = list(image_cache_dir.glob("**/*.jpg"))
    assert len(files) == 6

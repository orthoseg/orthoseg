import shutil

import pytest

import orthoseg
from tests import test_helper


@pytest.mark.parametrize(
    "overrules",
    [
        ["predict.image_layer=BEFL-2023-WMTS"],
        ["predict.image_layer=BEFL-2023-XYZ"],
        [],
    ],
)
def test_load_images(tmp_path, overrules):
    # Use footballfields sample project for these end to end tests
    testprojects_dir = tmp_path / "sample_projects"
    footballfields_dir = testprojects_dir / "footballfields"
    image_cache_dir = testprojects_dir / "_image_cache"
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)

    # Run task to load images
    orthoseg.load_images(
        footballfields_dir / "footballfields_BEFL-2019_test.ini",
        config_overrules=overrules,
    )

    # Check if the right number of files was loaded
    assert image_cache_dir.exists()
    files = list(image_cache_dir.glob("**/*.jpg"))
    assert len(files) == 6

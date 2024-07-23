from contextlib import nullcontext

import pytest

from orthoseg.util.data import aidetection_info


@pytest.mark.parametrize(
    "file, exp_error",
    [
        ("footballfields_01_201_BEFL-2019_dissolve.gpkg", False),
        ("footballfields_01_201.gpkg", True),
    ],
)
def test_aidetection_info(tmp_path, file: str, exp_error: bool):
    # Create test project
    testproject_dir = tmp_path / "orthoseg_test_aidetection_info"
    testproject_dir.mkdir(parents=True, exist_ok=True)
    path = testproject_dir / file
    path.touch()

    if exp_error:
        matchstr = r"Error in get_aidetection_info on .*"
        handler = pytest.raises(ValueError, match=matchstr)
    else:
        handler = nullcontext()
    with handler:
        # Get aidetection info
        ai_detection_info = aidetection_info(path=path)

        assert ai_detection_info.path == path
        assert ai_detection_info.subject == "footballfields"
        assert ai_detection_info.traindata_version == 1
        assert ai_detection_info.image_layer == "BEFL-2019"
        assert ai_detection_info.image_layer_year == 2019
        assert ai_detection_info.postprocessing == "dissolve"

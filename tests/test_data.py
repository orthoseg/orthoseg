from contextlib import AbstractContextManager, nullcontext

import pytest

from orthoseg.util.data import aidetection_info


@pytest.mark.parametrize(
    "file, exp_error, expected",
    [
        (
            "test-subject_01_201_BEFL-2019_dissolve.gpkg",
            False,
            ("test-subject", 1, "BEFL-2019", "dissolve"),
        ),
        (
            "test-subject_BEFL-2019_01.gpkg",
            False,
            ("test-subject", 1, "BEFL-2019", ""),
        ),
        (
            "test-subject_BEFL-2019_01_predict.gpkg",
            False,
            ("test-subject", 1, "BEFL-2019", "predict"),
        ),
        (
            "test-subject_01_BEFL-2019.gpkg",
            False,
            ("test-subject", 1, "BEFL-2019", ""),
        ),
        ("test-subject_01_201.gpkg", True, None),
    ],
)
def test_aidetection_info(
    tmp_path,
    file: str,
    exp_error: bool,
    expected: tuple[str, int, str, str] | None,
):
    # Create test project
    testproject_dir = tmp_path / "orthoseg_test_aidetection_info"
    testproject_dir.mkdir(parents=True, exist_ok=True)
    path = testproject_dir / file
    path.touch()

    handler: AbstractContextManager
    if exp_error:
        matchstr = r"Error in get_aidetection_info on .*"
        handler = pytest.raises(ValueError, match=matchstr)
    else:
        handler = nullcontext()

    with handler:
        # Get aidetection info
        ai_detection_info = aidetection_info(path=path)

        assert expected is not None
        assert ai_detection_info.path == path
        assert ai_detection_info.subject == expected[0]
        assert ai_detection_info.traindata_version == expected[1]
        assert ai_detection_info.image_layer == expected[2]
        assert ai_detection_info.postprocessing == expected[3]

import shutil
import pytest

import orthoseg
from orthoseg.validate import _validate_args
from tests import test_helper


@pytest.mark.parametrize(
    "args",
    [
        (
            [
                "--config",
                "X:/Monitoring/OrthoSeg/test/test.ini",
                "predict.image_layer=LT-2023",
            ]
        )
    ],
)
def test_validate_args(args):
    try:
        valid_args = _validate_args(args=args)
    except Exception as ex:
        print(ex)
    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None


def test_validate_error(tmp_path):
    # Create test project
    project_dir = tmp_path / "footballfields"
    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        src=test_helper.SampleProjectFootball.project_dir
        / "footballfields_BEFL-2019.ini",
        dst=project_dir / "footballfields_BEFL-2019.ini",
    )

    with pytest.raises(Exception, match="Layer config file not found:"):
        orthoseg.validate(
            config_path=project_dir / "footballfields_BEFL-2019.ini",
        )

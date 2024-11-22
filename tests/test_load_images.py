import pytest

from orthoseg import load_images
from tests import test_helper


def test_load_images_error_handling():
    """Force an error so the general error handler in predict is tested."""
    with pytest.raises(
        RuntimeError,
        match="ERROR in load_images for footballfields_BEFL-2019_test",
    ):
        load_images(
            config_path=test_helper.SampleProjectFootball.predict_config_path,
            config_overrules=["predict.image_pixel_width=INVALID_TYPE"],
        )

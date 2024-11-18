import pytest

from orthoseg import postprocess
from tests import test_helper


def test_postprocess_error_handling():
    """Force an error so the general error handler in postprocess is tested."""
    with pytest.raises(
        RuntimeError,
        match="ERROR in postprocess for footballfields_BEFL-2019_test on UNEXISTING",
    ):
        postprocess(
            config_path=test_helper.SampleProjectFootball.predict_config_path,
            config_overrules=["predict.image_layer=UNEXISTING"],
        )

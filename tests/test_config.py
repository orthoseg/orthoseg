from orthoseg.helpers import config_helper as conf
from tests.test_helper import SampleProjectFootball


def test_read_orthoseg_config_predict():
    # Load project config to init some vars.
    conf.read_orthoseg_config(SampleProjectFootball.predict_config_path)

    layer = conf.image_layers.get("BEFL-2019")
    assert layer is not None
    pixel_x_size = layer.get("pixel_x_size")
    assert pixel_x_size == 0.25
    pixel_y_size = layer.get("pixel_y_size")
    assert pixel_y_size == 0.25

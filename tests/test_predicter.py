from contextlib import nullcontext
from pathlib import Path
import pytest


from orthoseg.helpers import config_helper as conf
from orthoseg.lib import predicter
from orthoseg.model import model_factory as mf
from orthoseg.model import model_helper as mh


@pytest.mark.parametrize(
    "ini_file, force_model_traindata_id, no_images_ok, exp_error",
    [
        ("X:/Monitoring/OrthoSeg/horsepastures/horsepastures.ini", 4, False, True),
        ("X:/Monitoring/OrthoSeg/horsepastures/horsepastures.ini", 4, True, False),
    ],
)
def test_predict_dir_test_dataset(
    ini_file: str,
    force_model_traindata_id: int,
    no_images_ok: bool,
    exp_value_warning: bool,
):
    # Init
    config_path = Path(ini_file)
    conf.read_orthoseg_config(config_path)

    classes = conf.train.getdict("classes")
    training_dir = conf.dirs.getpath("training_dir") / f"{force_model_traindata_id:02d}"
    testdata_dir = training_dir / "test"
    model_dir = conf.dirs.getpath("model_dir")
    best_model_curr_train_version = mh.get_best_model(
        model_dir=model_dir,
    )

    # Assert to evade typing warnings
    assert best_model_curr_train_version is not None
    model = mf.load_model(best_model_curr_train_version["filepath"], compile=False)

    if exp_value_warning:
        handler = pytest.raises(ValueError)
    else:
        handler = nullcontext()

    with handler:
        predicter.predict_dir(
            model=model,
            input_image_dir=testdata_dir / "image",
            output_image_dir=testdata_dir / "output",
            output_vector_path=None,
            classes=classes,
            no_images_ok=no_images_ok,
        )

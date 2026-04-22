"""Tests for module trainer."""

from unittest.mock import patch

import pytest

import orthoseg.model.model_helper as mh
from orthoseg.lib import trainer


def test_train_error_model_exists_no_preload(tmp_path):
    """Raise ValueError when a model exists but no preload file is specified."""
    augmentations = {"fill_mode": "constant", "cval": 0}
    hyperparams = mh.HyperParams(
        architecture=mh.ArchitectureParams(
            architecture="mobilenetv2+unet", classes=["background", "class1"]
        ),
        train=mh.TrainParams(
            image_augmentations=augmentations, mask_augmentations=augmentations
        ),
    )

    fake_best_model = {"model_filepath": tmp_path / "some_model.keras"}

    with patch("orthoseg.lib.trainer.mh.get_best_model", return_value=fake_best_model):
        with pytest.raises(
            ValueError, match="Model exists but preload model file not specified"
        ):
            trainer.train(
                traindata_dir=tmp_path,
                validationdata_dir=tmp_path,
                model_save_dir=tmp_path,
                segment_subject="test",
                traindata_id=1,
                hyperparams=hyperparams,
                model_preload_filepath=None,
            )

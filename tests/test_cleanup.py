import os
from pathlib import Path
import shutil
import tempfile

import pytest
from orthoseg import cleanup
from tests import test_helper

testprojects_dir = Path(tempfile.gettempdir()) / "orthoseg_test_cleanup/sample_projects"
cleanup_dir = testprojects_dir / "cleanup"


def test_1_init_testproject():
    # Use footballfields sample project for these cleanup tests
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)

    # models, trainingdata and predictions are x time duplicated
    models_dir = cleanup_dir / "models"
    models = os.listdir(models_dir)
    predictions_dir = cleanup_dir / "output_vector"
    prediction_dir = os.listdir(predictions_dir)[0]
    predictions = os.listdir(predictions_dir / prediction_dir)
    training_dir = cleanup_dir / "training"
    training = os.listdir(training_dir)[0]
    for i in range(2, 6):
        for model in models:
            new_model = model.replace("_01_", f"_0{i}_")
            shutil.copy(models_dir / model, models_dir / new_model)
        for prediction in predictions:
            new_prediction = prediction.replace("_01_", f"_0{i}_")
            shutil.copy(
                predictions_dir / prediction_dir / prediction,
                predictions_dir / prediction_dir / new_prediction,
            )
        new_training = training.replace("01", f"0{i}")
        shutil.copytree(training_dir / training, training_dir / new_training)
    new_prediction_dir = prediction_dir + "_2"
    shutil.copytree(
        predictions_dir / prediction_dir, predictions_dir / new_prediction_dir
    )


@pytest.mark.parametrize(
    "type, simulate, versions_to_retain, expected_files_in_dir",
    [
        ("model", True, 3, 25),
        ("training", True, 2, 5),
        ("prediction", True, 1, 10),
        ("model", False, 3, 15),
        ("training", False, 2, 2),
        ("prediction", False, 1, 2),
    ],
)
@pytest.mark.order(after="test_1_init_testproject")
def test_cleanup(
    type: str, simulate: bool, versions_to_retain: int, expected_files_in_dir: int
):
    config_path = cleanup_dir / "cleanup_BEFL-2019.ini"
    config_overrules = [
        f"cleanup.simulate={simulate}",
        f"cleanup.{type}_versions_to_retain={versions_to_retain}",
    ]

    match type:
        case "model":
            cleanup.clean_models(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            models = os.listdir(cleanup_dir / "models")
            assert len(models) == expected_files_in_dir
        case "training":
            cleanup.clean_training_data_directories(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            training_dirs = os.listdir(cleanup_dir / "training")
            assert len(training_dirs) == expected_files_in_dir
        case "prediction":
            cleanup.clean_predictions(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            prediction_dirs = os.listdir(cleanup_dir / "output_vector")
            for prediction_dir in prediction_dirs:
                predictions = os.listdir(cleanup_dir / "output_vector" / prediction_dir)
                assert len(predictions) == expected_files_in_dir

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
    "type, simulate, versions_to_retain",
    [
        ("model", True, 3),
        ("training", True, 2),
        ("prediction", True, 1),
    ],
)
@pytest.mark.order(after="test_1_init_testproject")
def test_2_cleanup_simulate(type: str, simulate: bool, versions_to_retain: int):
    config_path = cleanup_dir / "cleanup_BEFL-2019.ini"
    config_overrules = [
        f"cleanup.simulate={simulate}",
        f"cleanup.{type}_versions_to_retain={versions_to_retain}",
    ]
    models_dir = cleanup_dir / "models"
    models_orig = os.listdir(models_dir)
    predictions_dir = cleanup_dir / "output_vector"
    prediction_dir = os.listdir(predictions_dir)[0]
    predictions_orig = os.listdir(predictions_dir / prediction_dir)
    training_dir = cleanup_dir / "training"
    training_orig = os.listdir(training_dir)

    match type:
        case "model":
            cleanup.clean_models(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            models = os.listdir(models_dir)
            assert len(models) == len(models_orig)
        case "training":
            cleanup.clean_training_data_directories(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            training_dirs = os.listdir(training_dir)
            assert len(training_dirs) == len(training_orig)
        case "prediction":
            cleanup.clean_predictions(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            prediction_dirs = os.listdir(predictions_dir)
            for prediction_dir in prediction_dirs:
                predictions = os.listdir(predictions_dir / prediction_dir)
                assert len(predictions) == len(predictions_orig)


@pytest.mark.parametrize(
    "type, simulate, versions_to_retain",
    [
        ("model", False, 3),
        ("training", False, 2),
        ("prediction", False, 1),
    ],
)
@pytest.mark.order(after="test_2_cleanup_simulate")
def test_3_cleanup(type: str, simulate: bool, versions_to_retain: int):
    config_path = cleanup_dir / "cleanup_BEFL-2019.ini"
    config_overrules = [
        f"cleanup.simulate={simulate}",
        f"cleanup.{type}_versions_to_retain={versions_to_retain}",
    ]
    models_dir = cleanup_dir / "models"
    models_orig = os.listdir(models_dir)
    predictions_dir = cleanup_dir / "output_vector"
    prediction_dir = os.listdir(predictions_dir)[0]
    predictions_orig = os.listdir(predictions_dir / prediction_dir)
    training_dir = cleanup_dir / "training"
    training_orig = os.listdir(training_dir)

    match type:
        case "model":
            cleanup.clean_models(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            models = os.listdir(models_dir)
            assert len(models) == versions_to_retain * len(models_orig) / 5
        case "training":
            cleanup.clean_training_data_directories(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            training_dirs = os.listdir(training_dir)
            assert len(training_dirs) == versions_to_retain * len(training_orig) / 5
        case "prediction":
            cleanup.clean_predictions(
                config_path=config_path / config_path,
                config_overrules=config_overrules,
            )
            prediction_dirs = os.listdir(predictions_dir)
            for prediction_dir in prediction_dirs:
                predictions = os.listdir(predictions_dir / prediction_dir)
                assert (
                    len(predictions) == versions_to_retain * len(predictions_orig) / 5
                )

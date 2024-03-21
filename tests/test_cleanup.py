from pathlib import Path
from typing import List

import pytest
from orthoseg import cleanup


# @pytest.mark.parametrize(
#     "config_path, config_overrules",
#     [
#         (
#             Path("X:/Monitoring/OrthoSeg/fields-arable/fields-arable.ini"),
#             ["predict.image_layer=BEFL-2021"],
#         )
#     ],
# )

orthoseg_path = Path("X:/Monitoring/OrthoSeg")


@pytest.mark.parametrize(
    "type, config_path, config_overrules",
    [
        ("models", "fields-arable/fields-arable.ini", []),
        ("trainingdata", "fields-arable/fields-arable.ini", []),
        ("predictions", "fields-arable/fields-arable.ini", []),
    ],
)
def test_cleanup(type: str, config_path: str, config_overrules: List[str]):
    # config_path = SampleProjectFootball.train_config_path
    match type:
        case "models":
            cleanup.clean_models(
                config_path=orthoseg_path / config_path,
                config_overrules=config_overrules,
            )
        case "trainingdata":
            cleanup.clean_training_data_directories(
                config_path=orthoseg_path / config_path,
                config_overrules=config_overrules,
            )
        case "predictions":
            cleanup.clean_predictions(
                config_path=orthoseg_path / config_path,
                config_overrules=config_overrules,
            )

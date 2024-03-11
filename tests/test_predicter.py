import os
from pathlib import Path
import pytest


from orthoseg.helpers import config_helper as conf
from orthoseg.lib import predicter
from orthoseg.model import model_factory as mf
from orthoseg.model import model_helper as mh


@pytest.mark.parametrize(
    "ini_file, force_model_traindata_id, no_images_ok",
    [
        ("X:/Monitoring/OrthoSeg/horsepastures/horsepastures.ini", 4, False),
        ("X:/Monitoring/OrthoSeg/horsepastures/horsepastures.ini", 4, True),
    ],
)
def test_predict_dir_test_dataset(
    ini_file: str, force_model_traindata_id: int, no_images_ok: bool
):
    # Init
    # Load the config and save in a bunch of global variables so it
    # is accessible everywhere
    config_path = Path(ini_file)
    conf.read_orthoseg_config(config_path)  # , overrules=config_overrules)

    # If the training data doesn't exist yet, create it
    train_label_infos = conf.get_train_label_infos()

    # Determine the projection of (the first) train layer... it will be used for all
    train_image_layer = train_label_infos[0].image_layer
    train_projection = conf.image_layers[train_image_layer]["projection"]

    # Determine classes
    classes = conf.train.getdict("classes")

    # Now create the train datasets (train, validation, test)
    training_dir = conf.dirs.getpath("training_dir") / f"{force_model_traindata_id:02d}"
    traindata_id = force_model_traindata_id

    testdata_dir = training_dir / "test"

    # Check if training is needed
    # Get hyper parameters from the config
    architectureparams = mh.ArchitectureParams(
        architecture=conf.model["architecture"],
        classes=list(classes),
        nb_channels=conf.model.getint("nb_channels"),
        architecture_id=conf.model.getint("architecture_id"),
        activation_function="softmax",
    )
    trainparams = mh.TrainParams(
        trainparams_id=conf.train.getint("trainparams_id"),
        image_augmentations=conf.train.getdict("image_augmentations"),
        mask_augmentations=conf.train.getdict("mask_augmentations"),
        class_weights=[classes[classname]["weight"] for classname in classes],
        batch_size=conf.train.getint("batch_size_fit"),
        optimizer=conf.train.get("optimizer"),
        optimizer_params=conf.train.getdict("optimizer_params"),
        loss_function=conf.train.get("loss_function"),
        monitor_metric=conf.train.get("monitor_metric"),
        monitor_metric_mode=conf.train.get("monitor_metric_mode"),
        save_format=conf.train.get("save_format"),
        save_best_only=conf.train.getboolean("save_best_only"),
        save_min_accuracy=conf.train.getfloat("save_min_accuracy"),
        nb_epoch=conf.train.getint("max_epoch"),
        nb_epoch_with_freeze=conf.train.getint("nb_epoch_with_freeze"),
        earlystop_patience=conf.train.getint("earlystop_patience"),
        earlystop_monitor_metric=conf.train.get("earlystop_monitor_metric"),
        earlystop_monitor_metric_mode=conf.train.get("earlystop_monitor_metric_mode"),
        log_tensorboard=conf.train.getboolean("log_tensorboard"),
        log_csv=conf.train.getboolean("log_csv"),
    )

    # Check if there exists already a model for this train dataset + hyperparameters
    model_dir = conf.dirs.getpath("model_dir")
    segment_subject = conf.general["segment_subject"]
    best_model_curr_train_version = mh.get_best_model(
        model_dir=model_dir,
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        architecture_id=architectureparams.architecture_id,
        trainparams_id=trainparams.trainparams_id,
    )

    # Train!!!
    min_probability = conf.predict.getfloat("min_probability")

    # Assert to evade typing warnings
    assert best_model_curr_train_version is not None

    model = mf.load_model(best_model_curr_train_version["filepath"], compile=False)

    # Prepare output subdir to be used for predictions
    predict_out_subdir, _ = os.path.splitext(best_model_curr_train_version["filename"])

    try:
        # Predict test dataset, if it exists
        if testdata_dir is not None and testdata_dir.exists():
            predicter.predict_dir(
                model=model,
                input_image_dir=testdata_dir / "image",
                output_image_dir=testdata_dir / predict_out_subdir,
                output_vector_path=None,
                projection_if_missing=train_projection,
                input_mask_dir=testdata_dir / "mask",
                batch_size=conf.train.getint("batch_size_predict"),
                evaluate_mode=True,
                classes=classes,
                min_probability=min_probability,
                cancel_filepath=conf.files.getpath("cancel_filepath"),
                max_prediction_errors=conf.predict.getint("max_prediction_errors"),
                no_images_ok=no_images_ok,
            )
    except Exception as excinfo:
        pytest.fail(f"Unexpected exception raised: {excinfo}")

"""Module with high-level operations to segment images."""

import logging
from pathlib import Path
from typing import Any

import keras
import pandas as pd
import tensorflow as tf

import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh
from orthoseg.lib.training_dataset import create_train_dataset

# Get a logger...
logger = logging.getLogger(__name__)


def train(
    traindata_dir: Path,
    validationdata_dir: Path,
    model_save_dir: Path,
    *,
    segment_subject: str,
    traindata_id: int,
    hyperparams: mh.HyperParams,
    weights_dir: Path | None = None,
    model_preload_filepath: Path | None = None,
    image_width: int = 512,
    image_height: int = 512,
    image_subdir: str = "image",
    mask_subdir: str = "mask",
    save_augmented_subdir: str | None = None,
):
    """Create/load a neural network and train it.

    Data from the train and validation directories specified for the training.

    The best models will be saved to model_save_dir. The filenames of the
    models will be constructed like this:
    {model_save_base_filename}_{combined_acc}_{train_acc}_{validation_acc}_{epoch}
        * combined_acc: average of train_acc and validation_acc
        * train_acc: the jaccard coëficient of train dataset for the model
        * validation_acc: the jaccard coëficient of validation dataset
    In the scripts, if the "best model" is mentioned, this is the one with the
    highest "combined_acc".

    Args:
        traindata_dir: dir where the train data is located
        validationdata_dir: dir where the validation data is located
        model_save_dir: dir where (intermediate) best models will be saved
        segment_subject: segment subject
        traindata_id: train data version
        hyperparams: the hyper parameters to use for the model
        weights_dir: directory where pretrained weights are cached and read from
        model_preload_filepath: filepath to the model to continue training on,
            or None if you want to start from scratch
        image_width: width the input images will be rescaled to for training
        image_height: height the input images will be rescaled to for training
        image_subdir: subdir where the images can be found in traindata_dir and
            validationdata_dir
        mask_subdir: subdir where the corresponding masks can be found in traindata_dir
            and validationdata_dir
        save_augmented_subdir: (str, optional): the subdirectory to save the augmented
            images to. If None, the images aren't saved. Defaults to None.
    """
    # First if there exists a model already for the traindata_id,...
    curr_best_model = mh.get_best_model(
        model_dir=model_save_dir,
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        architecture_id=hyperparams.architecture.architecture_id,
        trainparams_id=hyperparams.train.trainparams_id,
    )
    model_save_base_filename = mh.format_model_basefilename(
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        architecture_id=hyperparams.architecture.architecture_id,
        trainparams_id=hyperparams.train.trainparams_id,
    )

    # If there is an existing model and we want to continue training on that, recuperate
    # the epoch.
    start_epoch = 0
    csv_log_path = model_save_dir / f"{model_save_base_filename}_train_log.csv"
    model_json_path = model_save_dir / f"{model_save_base_filename}_model.json"
    hyperparams_path = model_save_dir / f"{model_save_base_filename}_hyperparams.json"

    if curr_best_model is not None:
        if not model_preload_filepath:
            message = (
                "Model exists but preload model file not specified: "
                f"{curr_best_model.filepath}"
            )
            logger.critical(message)
            raise ValueError(message)

        train_log_df = pd.read_csv(csv_log_path, sep=";", usecols=["epoch", "lr"])
        assert isinstance(train_log_df, pd.DataFrame)
        logger.debug(f"train_log csv contents:\n{train_log_df}")
        start_epoch = train_log_df["epoch"].max()
        hyperparams.train.optimizer_params["learning_rate"] = float(
            pd.to_numeric(train_log_df["lr"], downcast="float").min()
        )
    logger.info(
        f"start_epoch: {start_epoch}, learning_rate: "
        f"{hyperparams.train.optimizer_params['learning_rate']}"
    )

    if start_epoch == 0:
        # No existing model, so clean any existing files that might exist
        csv_log_path.unlink(missing_ok=True)
        model_json_path.unlink(missing_ok=True)
        hyperparams_path.unlink(missing_ok=True)

    # If no existing model provided, create it from scratch
    if not model_preload_filepath:
        # If the encode should be frozen for some epochs...
        if hyperparams.train.nb_epoch_with_freeze > 0:
            freeze = True
        else:
            freeze = False

        # Get the model we want to use
        model, model_preprocess_input = mf.get_model(
            architecture=hyperparams.architecture.architecture,
            nb_channels=hyperparams.architecture.nb_channels,
            nb_classes=len(hyperparams.architecture.classes),
            activation=hyperparams.architecture.activation_function,
            weights_type=hyperparams.train.weights_type,
            weights_dir=weights_dir,
            freeze=freeze,
        )

        # Save the model architecture to json
        model_save_dir.mkdir(parents=True, exist_ok=True)
        with model_json_path.open("w") as dst:
            dst.write(str(model.to_json()))

    else:
        # If a preload model is provided, load that if it exists...
        if not model_preload_filepath.exists():
            message = (
                f"Error: preload model file doesn't exist: {model_preload_filepath}"
            )
            logger.critical(message)
            raise RuntimeError(message)

        # Load the existing model
        # Remark: compiling during load crashes, so compile 'manually'
        logger.info(f"Load model from {model_preload_filepath}")
        model, model_preprocess_input = mf.load_model(
            model_preload_filepath, compile_model=False
        )

    # Now prepare the model for training
    nb_gpu = mh.get_number_gpus()

    # TODO: because of bug in tensorflow 1.14, multi GPU doesn't work (this way),
    # so always use standard model
    if nb_gpu <= 1:
        model_for_train = mf.compile_model(
            model=model,
            optimizer=hyperparams.train.optimizer,
            optimizer_params=hyperparams.train.optimizer_params,
            loss=hyperparams.train.loss_function,
            class_weights=hyperparams.train.class_weights,
        )
        logger.info(f"Train using single GPU or CPU, with nb_gpu: {nb_gpu}")
    else:
        # If multiple GPU's available, should create a multi-GPU model
        model_for_train = model
        logger.info(f"Train using all GPU's, with nb_gpu: {nb_gpu}")
        logger.warning("MULTI GPU TRAINING NOT TESTED BUT WILL BE TRIED ANYWAY")
        raise NotImplementedError("Multi GPU training not implemented yet")
        """
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model_for_train = mf.compile_model(
                model=model,
                optimizer=hyperparams.train.optimizer,
                optimizer_params=hyperparams.train.optimizer_params,
                loss=hyperparams.train.loss_function,
                class_weights=hyperparams.train.class_weights,
            )
        """

    # Prepare the data generators for training and validation that will also take care
    # of the augmentations.
    # Remarks:
    #   - fill_mode + cval are defined as they are so missing pixels after
    #     eg. rotation are filled with 0, and so the mask will take care that they are
    #     +- ignored.
    #   - if rescaling is included in the image augmentations, don't apply the standard
    #     model preprocess function, as this also includes rescaling.
    rescale_train = hyperparams.train.image_augmentations.get("rescale")
    if rescale_train is not None:
        model_preprocess_input = None

    train_dataset, train_dataset_size = create_train_dataset(
        input_data_dir=traindata_dir,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        image_augment_dict=hyperparams.train.image_augmentations,
        mask_augment_dict=hyperparams.train.mask_augmentations,
        model_preprocess_input=model_preprocess_input,
        batch_size=hyperparams.train.batch_size,
        target_size=(image_width, image_height),
        nb_classes=len(hyperparams.architecture.classes),
        save_to_subdir=save_augmented_subdir,
        shuffle=True,
        seed=2,
    )

    # For validation data:
    #   - don't apply augmentation except potentially rescaling
    #   - don't shuffle the validation dataset
    validation_augm = {"rescale": rescale_train} if rescale_train is not None else {}
    rescale_mask = hyperparams.train.mask_augmentations.get("rescale")
    validation_mask_augm = {"rescale": rescale_mask} if rescale_mask is not None else {}

    validation_dataset, validation_dataset_size = create_train_dataset(
        input_data_dir=validationdata_dir,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        image_augment_dict=validation_augm,
        mask_augment_dict=validation_mask_augm,
        model_preprocess_input=model_preprocess_input,
        batch_size=hyperparams.train.batch_size,
        target_size=(image_width, image_height),
        nb_classes=len(hyperparams.architecture.classes),
        save_to_subdir=save_augmented_subdir,
        shuffle=False,
        seed=3,
    )

    # Define some callbacks for the training
    train_callbacks: list[Any] = []
    # Reduce the learning rate if the loss doesn't improve anymore
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.2,
        patience=20,
        min_lr=1e-20,
        verbose=1,
    )
    train_callbacks.append(reduce_lr)

    # Custom callback that saves the best models using both train and
    # validation metric
    # Remark: the save of the model should be done on the standard model, not
    #         on the parallel_model, otherwise issues to use it afterwards
    if nb_gpu > 1:
        model_template_for_save = model
    else:
        model_template_for_save = None
    model_checkpoint_saver = mh.ModelCheckpointExt(
        model_save_dir=model_save_dir,
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        architecture_id=hyperparams.architecture.architecture_id,
        trainparams_id=hyperparams.train.trainparams_id,
        monitor_metric=hyperparams.train.monitor_metric,
        monitor_metric_mode=hyperparams.train.monitor_metric_mode,
        save_format=hyperparams.train.save_format,
        save_best_only=hyperparams.train.save_best_only,
        save_min_accuracy=hyperparams.train.save_min_accuracy,
        model_template_for_save=model_template_for_save,
    )
    train_callbacks.append(model_checkpoint_saver)

    # Callbacks for logging
    if hyperparams.train.log_tensorboard is True:
        tensorboard_log_dir = model_save_dir / (
            model_save_base_filename + "_tensorboard_log"
        )
        tensorboard_logger = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_log_dir)
        )
        train_callbacks.append(tensorboard_logger)
    if hyperparams.train.log_csv is True:
        csv_logger = mh.CSVLoggerExt(
            str(csv_log_path),
            append=True,
            separator=";",
            custom_metrics={"monitor_metric": hyperparams.train.monitor_metric},
        )
        train_callbacks.append(csv_logger)

    # Stop if no more improvement
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=hyperparams.train.earlystop_monitor_metric,
        patience=hyperparams.train.earlystop_patience,
        mode=hyperparams.train.earlystop_monitor_metric_mode,
        restore_best_weights=False,
    )
    train_callbacks.append(early_stopping)

    # Prepare the parameters to pass to fit...
    # Calculate the number of steps within an epoch
    train_steps_per_epoch = int(tf.data.experimental.cardinality(train_dataset).numpy())
    validation_steps_per_epoch = int(
        tf.data.experimental.cardinality(validation_dataset).numpy()
    )

    # Start training
    logger.info(
        f"Start training with batch_size: {hyperparams.train.batch_size}, "
        f"train_dataset_size: {train_dataset_size}, train_steps_per_epoch: "
        f"{train_steps_per_epoch}, validation_dataset_size: {validation_dataset_size}, "
        f"validation_steps_per_epoch: {validation_steps_per_epoch}"
    )

    logger.info(f"{hyperparams.toJSON()}")
    hyperparams_path.write_text(hyperparams.toJSON())

    try:
        if hyperparams.train.nb_epoch_with_freeze > 0:
            # First train only the top layers for a few epochs.
            logger.info(
                f"First train for {hyperparams.train.nb_epoch_with_freeze} epochs with "
                "frozen layers"
            )
            model_for_train.fit(
                train_dataset,
                steps_per_epoch=train_steps_per_epoch,
                epochs=hyperparams.train.nb_epoch_with_freeze,
                validation_data=validation_dataset,
                validation_steps=validation_steps_per_epoch,
                callbacks=train_callbacks,
                initial_epoch=start_epoch,
                verbose=2,
                shuffle=False,  # Shuffle is done by the dataset itself
            )
            mf.set_trainable(model=model_for_train, recompile=False)
            model_for_train = mf.compile_model(
                model=model_for_train,
                optimizer=hyperparams.train.optimizer,
                optimizer_params=hyperparams.train.optimizer_params,
                loss=hyperparams.train.loss_function,
                class_weights=hyperparams.train.class_weights,
            )

        # Train!
        model_for_train.fit(
            train_dataset,
            steps_per_epoch=train_steps_per_epoch,
            epochs=hyperparams.train.nb_epoch,
            validation_data=validation_dataset,
            validation_steps=validation_steps_per_epoch,
            callbacks=train_callbacks,
            initial_epoch=start_epoch,
            verbose=2,
            shuffle=False,  # Shuffle is done by the dataset itself
        )

        # Write some reporting
        train_report_path = model_save_dir / (model_save_base_filename + "_report.pdf")
        train_log_df = pd.read_csv(csv_log_path, sep=";")
        assert isinstance(train_log_df, pd.DataFrame)
        columns_to_keep = []
        for column in train_log_df.columns:
            if column.endswith(("accuracy", "f1-score")):
                columns_to_keep.append(column)

        train_log_vis_df = train_log_df[columns_to_keep]
        fig = train_log_vis_df.plot().get_figure()
        if fig is not None:
            fig.savefig(str(train_report_path))  # type: ignore[union-attr]

    finally:
        # Release the memory from the GPU...
        # from keras import backend as K
        # K.clear_session()
        keras.backend.clear_session()

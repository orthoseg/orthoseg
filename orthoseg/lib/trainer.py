# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images.
"""

import logging
import math
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras as kr

# import keras as kr

import pandas as pd
from PIL import Image

import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def train(
    traindata_dir: Path,
    validationdata_dir: Path,
    model_save_dir: Path,
    segment_subject: str,
    traindata_id: int,
    hyperparams: mh.HyperParams,
    model_preload_filepath: Optional[Path] = None,
    image_width: int = 512,
    image_height: int = 512,
    image_subdir: str = "image",
    mask_subdir: str = "mask",
    save_augmented_subdir: Optional[str] = None,
):
    """
    Create a new or load an existing neural network and train it using
    data from the train and validation directories specified.

    The best models will be saved to model_save_dir. The filenames of the
    models will be constructed like this:
    {model_save_base_filename}_{combined_acc}_{train_acc}_{validation_acc}_{epoch}
        * combined_acc: average of train_acc and validation_acc
        * train_acc: the jaccard coëficient of train dataset for the model
        * validation_acc: the jaccard coëficient of validation dataset
    In the scripts, if the "best model" is mentioned, this is the one with the
    highest "combined_acc".

    Args
        traindata_dir: dir where the train data is located
        validationdata_dir: dir where the validation data is located
        model_save_dir: dir where (intermediate) best models will be saved
        segment_subject (str): segment subject
        traindata_id (int): train data version
        hyperparams (mh.HyperParams): the hyper parameters to use for the model
        image_width: width the input images will be rescaled to for training
        image_height: height the input images will be rescaled to for training
        image_subdir: subdir where the images can be found in traindata_dir and
            validationdata_dir
        mask_subdir: subdir where the corresponding masks can be found in traindata_dir
            and validationdata_dir
        model_preload_filepath: filepath to the model to continue training on,
            or None if you want to start from scratch
    """
    # These are the augmentations that will be applied to the input training
    # images/masks.
    # Remark: fill_mode + cval are defined as they are so missing pixels after
    #    eg. rotation are filled with 0, and so the mask will take care that they are
    #    +- ignored.

    # Create the train generator
    train_gen = create_train_generator(
        input_data_dir=traindata_dir,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        image_augment_dict=hyperparams.train.image_augmentations,
        mask_augment_dict=hyperparams.train.mask_augmentations,
        batch_size=hyperparams.train.batch_size,
        target_size=(image_width, image_height),
        nb_classes=len(hyperparams.architecture.classes),
        save_to_subdir=save_augmented_subdir,
        seed=2,
    )

    # Create validation generator
    validation_augmentations = dict(
        rescale=hyperparams.train.image_augmentations["rescale"]
    )
    validation_mask_augmentations = dict(
        rescale=hyperparams.train.mask_augmentations["rescale"]
    )
    validation_gen = create_train_generator(
        input_data_dir=validationdata_dir,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        image_augment_dict=validation_augmentations,
        mask_augment_dict=validation_mask_augmentations,
        batch_size=hyperparams.train.batch_size,
        target_size=(image_width, image_height),
        nb_classes=len(hyperparams.architecture.classes),
        save_to_subdir=save_augmented_subdir,
        shuffle=False,
        seed=3,
    )

    # Get the max epoch number from the log file if it exists...
    start_epoch = 0
    model_save_base_filename = mh.format_model_basefilename(
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        architecture_id=hyperparams.architecture.architecture_id,
        trainparams_id=hyperparams.train.trainparams_id,
    )
    csv_log_filepath = model_save_dir / (model_save_base_filename + "_log.csv")
    if csv_log_filepath.exists() and os.path.getsize(csv_log_filepath) > 0:
        if not model_preload_filepath:
            message = (
                "log file exists but preload model file not specified: "
                f"{csv_log_filepath}"
            )
            logger.critical(message)
            raise Exception(message)

        train_log_df = pd.read_csv(csv_log_filepath, sep=";")
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

    # If no existing model provided, create it from scratch
    if not model_preload_filepath:
        # If the encode should be frozen for some epochs...
        if hyperparams.train.nb_epoch_with_freeze > 0:
            freeze = True
        else:
            freeze = False

        # Get the model we want to use
        model = mf.get_model(
            architecture=hyperparams.architecture.architecture,
            nb_channels=hyperparams.architecture.nb_channels,
            nb_classes=len(hyperparams.architecture.classes),
            activation=hyperparams.architecture.activation_function,
            freeze=freeze,
        )

        # Save the model architecture to json
        model_json_filepath = model_save_dir / f"{model_save_base_filename}_model.json"
        if not model_save_dir.exists():
            model_save_dir.mkdir(parents=True)
        if not model_json_filepath.exists():
            with model_json_filepath.open("w") as dst:
                dst.write(str(model.to_json()))
    else:
        # If a preload model is provided, load that if it exists...
        if not model_preload_filepath.exists():
            message = (
                f"Error: preload model file doesn't exist: {model_preload_filepath}"
            )
            logger.critical(message)
            raise Exception(message)

        # Load the existing model
        # Remark: compiling during load crashes, so compile 'manually'
        logger.info(f"Load model from {model_preload_filepath}")
        model = mf.load_model(model_preload_filepath, compile=False)

    # Now prepare the model for training
    nb_gpu = len(tf.config.experimental.list_physical_devices("GPU"))

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
        logger.warn("MULTI GPU TRAINING NOT TESTED BUT WILL BE TRIED ANYWAY")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model_for_train = mf.compile_model(
                model=model,
                optimizer=hyperparams.train.optimizer,
                optimizer_params=hyperparams.train.optimizer_params,
                loss=hyperparams.train.loss_function,
                class_weights=hyperparams.train.class_weights,
            )

    # Define some callbacks for the training
    train_callbacks = []
    # Reduce the learning rate if the loss doesn't improve anymore
    reduce_lr = kr.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.2, patience=20, min_lr=1e-20, verbose=True
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
        tensorboard_logger = kr.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir))
        train_callbacks.append(tensorboard_logger)
    if hyperparams.train.log_csv is True:
        csv_logger = kr.callbacks.CSVLogger(
            str(csv_log_filepath), append=True, separator=";"
        )
        train_callbacks.append(csv_logger)

    # Stop if no more improvement
    early_stopping = kr.callbacks.EarlyStopping(
        monitor=hyperparams.train.earlystop_monitor_metric,
        patience=hyperparams.train.earlystop_patience,
        mode=hyperparams.train.earlystop_monitor_metric_mode,
        restore_best_weights=False,
    )
    train_callbacks.append(early_stopping)

    # Prepare the parameters to pass to fit...
    # Supported filetypes to train/validate on
    input_ext = [".tif", ".jpg", ".png"]

    # Calculate the size of the input datasets
    # train_dataset_size = len(glob.glob(f"{traindata_dir}/{image_subdir}/*.*"))
    train_dataset_size = 0
    for input_ext_cur in input_ext:
        traindata_image_dir = traindata_dir / image_subdir
        train_dataset_size += len(list(traindata_image_dir.rglob("*" + input_ext_cur)))
    validation_dataset_size = 0
    for input_ext_cur in input_ext:
        validationdata_image_dir = validationdata_dir / image_subdir
        validation_dataset_size += len(
            list(validationdata_image_dir.rglob("*" + input_ext_cur))
        )

    # Calculate the number of steps within an epoch
    # Remark: number of steps per epoch should be at least 1, even if nb samples
    # < batch size...
    train_steps_per_epoch = math.ceil(train_dataset_size / hyperparams.train.batch_size)
    validation_steps_per_epoch = math.ceil(
        validation_dataset_size / hyperparams.train.batch_size
    )

    # Start training
    logger.info(
        f"Start training with batch_size: {hyperparams.train.batch_size}, "
        f"train_dataset_size: {train_dataset_size}, train_steps_per_epoch: "
        f"{train_steps_per_epoch}, validation_dataset_size: {validation_dataset_size}, "
        f"validation_steps_per_epoch: {validation_steps_per_epoch}"
    )

    logger.info(f"{hyperparams.toJSON()}")
    hyperparams_filepath = (
        model_save_dir / f"{model_save_base_filename}_hyperparams.json"
    )
    hyperparams_filepath.write_text(hyperparams.toJSON())

    try:
        # If the encoder should be frozen for the first epochs, do so
        if hyperparams.train.nb_epoch_with_freeze > 0:
            logger.info(
                f"First train for {hyperparams.train.nb_epoch_with_freeze} epochs with "
                "frozen layers"
            )
            model_for_train.fit(
                train_gen,
                steps_per_epoch=train_steps_per_epoch,
                epochs=hyperparams.train.nb_epoch_with_freeze,
                validation_data=validation_gen,
                # Number of items in validation/batch_size
                validation_steps=validation_steps_per_epoch,
                callbacks=train_callbacks,
                initial_epoch=start_epoch,
                verbose=2,  # type: ignore
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
            train_gen,
            steps_per_epoch=train_steps_per_epoch,
            epochs=hyperparams.train.nb_epoch,
            validation_data=validation_gen,
            # Number of items in validation/batch_size
            validation_steps=validation_steps_per_epoch,
            callbacks=train_callbacks,
            initial_epoch=start_epoch,
            verbose=2,  # type: ignore
        )

        # Write some reporting
        train_report_path = model_save_dir / (model_save_base_filename + "_report.pdf")
        train_log_df = pd.read_csv(csv_log_filepath, sep=";")
        assert isinstance(train_log_df, pd.DataFrame)
        columns_to_keep = []
        for column in train_log_df.columns:
            if column.endswith("accuracy") or column.endswith("f1-score"):
                columns_to_keep.append(column)

        train_log_vis_df = train_log_df[columns_to_keep]
        train_log_vis_df.plot().get_figure().savefig(train_report_path)

    finally:
        # Release the memory from the GPU...
        # from keras import backend as K
        # K.clear_session()
        kr.backend.clear_session()


def create_train_generator(
    input_data_dir: Path,
    image_subdir: str,
    mask_subdir: str,
    image_augment_dict: dict,
    mask_augment_dict: dict,
    batch_size: int = 32,
    image_color_mode: str = "rgb",
    mask_color_mode: str = "grayscale",
    save_to_subdir: Optional[str] = None,
    image_save_prefix: str = "image",
    mask_save_prefix: str = "mask",
    nb_classes: int = 1,
    target_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: int = 1,
):
    """
    Creates a generator to generate and augment train images. The augmentations
    specified in aug_dict will be applied. For the augmentations that can be
    specified in aug_dict look at the documentation of
    keras.preprocessing.image.ImageDataGenerator

    For more info about the other parameters, check keras flow_from_directory.

    Remarks: * use the same seed for image_datagen and mask_datagen to ensure
               the transformation for image and mask is the same
             * set save_to_dir = "your path" to check results of the generator
    """
    # Init
    # Do some checks on the augmentations specified, as it is easy to
    # introduce illogical values
    if (
        image_augment_dict is not None
        and mask_augment_dict is None
        or image_augment_dict is not None
        and mask_augment_dict is None
    ):
        logger.warn(
            "Only augmentations specified for either image or mask: "
            f"image_augment_dict: {image_augment_dict}, mask_augment_dict: "
            f"{mask_augment_dict}"
        )

    # Checks that involve comparing augmentations between the image and the mask
    if image_augment_dict is not None and mask_augment_dict is not None:
        # If an augmentation is specified for image, it should be specified
        # for the mask as well and the other way around to evade issues
        for augmentation in image_augment_dict:
            if augmentation not in mask_augment_dict:
                raise Exception(
                    f"{augmentation} in image_augment_dict but not in mask_augment_dict"
                )
        for augmentation in mask_augment_dict:
            if augmentation not in image_augment_dict:
                raise Exception(
                    f"{augmentation} in mask_augment_dict but not in image_augment_dict"
                )

        # Check if the brightness range has valid values
        image_brightness_range = image_augment_dict.get("brightness_range")
        mask_brightness_range = mask_augment_dict.get("brightness_range")
        if image_brightness_range is None and mask_brightness_range is not None:
            # If brightness_range None for image, it should be None for the mask as well
            raise Exception(
                "augmentation brightness_range is None (null) in image_augment_dict "
                f"but isn't in mask_augment_dict: {mask_brightness_range}"
            )
        elif image_brightness_range is not None and mask_brightness_range is None:
            # If brightness_range None for mask, it should be None for the image as well
            raise Exception(
                "augmentation brightness_range is None (null) in mask_augment_dict but "
                f"isn't in image_augment_dict: {image_brightness_range}"
            )
        elif image_brightness_range is not None and mask_brightness_range is not None:
            # The brightness_range values should be >= 0
            if image_brightness_range[0] < 0 or image_brightness_range[1] < 0:
                raise Exception(
                    "augmentation brightness_range values should be > 0: value 1.0 = "
                    "don't do anything, 0 = black"
                )
            # brightness_range is applied to the image, it should be [1,1] for the mask
            if mask_brightness_range[0] != 1.0 or mask_brightness_range[1] != 1.0:
                raise Exception(
                    "augmentation brightness_range is specified on the image: then the "
                    f"mask should get range [1, 1], not {mask_brightness_range}"
                )

    # Some checks specific for the mask
    if mask_augment_dict is not None:
        # Check cval value
        if "cval" in mask_augment_dict and mask_augment_dict["cval"] != 0:
            logger.warn(
                "cval typically should be 0 for the mask, even if it is different for "
                "the image, as the cval of the mask refers to these locations being of "
                f"class 'background'. It is: {mask_augment_dict['cval']}"
            )

        # If there are more than two classes, the mask will have integers as values
        # to code the different masks in, and one hot-encoding will be applied to
        # it, so it should not be rescaled!!!
        if (
            nb_classes > 2
            and "rescale" in mask_augment_dict
            and mask_augment_dict["rescale"] != 1
        ):
            raise Exception(
                f"With nb_classes > 2 ({nb_classes}), the mask should have a rescale "
                f"value of 1, not {mask_augment_dict['rescale']}"
            )

    # Now create the generators
    # Create the image generators with the augment info

    # TODO: brightness_range is buggy on 2021-08-11, with keras preprocessing
    # version 1.1, so is implemented here in a hacky way!
    image_augment_dict_temp = {
        key: value
        for (key, value) in image_augment_dict.items()
        if key != "brightness_range"
    }
    mask_augment_dict_temp = {
        key: value
        for (key, value) in mask_augment_dict.items()
        if key != "brightness_range"
    }
    image_datagen = kr.preprocessing.image.ImageDataGenerator(**image_augment_dict_temp)
    mask_datagen = kr.preprocessing.image.ImageDataGenerator(**mask_augment_dict_temp)

    # Format save_to_dir
    # Remark: flow_from_directory doesn't support Path, so supply str immediately as
    # well, otherwise, if str(Path) is used later on, it becomes 'None' instead of None!
    save_to_dir = None
    if save_to_subdir is not None:
        save_to_dir = input_data_dir / save_to_subdir
        if not save_to_dir.exists():
            save_to_dir.mkdir(parents=True)

    image_generator = image_datagen.flow_from_directory(
        directory=str(input_data_dir),
        classes=[image_subdir],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=None,
        save_prefix=image_save_prefix,
        shuffle=shuffle,
        seed=seed,
    )

    mask_generator = mask_datagen.flow_from_directory(
        directory=str(input_data_dir),
        classes=[mask_subdir],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=None,
        save_prefix=mask_save_prefix,
        shuffle=shuffle,
        seed=seed,
    )

    train_generator = zip(image_generator, mask_generator)

    for batch_id, (image, mask) in enumerate(train_generator):
        # Cast to arrays to evade type errors
        image = np.array(image)
        mask = np.array(mask)

        # TODO: brightness_range is buggy on 2021-08-11, with keras preprocessing
        # version 1.1, so is implemented here in a hacky way!
        if (
            "brightness_range" in image_augment_dict
            and image_augment_dict["brightness_range"] is not None
            and (
                image_augment_dict["brightness_range"][0] != 1
                or image_augment_dict["brightness_range"][1] != 1
            )
        ):

            # Random brightness shift to apply to all images in batch
            brightness_shift = np.random.uniform(
                image_augment_dict["brightness_range"][0],
                image_augment_dict["brightness_range"][1],
            )
            image = image * brightness_shift

        # One-hot encode mask if multiple classes
        if nb_classes > 1:
            mask = kr.utils.to_categorical(mask, nb_classes)

        # Because the default save_to_dir option doesn't support saving the
        # augmented masks in seperate files per class, implement this here.
        if save_to_dir is not None:
            # Save mask for every class seperately
            # Get the number of images in this batch + the nb classes
            mask_shape = mask.shape
            nb_images = mask_shape[0]
            nb_classes = mask_shape[3]

            # Loop through images in this batch
            for image_id in range(nb_images):
                # Slice the next image from the array
                image_to_save = image[image_id, :, :, :]

                # Reverse the rescale if there is one
                if (
                    image_augment_dict is not None
                    and "rescale" in image_augment_dict
                    and image_augment_dict["rescale"] != 1
                ):
                    image_to_save = image_to_save / image_augment_dict["rescale"]

                # Now convert to uint8 image and save!
                # , image_color_mode
                colormode = None
                if image_color_mode.upper() == "RGB":
                    colormode = "RGB"
                im = Image.fromarray(image_to_save.astype(np.uint8), mode=colormode)
                image_dir = save_to_dir / f"{batch_id:0>4}"
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{image_id}_image.jpg"
                im.save(image_path)

                # Loop through the masks for each class
                for channel_id in range(nb_classes):
                    mask_to_save = mask[image_id, :, :, channel_id]
                    mask_dir = save_to_dir / f"{batch_id:0>4}"
                    mask_path = mask_dir / f"{image_id}_mask_{channel_id}.png"
                    im = Image.fromarray((mask_to_save * 255).astype(np.uint8))
                    im.save(mask_path)

        yield (image, mask)


# If the script is ran directly...
if __name__ == "__main__":
    raise Exception("Not implemented")

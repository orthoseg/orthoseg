"""Module with high-level operations to segment images."""

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import preprocessing

import orthoseg.model.model_factory as mf
import orthoseg.model.model_helper as mh

# Get a logger...
logger = logging.getLogger(__name__)


def train(
    traindata_dir: Path,
    validationdata_dir: Path,
    model_save_dir: Path,
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
        segment_subject (str): segment subject
        traindata_id (int): train data version
        hyperparams (mh.HyperParams): the hyper parameters to use for the model
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
                f"{curr_best_model['model_filepath']}"
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
            weights=hyperparams.train.weights_type,
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

    train_gen = create_train_generator(
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
        seed=2,
    )

    # For validation data, don't apply augmentation except potentially rescaling.
    validation_augm = {"rescale": rescale_train} if rescale_train is not None else {}
    rescale_mask = hyperparams.train.mask_augmentations.get("rescale")
    validation_mask_augm = {"rescale": rescale_mask} if rescale_mask is not None else {}

    validation_gen = create_train_generator(
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
        csv_logger = keras.callbacks.CSVLogger(
            str(csv_log_path), append=True, separator=";"
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
    hyperparams_path.write_text(hyperparams.toJSON())

    try:
        if hyperparams.train.nb_epoch_with_freeze > 0:
            # First train only the top layers for a few epochs.
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
                verbose=2,
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
            verbose=2,
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


def create_train_generator(
    input_data_dir: Path,
    image_subdir: str,
    mask_subdir: str,
    image_augment_dict: dict,
    mask_augment_dict: dict,
    model_preprocess_input: Callable | None,
    batch_size: int = 32,
    image_color_mode: str = "rgb",
    mask_color_mode: str = "grayscale",
    save_to_subdir: str | None = None,
    image_save_prefix: str = "image",
    mask_save_prefix: str = "mask",
    nb_classes: int = 1,
    target_size: tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: int = 1,
):
    """Creates a generator to generate and augment train images.

    The augmentations
    specified in aug_dict will be applied. For the augmentations that can be
    specified in aug_dict look at the documentation of
    keras.preprocessing.image.ImageDataGenerator

    For more info about the other parameters, check keras flow_from_directory.

    Remarks: * use the same seed for image_datagen and mask_datagen to ensure
               the transformation for image and mask is the same
             * set save_to_dir = "your path" to check results of the generator
             * it is possible to convert the generator to a Dataset using
               tf.data.Dataset.from_generator, but this halves training speed.
                    return tf.data.Dataset.from_generator(
                        lambda: train_gen,
                        output_types=(tf.float32, tf.float32),
                        output_shapes=(
                            [None, target_size[0], target_size[1], 3],
                            [None, target_size[0], target_size[1], nb_classes],
                        ),
                    )
    """
    # Init
    # Do some checks on the augmentations specified, as it is easy to
    # introduce illogical values
    if (image_augment_dict is not None and mask_augment_dict is None) or (
        image_augment_dict is not None and mask_augment_dict is None
    ):
        logger.warning(
            "Only augmentations specified for either image or mask: "
            f"image_augment_dict: {image_augment_dict}, mask_augment_dict: "
            f"{mask_augment_dict}"
        )

    # Checks that involve comparing augmentations between the image and the mask
    if image_augment_dict is not None and mask_augment_dict is not None:
        # If an augmentation is specified for image, it should be specified
        # for the mask as well and the other way around to avoid issues
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
                raise ValueError(
                    "augmentation brightness_range is specified on the image: then the "
                    f"mask should get range [1, 1], not {mask_brightness_range}"
                )

    # Some checks specific for the mask
    if mask_augment_dict is not None:
        # Check cval value
        if "cval" in mask_augment_dict and mask_augment_dict["cval"] != 0:
            logger.warning(
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
    image_datagen = preprocessing.image.ImageDataGenerator(**image_augment_dict_temp)
    mask_datagen = preprocessing.image.ImageDataGenerator(**mask_augment_dict_temp)

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

    train_generator = zip(image_generator, mask_generator, strict=True)

    for batch_id, (image, mask) in enumerate(train_generator):
        # Cast to arrays to avoid type errors
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
            brightness_shift = np.random.uniform(  # noqa: NPY002
                image_augment_dict["brightness_range"][0],
                image_augment_dict["brightness_range"][1],
            )
            image = image * brightness_shift

        # One-hot encode mask if multiple classes
        if nb_classes > 1:
            mask = keras.utils.to_categorical(mask, nb_classes)

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

        # Apply the model preprocess function if there is one
        if model_preprocess_input is not None:
            image = model_preprocess_input(image)

        yield (image, mask)

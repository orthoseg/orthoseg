"""Module with helper functions regarding (keras) models."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from keras import callbacks

# Get a logger...
logger = logging.getLogger(__name__)


class ArchitectureParams:
    """Parameters regarding the neural network architecture to use."""

    def __init__(
        self,
        architecture: str,
        classes: list | None = None,
        nb_channels: int = 3,
        activation_function: str = "softmax",
        architecture_id: int = 0,
    ):
        """Class containing the hyper parameters needed to create the model.

        Args:
            architecture (str): the model architecture to use.
            classes (list, optional): list of classes that will be detected by
                the model. Default to None, and then 2 generic classes are
                supposed.
            nb_channels (int, optional): Number of channels of the images.
                Defaults to 3.
            activation_function (str, optional): activation function to use.
                Defaults to softmax.
            architecture_id (int, optional): id of the architecture.
                Defaults to 0.
        """
        self.architecture = architecture
        if classes is not None:
            self.classes = classes
        else:
            self.classes = ["background", "segmentsubject"]
        self.nb_channels = nb_channels
        self.activation_function = activation_function
        self.architecture_id = architecture_id

    def toJSON(self) -> str:
        """Serialize to JSON string.

        Returns:
            str: JSON string
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class TrainParams:
    """Hyperparameter on how to train the neural network."""

    def __init__(
        self,
        image_augmentations: dict,
        mask_augmentations: dict,
        trainparams_id: int = 0,
        class_weights: list | None = None,
        batch_size: int = 4,
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss_function: str | None = None,
        monitor_metric: str | None = None,
        monitor_metric_mode: str = "auto",
        save_format: str = "h5",
        save_best_only: bool = True,
        save_min_accuracy: float = 0.95,
        nb_epoch: int = 1000,
        nb_epoch_with_freeze: int = 100,
        earlystop_patience: int = 100,
        earlystop_monitor_metric: str | None = None,
        earlystop_monitor_metric_mode: str = "auto",
        log_tensorboard: bool = False,
        log_csv: bool = True,
    ):
        """Class containing the hyper parameters needed to perform a training.

        Args:
            image_augmentations (dict): The augmentations to use on the input image
                during training.
            mask_augmentations (dict): The augmentations to use on the input mask
                during training.
            architecture_id (int, optional): id of the architecture. Defaults to 0.
            trainparams_id (int, optional): id of the hyperparams. Defaults to 0.
            class_weights (list, optional): [description]. Defaults to None.
            batch_size (int, optional): batch size to use while training. This must be
                choosen depending on the neural network architecture
                and available memory on you GPU. Defaults to 4.
            optimizer (str, optional): Optimizer to use for training.
                Defaults to 'adam'.
            optimizer_params (dict, optional): Optimizer params to use.
                Defaults to { 'learning_rate': 0.0001 }.
            loss_function (str, optional): [description]. Defaults to None.
            monitor_metric (str, optional): Metric to monitor. If not specified
                the loss function will drive the metric. Defaults to None.
            monitor_metric_mode (str, optional): Mode of the metric to monitor.
                Defaults to 'auto'.
            save_format (str, optional): [description]. Defaults to 'h5'.
            save_best_only (bool, optional): [description]. Defaults to True.
            save_min_accuracy (float, optional): minimum accuracy to save a model.
                Defaults to 0.95.
            nb_epoch (int, optional): maximum number of epochs to train.
                Defaults to 1000.
            nb_epoch_with_freeze (int, optional): number epochs to train with
                    part of the layers frozen. Defaults to 20.
            earlystop_patience (int, optional): [description]. Defaults to 100.
            earlystop_monitor_metric (str, optional): [description]. Defaults to None.
            earlystop_monitor_metric_mode (str, optional): Mode to monitor the
                metric: 'max' if the metric should be as high as possible,
                'min' if it should be low. Defaults to 'auto'.
            log_tensorboard (bool, optional): True to activate tensorboard
                logging. Defaults to False.
            log_csv (bool, optional): True to activate logging to a csv.
                Defaults to True

        Raises:
            Exception: [description]
        """
        # Validate if the augmentations specified are OK
        _validate_augmentations(image_augmentations, mask_augmentations)

        self.trainparams_id = trainparams_id
        self.image_augmentations = image_augmentations
        self.mask_augmentations = mask_augmentations

        self.class_weights = class_weights
        self.batch_size = batch_size

        self.optimizer = optimizer
        if optimizer_params is None:
            # Best set to 0.0001 to start (1e-3 is not ok)
            self.optimizer_params = {"learning_rate": 0.0001}
        else:
            self.optimizer_params = optimizer_params

        if self.class_weights is not None:
            self.loss_function = "weighted_categorical_crossentropy"
        else:
            self.loss_function = "categorical_crossentropy"

        # Properties to choose the best model
        if monitor_metric is not None:
            self.monitor_metric = monitor_metric
        elif self.loss_function in (
            "weighted_categorical_crossentropy",
            "categorical_crossentropy",
        ):
            self.monitor_metric = "categorical_accuracy"
        self.monitor_metric_mode = monitor_metric_mode

        self.save_format = save_format
        self.save_best_only = save_best_only
        self.save_min_accuracy = save_min_accuracy
        self.nb_epoch = nb_epoch
        self.nb_epoch_with_freeze = nb_epoch_with_freeze

        # Properties to stop the training
        self.earlystop_patience = earlystop_patience
        if earlystop_monitor_metric is not None:
            self.earlystop_monitor_metric = earlystop_monitor_metric
        else:
            self.earlystop_monitor_metric = self.monitor_metric
        self.earlystop_monitor_metric_mode = earlystop_monitor_metric_mode

        # Properties regarding logging
        self.log_tensorboard = log_tensorboard
        self.log_csv = log_csv

    def toJSON(self) -> str:
        """Serializes object to a JSON string.

        Returns:
            str: JSON string
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class HyperParams:
    """Groups all hyper parameters to use for creating a neural network model."""

    def __init__(
        self,
        architecture: ArchitectureParams | None = None,
        train: TrainParams | None = None,
        path: Path | None = None,
    ):
        """Class to store the hyperparameters to use for the machine learning algorythm.

        Args:
            architecture (ArchitectureParams): the fixed parameters that define the
                architecture of the neural network. When training a network, it is
                possible to reuse the weights of another network if these parameters
                are the same.
            train (TrainParams): these are the parameters that can be changed with each
                training.
            path (Optional[Path]): file path to read the hyper parameters from.
        """
        self.fileversion = 1.1
        if architecture is not None:
            self.architecture = architecture
        if train is not None:
            self.train = train
        if path is not None:
            with open(path) as jsonfile:
                jsonstr = jsonfile.read()
                data = json.loads(jsonstr)

                # Read file version if it is present
                if "fileversion" in data:
                    self.fileversion = data["fileversion"]

                # Now parse the real data. For backwards compatibility, remove
                # 'nb_classes' parameter if it exists
                if "nb_classes" in data["architecture"]:
                    del data["architecture"]["nb_classes"]

                self.architecture = ArchitectureParams(**data["architecture"])
                self.train = TrainParams(**data["train"])

    def toJSON(self) -> str:
        """Serialize the data to a JSON string.

        Returns:
            str: the object serialized as JSON string.
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def _validate_augmentations(
    image_augmentations: dict[str, Any], mask_augmentations: dict[str, Any]
):
    errors = []

    # The same augmentations should be specified for images and masks
    if not list(image_augmentations) == list(mask_augmentations):
        raise ValueError(
            "the same augmentations should be specified in the same order for images "
            "and masks to avoid possible issues with random augmentation factors "
            "getting mixed up. Values can be different for some augmentations."
        )

    # These augmentations should be the same for the mask and the image
    should_be_same = [
        "rotation_range",
        "width_shift_range",
        "height_shift_range",
        "zoom_range",
    ]
    for key in should_be_same:
        error_message = (
            f"when {key} is used, it should be in image_augmentations and "
            "mask_augmentations with the same value"
        )
        if key in image_augmentations:
            if key not in mask_augmentations:
                # Key is only in image augmentation -> error
                errors.append(error_message)
            elif mask_augmentations[key] != image_augmentations[key]:
                # Augmentation parameter is not the same -> error
                errors.append(error_message)
        elif key in mask_augmentations:
            # Key is only in mask augmentation -> error
            errors.append(error_message)

    # Check augmentations that should have specific values for mask_augmentations
    mask_specifics = [
        ("fill_mode", "constant", True),
        ("cval", 0, True),
        ("rescale", 1, True),
        ("brightness_range", [1.0, 1.0], False),
    ]
    for key, value, mandatory in mask_specifics:
        if key not in mask_augmentations:
            if mandatory:
                errors.append(
                    f"{key} is a mandatory augmentation that should be {value} for "
                    "mask_augmentations"
                )
        else:
            if mask_augmentations[key] != value:
                errors.append(
                    f"{key} for mask_augmentations should be {value}, not "
                    f"{mask_augmentations[key]}"
                )

    if len(errors) > 0:
        raise ValueError(f"issues found in augmentation parameters: {errors}")


def format_model_basefilename(
    segment_subject: str,
    traindata_id: int,
    architecture_id: int = 0,
    trainparams_id: int = 0,
) -> str:
    """Format the parameters into a model_filename.

    Args;
        segment_subject: the segment subject
        traindata_id: the id of the data used to train the model
        architecture_id: the id of the model architecture used
        trainparams_id: the id of the hyper parameters used to train the model
    """
    # Format file name
    filename = f"{segment_subject}_{traindata_id:02d}"

    # If an architecture id and/or hyperparams id is specified, add those as well
    if architecture_id > 0 or trainparams_id > 0:
        filename = f"{filename}.{architecture_id}.{trainparams_id}"
    return filename


def format_model_filename(
    segment_subject: str,
    traindata_id: int,
    architecture_id: int,
    trainparams_id: int,
    monitor_metric_accuracy: float,
    epoch: int,
    save_format: str,
) -> str:
    """Format the parameters into a model_filename.

    Args:
        segment_subject: the segment subject
        traindata_id: the version of the data used to train the model
        architecture_id: the id of the model architecture used
        trainparams_id: the version of the hyper parameters used to train
        monitor_metric_accuracy: the monitor metric accuracy
        epoch: the epoch during training that reached these model weights
        save_format (str): the format to save in:
            * keras format: 'h5'
            * tensorflow savedmodel: 'tf'
    """
    # Format file name
    filename = format_model_basefilename(
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        architecture_id=architecture_id,
        trainparams_id=trainparams_id,
    )
    filename += f"_{monitor_metric_accuracy:.5f}_{epoch}"

    # Add suffix
    if save_format == "tf":
        filename += "_tf"
    else:
        filename += ".hdf5"

    return filename


def parse_model_filename(filepath: Path) -> dict | None:
    """Parse a model_filename to a dict with the properties of the model.

    These are the properties:
        * segment_subject: the segment subject
        * traindata_id: the version of the data used to train the model
        * monitor_metric_accuracy: the monitored metric accuracy
        * trainparams_id: the version of the hyper parameters used to train
        * epoch: the epoch during training that reached these model weights
        * save_format (str): the format to save in:
            * keras format: 'h5'
            * tensorflow savedmodel: 'tf'

    Args:
        filepath: the filepath to the model file
    """
    # Prepare filepath to extract info
    if filepath.is_dir():
        # If it is a dir, it should end on _tf
        if not filepath.name.endswith("_tf"):
            logger.warning(
                f"Not a valid path for a model, dir needs to end on _tf: {filepath}"
            )
            return None
        save_format = "tf"
        filename = filepath.name
    else:
        filename = filepath.stem
        if filepath.suffix in (".h5", ".hdf5"):
            save_format = "h5"
        else:
            logger.warning(f"Model file should have .h5 of .hdf5 as suffix: {filepath}")
            return None

    # Now extract the fields...
    param_values = filename.split("_")
    if len(param_values) < 3:
        logger.warning(
            f"Model file name nok, split('_') must result in >= 2 fields: {filepath}"
        )

    segment_subject = param_values[0]

    model_info = param_values[1]
    model_info_values = model_info.split(".")
    traindata_id = int(model_info_values[0])
    architecture_id = 0
    trainparams_id = 0
    if len(model_info_values) > 1:
        architecture_id = int(model_info_values[1])
        if len(model_info_values) > 2:
            trainparams_id = int(model_info_values[2])

    monitor_metric_accuracy = float(param_values[2])
    epoch = int(param_values[3])

    basefilename = format_model_basefilename(
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        architecture_id=architecture_id,
        trainparams_id=trainparams_id,
    )

    return {
        "filepath": filepath,
        "filename": filename,
        "basefilename": basefilename,
        "segment_subject": segment_subject,
        "traindata_id": traindata_id,
        "architecture_id": architecture_id,
        "trainparams_id": trainparams_id,
        "monitor_metric_accuracy": monitor_metric_accuracy,
        "epoch": epoch,
        "save_format": save_format,
    }


def get_models(
    model_dir: Path,
    segment_subject: str | None = None,
    traindata_id: int | None = None,
    architecture_id: int | None = None,
    trainparams_id: int | None = None,
) -> list[dict]:
    """Return the list of models in the model_dir passed.

    It is returned as a dataframe with the columns as returned in parse_model_filename.

    Args:
        model_dir (Path): dir containing the models
        segment_subject (str, optional): only models with this the segment subject
        traindata_id (int, optional): only models with this traindata version
        architecture_id (int, optional): only models with this this architecture_id
        trainparams_id (int, optional): only models with this hyperparams version
    """
    # List models
    model_paths: list[Path] = []
    model_paths.extend(model_dir.glob("*.hdf5"))
    model_paths.extend(model_dir.glob("*.h5"))
    model_paths.extend(model_dir.glob("*_tf"))

    # Loop through all models and extract necessary info...
    model_info_list = []
    for model_path in model_paths:
        model_info = parse_model_filename(model_path)
        if model_info is not None:
            model_info_list.append(model_info)

    # Filter, if filters provided
    if len(model_info_list) > 0:
        if segment_subject is not None:
            model_info_list = [
                model_info
                for model_info in model_info_list
                if model_info["segment_subject"] == segment_subject
            ]
        if traindata_id is not None:
            model_info_list = [
                model_info
                for model_info in model_info_list
                if model_info["traindata_id"] == traindata_id
            ]
        if trainparams_id is not None:
            model_info_list = [
                model_info
                for model_info in model_info_list
                if model_info["trainparams_id"] == trainparams_id
            ]
        if architecture_id is not None:
            model_info_list = [
                model_info
                for model_info in model_info_list
                if model_info["architecture_id"] == architecture_id
            ]

    return model_info_list


def get_best_model(
    model_dir: Path,
    segment_subject: str | None = None,
    traindata_id: int | None = None,
    architecture_id: int | None = None,
    trainparams_id: int | None = None,
) -> dict | None:
    """Get the properties of the model with the highest combined accuracy.

    Only models with the highest traindata version in the dir are considered.

    Remark: regardless of the monitor function used when training, the accuracies
    are always better if higher!

    Args:
        model_dir: dir containing the models
        segment_subject (str, optional): only models with this the segment subject
        traindata_id (int, optional): only models with this train data id
        architecture_id (int, optional): only models with this the architecture_id
        trainparams_id (int, optional): only models with this hyperparams id

    Returns:
        A dictionary with the info of the best model as returned by
        parse_model_filename, or None if no model was found
    """
    # Get list of existing models for this train dataset
    model_info_list = get_models(
        model_dir=model_dir,
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        architecture_id=architecture_id,
        trainparams_id=trainparams_id,
    )

    # If nothing found, return None
    if len(model_info_list) == 0:
        return None

    # If no traindata_id provided, find highest traindata id
    max_traindata_id = -1
    for model_info in model_info_list:
        max_traindata_id = max(model_info["traindata_id"], max_traindata_id)

    # Get the list of newest model
    newest_model_info_list = [
        model_info
        for model_info in model_info_list
        if model_info["traindata_id"] == max_traindata_id
    ]

    # If only one result, return it
    if len(newest_model_info_list) == 1:
        return newest_model_info_list[0]
    else:
        max_monitor_metric_accuracy = -1
        max_monitor_metric_accuracy_idx = -1
        for model_info_idx, model_info in enumerate(model_info_list):
            if model_info["monitor_metric_accuracy"] > max_monitor_metric_accuracy:
                max_monitor_metric_accuracy = model_info["monitor_metric_accuracy"]
                max_monitor_metric_accuracy_idx = model_info_idx

        return model_info_list[max_monitor_metric_accuracy_idx]


class ModelCheckpointExt(callbacks.Callback):
    """Class to checkpoint a model while training it with extended options.

    Args:
        callbacks (_type_): _description_
    """

    def __init__(
        self,
        model_save_dir: Path,
        segment_subject: str,
        traindata_id: int,
        architecture_id: int,
        trainparams_id: int,
        monitor_metric: str,
        monitor_metric_mode: str,
        save_format: str = "h5",
        save_best_only: bool = False,
        save_min_accuracy: float = 0.90,
        save_min_accuracy_ignored_epoch: int = 100,
        save_weights_only: bool = False,
        model_template_for_save=None,
        verbose: bool = True,
        only_report: bool = False,
    ):
        """Constructor.

        Args:
            model_save_dir (Path): [description]
            segment_subject (str): segment subject
            traindata_id (int): train data id
            architecture_id (int): model architecture id
            trainparams_id (int): id of the hyper parameters used
            monitor_metric (str): The metric to monitor for accuracy
            monitor_metric_mode (str): use 'min' if the accuracy metrics
                should be  as low as possible, 'max' if a higher values is
                better.
            save_format (str, optional): The format to save in:
                'h5' (keras format) or 'tf' (tensorflow savedmodel).
                Defaults to 'tf'
            save_best_only (bool, optional): only keep the best model.
                Defaults to True.
            save_min_accuracy (float, optional): minimum accuracy to be
                reached to save model. Defaults to 0.9.
            save_min_accuracy_ignored_epoch (int, optional): at this epoch the
                save_min_accuracy is ignored, to make sure there is a model
                saved, even if the minimum accuracy is never reached.
            save_weights_only: optional: only save weights
            model_template_for_save: optional, if using multi-GPU training,
                pass the original model here to use this as template for saving
            verbose (bool, optional): [description]. Defaults to True.
            only_report (bool, optional): [description]. Defaults to False.
        """
        monitor_metric_mode_values = ["min", "max"]
        if monitor_metric_mode not in monitor_metric_mode_values:
            raise Exception(
                f"Invalid value for mode: {monitor_metric_mode}, should be one of "
                f"{monitor_metric_mode_values}"
            )
        save_format_values = ("h5", "tf")
        if save_format not in save_format_values:
            raise Exception(
                f"Invalid value for save_format: {save_format}, should be one of "
                f"{save_format_values}"
            )

        self.model_save_dir = model_save_dir
        self.segment_subject = segment_subject
        self.traindata_id = traindata_id
        self.architecture_id = architecture_id
        self.trainparams_id = trainparams_id
        self.monitor_metric = monitor_metric
        self.monitor_metric_mode = monitor_metric_mode
        self.save_format = save_format
        self.save_best_only = save_best_only
        self.save_min_accuracy = save_min_accuracy
        self.save_min_accuracy_ignored_epoch = save_min_accuracy_ignored_epoch
        self.save_weights_only = save_weights_only
        self.model_template_for_save = model_template_for_save
        self.verbose = verbose
        self.only_report = only_report

    def on_epoch_end(self, epoch, logs={}):
        """on_epoch_end method.

        Args:
            epoch (_type_): _description_
            logs (dict, optional): _description_. Defaults to {}.
        """
        logger.debug(f"Start in callback on_epoch_begin, logs contains: {logs}")

        # First determine the values of the monitor metric for train and validation
        # If the monitor_metric doesn't contain placeholder, it is easy
        if "{" not in self.monitor_metric:
            new_model_monitor_value = logs.get(self.monitor_metric)
        else:
            # There are placeholders, so we need some more logic
            monitor_metric_formatted = self.monitor_metric.format(**logs)
            new_model_monitor_value = eval(monitor_metric_formatted, {}, {})

        # Now we can save and clean models
        save_and_clean_models(
            model_save_dir=self.model_save_dir,
            segment_subject=self.segment_subject,
            traindata_id=self.traindata_id,
            architecture_id=self.architecture_id,
            trainparams_id=self.trainparams_id,
            monitor_metric_mode=self.monitor_metric_mode,
            new_model=self.model,
            new_model_monitor_value=new_model_monitor_value,
            new_model_epoch=epoch,
            save_format=self.save_format,
            save_best_only=self.save_best_only,
            save_min_accuracy=self.save_min_accuracy,
            save_min_accuracy_ignored_epoch=self.save_min_accuracy_ignored_epoch,
            save_weights_only=self.save_weights_only,
            model_template_for_save=self.model_template_for_save,
            verbose=self.verbose,
            only_report=self.only_report,
        )


def save_and_clean_models(
    model_save_dir: Path,
    segment_subject: str,
    traindata_id: int,
    architecture_id: int,
    trainparams_id: int,
    monitor_metric_mode: str,
    new_model=None,
    new_model_monitor_value: float | None = None,
    new_model_epoch: int | None = None,
    save_format: str = "h5",
    save_best_only: bool = False,
    save_min_accuracy: float = 0.9,
    save_min_accuracy_ignored_epoch: int = 100,
    save_weights_only: bool = False,
    model_template_for_save=None,
    verbose: bool = True,
    debug: bool = False,
    only_report: bool = False,
):
    """Save the new model.

    The model is only saved if it is good enough... and existing models are cleaned up
    if they are worse than the new or other existing models.

    Args:
        model_save_dir (Path): dir containing the models
        segment_subject (str): segment subject
        traindata_id (int): train data id
        architecture_id (int): model architecture id
        trainparams_id (int): id of the train params
        monitor_metric_mode (MetricMode): use 'min' if the monitored
            metrics should be as low as possible, 'max' if a higher values
            is better.
        new_model (optional): the keras model object that will be saved
        new_model_monitor_value (float, optional): the monitored metric value
        new_model_epoch (int, optional): the epoch in the training
        save_format (SaveFormat, optional): The format to save in:
            * h5: keras format (= default)
            * tf: tensorflow savedmodel
        save_best_only (bool, optional): only keep the best model
        save_min_accuracy (float, optional): minimum accuracy to save the
            model. Defaults to 0.9.
        save_min_accuracy_ignored_epoch (int, optional): at this epoch the
            save_min_accuracy is ignored, to make sure there is a model saved,
            even if the minimum accuracy is never reached.
        save_weights_only (bool, optional): only save weights
        model_template_for_save (optional): if using multi-GPU training, pass
            the original model here to use this as template for saving
        verbose (bool, optional): report the best model after save and cleanup
        debug (bool, optional): write debug logging
        only_report (bool, optional): only report which models would be cleaned up
    """
    # TODO: add option to specify minimum accuracy/iou score before saving to speed up,
    # because saving takes quite some time!
    # Check validaty of input
    monitor_metric_mode_values = ["min", "max"]
    if monitor_metric_mode not in monitor_metric_mode_values:
        raise Exception(
            f"Invalid value for mode: {monitor_metric_mode}, should be one of "
            f"{monitor_metric_mode_values}"
        )
    save_format_values = ("h5", "tf")
    if save_format not in save_format_values:
        raise Exception(
            f"Invalid value for save_format: {save_format}, should be one of "
            f"{save_format_values}"
        )

    # Get a list of all existing models
    model_info_list = get_models(
        model_dir=model_save_dir,
        segment_subject=segment_subject,
        traindata_id=traindata_id,
        trainparams_id=trainparams_id,
    )

    # If there is a new model passed as param, add it to the list
    new_model_path = None
    new_model_monitor_accuracy = None
    if new_model is not None:
        if new_model_monitor_value is None or new_model_epoch is None:
            raise Exception(
                "If new_model is not None, new_model_monitor_... parameters cannot be "
                f"None either???, new_model_monitor_value: {new_model_monitor_value}, "
                f"new_model_epoch: {new_model_epoch}"
            )

        # Build save filepath
        # Remark: accuracy values should always be as high as possible, so
        # recalculate values if monitor_metric_mode is 'min'
        if monitor_metric_mode == "max":
            new_model_monitor_accuracy = new_model_monitor_value
        else:
            new_model_monitor_accuracy = 1 - new_model_monitor_value

        new_model_filename = format_model_filename(
            segment_subject=segment_subject,
            traindata_id=traindata_id,
            architecture_id=architecture_id,
            trainparams_id=trainparams_id,
            monitor_metric_accuracy=new_model_monitor_accuracy,
            epoch=new_model_epoch,
            save_format=save_format,
        )
        new_model_path = Path(model_save_dir) / new_model_filename

        # Append model to the retrieved models...
        model_info_list.append(
            {
                "filepath": str(new_model_path),
                "filename": new_model_filename,
                "segment_subject": segment_subject,
                "traindata_id": traindata_id,
                "architecture_id": architecture_id,
                "trainparams_id": trainparams_id,
                "monitor_metric_accuracy": new_model_monitor_accuracy,
                "epoch": new_model_epoch,
                "save_format": save_format,
            }
        )

    # Loop through all existing models
    # Remark: the list is sorted descending before iterating it, this way new
    # modelss are saved bevore deleting the previous best one(s)
    model_info_df = pd.DataFrame(model_info_list)
    model_info_sorted_df = model_info_df.sort_values(
        by="monitor_metric_accuracy", ascending=False
    )
    for model_info in model_info_sorted_df.itertuples(index=False):
        # If only the best needs to be kept, check only on monitor_metric_accuracy...
        keep_model = True
        better_ones_df = None
        if save_best_only:
            better_ones_df = model_info_df[
                (model_info_df.filepath != model_info.filepath)
                & (
                    model_info_df.monitor_metric_accuracy
                    >= model_info.monitor_metric_accuracy
                )
            ]
            if len(better_ones_df) > 0:
                keep_model = False

        # If model is (relatively) ok, keep it
        if keep_model is True:
            logger.debug(f"KEEP {model_info.filename}")

            # If it is the new model that needs to be kept, keep it or save to disk
            if (
                new_model_path is not None
                and new_model is not None
                and new_model_epoch is not None
                and only_report is not True
                and model_info.filepath == str(new_model_path)
                and not new_model_path.exists()
            ):
                assert isinstance(model_info.monitor_metric_accuracy, float)
                if (
                    new_model_epoch > save_min_accuracy_ignored_epoch
                    or model_info.monitor_metric_accuracy > save_min_accuracy
                ):
                    logger.debug("Save model start")
                    if save_weights_only:
                        if model_template_for_save is not None:
                            model_template_for_save.save_weights(
                                str(new_model_path), save_format=save_format
                            )
                        else:
                            new_model.save_weights(
                                str(new_model_path), save_format=save_format
                            )
                    else:
                        if model_template_for_save is not None:
                            model_template_for_save.save(
                                str(new_model_path), save_format=save_format
                            )
                        else:
                            new_model.save(str(new_model_path), save_format=save_format)
                    logger.debug("Save model ready")
                else:
                    print(
                        "New model is best, but accuracy < save_min_accuracy: "
                        f"{new_model_monitor_accuracy} < {save_min_accuracy}"
                    )
        else:
            # Bad model... can be removed (or not saved)
            assert isinstance(model_info.filepath, str | Path)
            if only_report is True:
                logger.debug(f"DELETE {model_info.filename}")
            elif Path(model_info.filepath).exists() is True:
                logger.debug(f"DELETE {model_info.filename}")
                if Path(model_info.filepath).is_dir() is True:
                    shutil.rmtree(model_info.filepath)
                else:
                    Path(model_info.filepath).unlink()

            if debug is True and better_ones_df is not None:
                print(f"Better one(s) found for{model_info.filename}:")
                for better_one in better_ones_df.itertuples(index=False):
                    print(f"  {better_one.filename}")

    if verbose is True or debug is True:
        best_model = get_best_model(
            model_dir=model_save_dir,
            segment_subject=segment_subject,
            traindata_id=traindata_id,
            architecture_id=architecture_id,
            trainparams_id=trainparams_id,
        )
        if best_model is not None:
            logger.info(
                f"Current best model for {segment_subject}_{traindata_id}: "
                f"monitor_metric_accuracy: {best_model['monitor_metric_accuracy']}, "
                f"epoch: {best_model['epoch']}"
            )

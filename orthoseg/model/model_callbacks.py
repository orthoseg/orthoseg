"""Keras callbacks used during model training."""

import logging
from pathlib import Path

from orthoseg._compat import get_keras_backend
from orthoseg._keras import keras
from orthoseg.model.model_helper import save_and_clean_models

logger = logging.getLogger(__name__)


class ModelCheckpointExt(keras.callbacks.Callback):
    """Checkpoint a model while training it with extended options."""

    def __init__(
        self,
        model_save_dir: Path,
        segment_subject: str,
        *,
        traindata_id: int,
        monitor_metric: str,
        monitor_metric_mode: str,
        model_config_name: str = "",
        save_format: str = "keras",
        save_best_only: bool = False,
        save_min_accuracy: float = 0.90,
        save_min_accuracy_ignored_epoch: int = 100,
        save_weights_only: bool = False,
        model_template_for_save=None,
        verbose: bool = True,
        only_report: bool = False,
    ):
        """Initialize the callback."""
        monitor_metric_mode_values = ["min", "max"]
        if monitor_metric_mode not in monitor_metric_mode_values:
            raise ValueError(
                f"Invalid value for mode: {monitor_metric_mode}, should be one of "
                f"{monitor_metric_mode_values}"
            )
        save_format_values = ("keras", "h5", "tf")
        if save_format not in save_format_values:
            raise ValueError(
                f"Invalid value for save_format: {save_format}, should be one of "
                f"{save_format_values}"
            )
        if save_format == "tf" and get_keras_backend(keras) != "tensorflow":
            raise ValueError(
                "TensorFlow SavedModel output requires the TensorFlow Keras backend."
            )

        super().__init__()
        self.model_save_dir = model_save_dir
        self.segment_subject = segment_subject
        self.traindata_id = traindata_id
        self.model_config_name = model_config_name
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

    def on_epoch_end(self, epoch, logs=None):
        """Save the current model when it meets the configured criteria."""
        if logs is None:
            logs = {}
        logger.debug(f"Start in callback on_epoch_begin, logs contains: {logs}")

        if "{" not in self.monitor_metric:
            new_model_monitor_value = logs.get(self.monitor_metric)
        else:
            monitor_metric_formatted = self.monitor_metric.format(**logs)
            new_model_monitor_value = eval(monitor_metric_formatted, {}, {})

        save_and_clean_models(
            model_save_dir=self.model_save_dir,
            segment_subject=self.segment_subject,
            traindata_id=self.traindata_id,
            model_config_name=self.model_config_name,
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


class CSVLoggerExt(keras.callbacks.CSVLogger):
    """CSVLogger that adds metrics calculated from formulas."""

    def __init__(
        self,
        filepath: str | Path,
        separator: str = ",",
        append: bool = False,
        *,
        custom_metrics: dict[str, str] | None = None,
    ):
        """Initialize the logger with optional derived metric formulas."""
        super().__init__(filepath, separator=separator, append=append)
        self.custom_metrics = custom_metrics or {}

    def on_epoch_end(self, epoch, logs=None):
        """Calculate custom metrics and write this epoch through CSVLogger."""
        logs_with_custom_metrics = dict(logs or {})
        for metric_name, formula in self.custom_metrics.items():
            if "{" not in formula:
                logs_with_custom_metrics[metric_name] = logs_with_custom_metrics.get(
                    formula
                )
            else:
                formatted_formula = formula.format(**logs_with_custom_metrics)
                logs_with_custom_metrics[metric_name] = eval(
                    formatted_formula, {"__builtins__": {}}, {}
                )

        super().on_epoch_end(epoch, logs_with_custom_metrics)

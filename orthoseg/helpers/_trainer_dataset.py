"""Streaming Keras input sequence for paired training images and masks."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from orthoseg._compat import get_keras_backend
from orthoseg._keras import keras

from ._trainer_dataset_common import (
    _decode_with_pil,
    _has_spatial_augmentation,
    _list_image_mask_pairs,
    _save_augmented_batch,
    _validate_dataset_augmentations,  # noqa: F401
    _validate_train_dataset_options,
    _zoom_bounds,
)

logger = logging.getLogger(__name__)


_TRAINING_DATA_WORKERS = 4
_TRAINING_DATA_MAX_QUEUE_SIZE = 2


def create_train_dataset(
    input_data_dir: Path,
    image_subdir: str,
    mask_subdir: str,
    image_augment_dict: dict[str, Any],
    mask_augment_dict: dict[str, Any],
    *,
    model_preprocess_input: Callable | None,
    batch_size: int = 32,
    image_color_mode: str = "rgb",
    mask_color_mode: str = "grayscale",
    save_to_subdir: str | None = None,
    nb_classes: int = 1,
    target_size: tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: int = 1,
) -> tuple[Any, int]:
    """Create a streaming dataset for paired training images and masks.

    TensorFlow's parallel ``tf.data`` pipeline is used when TensorFlow is the active
    Keras backend. Other backends use the NumPy/Pillow Keras sequence.
    """
    if get_keras_backend(keras) == "tensorflow":
        from . import _trainer_dataset_tf  # noqa: PLC0415

        return _trainer_dataset_tf.create_train_dataset(
            input_data_dir=input_data_dir,
            image_subdir=image_subdir,
            mask_subdir=mask_subdir,
            image_augment_dict=image_augment_dict,
            mask_augment_dict=mask_augment_dict,
            model_preprocess_input=model_preprocess_input,
            batch_size=batch_size,
            image_color_mode=image_color_mode,
            mask_color_mode=mask_color_mode,
            save_to_subdir=save_to_subdir,
            nb_classes=nb_classes,
            target_size=target_size,
            shuffle=shuffle,
            seed=seed,
        )

    image_augment_dict, mask_augment_dict = _validate_train_dataset_options(
        image_augment_dict,
        mask_augment_dict,
        nb_classes=nb_classes,
        pipeline_name="training",
        logger=logger,
    )

    pairs = _list_image_mask_pairs(input_data_dir, image_subdir, mask_subdir)
    save_to_dir = None
    if save_to_subdir is not None:
        save_to_dir = input_data_dir / save_to_subdir
        save_to_dir.mkdir(parents=True, exist_ok=True)

    dataset = _TrainDatasetSequence(
        pairs=pairs,
        image_augmentations=image_augment_dict,
        mask_augmentations=mask_augment_dict,
        model_preprocess_input=model_preprocess_input,
        batch_size=batch_size,
        image_color_mode=image_color_mode,
        mask_color_mode=mask_color_mode,
        save_to_dir=save_to_dir,
        nb_classes=nb_classes,
        target_size=target_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset, len(pairs)


class _TrainDatasetSequence(keras.utils.Sequence):
    def __init__(
        self,
        *,
        pairs: list[tuple[Path, Path]],
        image_augmentations: dict[str, Any],
        mask_augmentations: dict[str, Any],
        model_preprocess_input: Callable | None,
        batch_size: int,
        image_color_mode: str,
        mask_color_mode: str,
        save_to_dir: Path | None,
        nb_classes: int,
        target_size: tuple[int, int],
        shuffle: bool,
        seed: int,
        workers: int = _TRAINING_DATA_WORKERS,
        max_queue_size: int = _TRAINING_DATA_MAX_QUEUE_SIZE,
    ) -> None:
        if int(keras.__version__.split(".", maxsplit=1)[0]) >= 3:
            super().__init__(
                workers=workers,
                use_multiprocessing=False,
                max_queue_size=max_queue_size,
            )
        else:
            super().__init__()
        self.pairs = pairs
        self.image_augmentations = image_augmentations
        self.mask_augmentations = mask_augmentations
        self.model_preprocess_input = model_preprocess_input
        self.batch_size = batch_size
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        self.save_to_dir = save_to_dir
        self.nb_classes = nb_classes
        self.target_size = target_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = np.arange(len(pairs))
        self.random = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self) -> int:
        return (len(self.pairs) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(f"Batch index out of range: {batch_index}")

        start = batch_index * self.batch_size
        batch_indices = self.indices[start : start + self.batch_size]
        image_batch: list[np.ndarray] = []
        mask_batch: list[np.ndarray] = []
        for sample_offset, pair_index in enumerate(batch_indices, start=start):
            image_path, mask_path = self.pairs[int(pair_index)]
            image, mask = _load_pair(
                sample_offset,
                (image_path, mask_path),
                image_color_mode=self.image_color_mode,
                mask_color_mode=self.mask_color_mode,
                target_size=self.target_size,
                image_augmentations=self.image_augmentations,
                mask_augmentations=self.mask_augmentations,
                seed=self.seed,
            )
            image_batch.append(image)
            mask_batch.append(mask)

        return _prepare_batch(
            batch_index,
            (np.stack(image_batch), np.stack(mask_batch)),
            image_brightness=self.image_augmentations.get("brightness_range"),
            seed=self.seed,
            nb_classes=self.nb_classes,
            save_to_dir=self.save_to_dir,
            image_rescale=self.image_augmentations.get("rescale"),
            model_preprocess_input=self.model_preprocess_input,
        )

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.random.shuffle(self.indices)


def _load_pair(
    index: int,
    paths: tuple[str | Path, str | Path],
    *,
    image_color_mode: str,
    mask_color_mode: str,
    target_size: tuple[int, int],
    image_augmentations: dict[str, Any],
    mask_augmentations: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    image = _decode_and_resize(
        paths[0], color_mode=image_color_mode, target_size=target_size
    )
    mask = _decode_and_resize(
        paths[1], color_mode=mask_color_mode, target_size=target_size
    )
    if _has_spatial_augmentation(image_augmentations):
        transform = _make_projective_transform(
            image_augmentations,
            seed=seed,
            index=index,
            height=image.shape[0],
            width=image.shape[1],
        )
        image = _apply_projective_transform(
            image, transform, image_augmentations, is_mask=False
        )
        mask = _apply_projective_transform(
            mask, transform, mask_augmentations, is_mask=True
        )

    image_rescale = image_augmentations.get("rescale")
    if image_rescale is not None:
        image *= np.float32(image_rescale)
    mask_rescale = mask_augmentations.get("rescale")
    if mask_rescale is not None:
        mask *= np.float32(mask_rescale)
    return image, mask


def _prepare_batch(
    batch_index: int,
    batch: tuple[np.ndarray, np.ndarray],
    *,
    image_brightness: list[float] | tuple[float, float] | None,
    seed: int,
    nb_classes: int,
    save_to_dir: Path | None,
    image_rescale: float | None,
    model_preprocess_input: Callable | None,
) -> tuple[np.ndarray, np.ndarray]:
    image, mask = batch
    if image_brightness is not None and (
        image_brightness[0] != 1 or image_brightness[1] != 1
    ):
        brightness = _stateless_uniform(
            seed,
            batch_index,
            100,
            float(image_brightness[0]),
            float(image_brightness[1]),
        )
        image *= brightness

    if nb_classes > 1:
        mask_classes = np.squeeze(mask, axis=-1).astype(np.int32)
        valid_mask_classes = (mask_classes >= 0) & (mask_classes < nb_classes)
        mask = np.eye(nb_classes, dtype=np.float32)[
            np.clip(mask_classes, 0, nb_classes - 1)
        ]
        mask *= valid_mask_classes[..., np.newaxis]

    if save_to_dir is not None:
        _save_augmented_batch(
            image,
            mask,
            np.asarray(batch_index, dtype=np.int64),
            save_to_dir=save_to_dir,
            image_rescale=image_rescale,
        )

    if model_preprocess_input is not None:
        image = model_preprocess_input(image)
    return image, mask


def _decode_and_resize(
    path: str | Path,
    *,
    color_mode: str,
    target_size: tuple[int, int],
) -> np.ndarray:
    if color_mode.lower() not in {"grayscale", "rgb", "rgba"}:
        raise ValueError(f"Unsupported color mode: {color_mode}")
    decoded = _decode_with_pil(path, color_mode.lower())
    source_height, source_width = decoded.shape[:2]
    target_height, target_width = target_size
    row_indices = np.floor(
        (np.arange(target_height, dtype=np.float32) + 0.5)
        * source_height
        / target_height
    ).astype(np.intp)
    column_indices = np.floor(
        (np.arange(target_width, dtype=np.float32) + 0.5) * source_width / target_width
    ).astype(np.intp)
    row_indices = np.clip(row_indices, 0, source_height - 1)
    column_indices = np.clip(column_indices, 0, source_width - 1)
    resized = decoded[row_indices[:, np.newaxis], column_indices[np.newaxis, :], :]
    return resized.astype(np.float32, copy=True)


def _stateless_uniform(
    seed: int,
    index: int,
    stream: int,
    minval: float,
    maxval: float,
) -> float:
    if minval == maxval:
        return minval
    random = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed) & 0xFFFFFFFF, (int(index) + stream) & 0xFFFFFFFF]
        )
    )
    return float(random.uniform(minval, maxval))


def _sample_range(
    value: Any,
    *,
    seed: int,
    index: int,
    stream: int,
    dimension: int | float | None = None,
) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a two-value augmentation range, got {value}")
        low, high = float(value[0]), float(value[1])
        is_fraction = dimension is not None and all(
            isinstance(item, float) for item in value
        )
    else:
        magnitude = abs(float(value))
        low, high = -magnitude, magnitude
        is_fraction = dimension is not None and isinstance(value, float)

    sample = _stateless_uniform(seed, index, stream, low, high)
    if is_fraction and dimension is not None:
        sample *= float(dimension)
    return sample


def _make_projective_transform(
    augmentations: dict[str, Any],
    *,
    seed: int,
    index: int,
    height: int,
    width: int,
) -> np.ndarray:
    angle = _sample_range(
        augmentations.get("rotation_range"), seed=seed, index=index, stream=1
    ) * (3.141592653589793 / 180.0)
    shear = _sample_range(
        augmentations.get("shear_range"), seed=seed, index=index, stream=2
    ) * (3.141592653589793 / 180.0)
    shift_x = _sample_range(
        augmentations.get("width_shift_range"),
        seed=seed,
        index=index,
        stream=3,
        dimension=width,
    )
    shift_y = _sample_range(
        augmentations.get("height_shift_range"),
        seed=seed,
        index=index,
        stream=4,
        dimension=height,
    )
    zoom_min, zoom_max = _zoom_bounds(augmentations.get("zoom_range"))
    zoom_x = _stateless_uniform(seed, index, 5, zoom_min, zoom_max)
    zoom_y = _stateless_uniform(seed, index, 6, zoom_min, zoom_max)
    flip_x = (
        -1.0
        if augmentations.get("horizontal_flip")
        and _stateless_uniform(seed, index, 7, 0.0, 1.0) < 0.5
        else 1.0
    )
    flip_y = (
        -1.0
        if augmentations.get("vertical_flip")
        and _stateless_uniform(seed, index, 8, 0.0, 1.0) < 0.5
        else 1.0
    )

    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    shear_matrix = np.array(
        [[1.0, -np.sin(shear)], [0.0, np.cos(shear)]],
    )
    zoom_matrix = np.diag([zoom_x, zoom_y])
    flip_matrix = np.diag([flip_x, flip_y])
    linear = rotation_matrix @ shear_matrix @ zoom_matrix @ flip_matrix

    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    offset_x = center_x - linear[0, 0] * center_x - linear[0, 1] * center_y + shift_x
    offset_y = center_y - linear[1, 0] * center_x - linear[1, 1] * center_y + shift_y
    return np.asarray(
        [
            linear[0, 0],
            linear[0, 1],
            offset_x,
            linear[1, 0],
            linear[1, 1],
            offset_y,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _apply_projective_transform(
    image: np.ndarray,
    transform: np.ndarray,
    augmentations: dict[str, Any],
    *,
    is_mask: bool,
) -> np.ndarray:
    interpolation_order = int(augmentations.get("interpolation_order", 1))
    if interpolation_order not in {0, 1}:
        raise ValueError("interpolation_order must be 0 or 1 for the training pipeline")
    fill_mode = str(augmentations.get("fill_mode", "nearest")).upper()
    if fill_mode not in {"CONSTANT", "REFLECT", "WRAP", "NEAREST"}:
        raise ValueError(f"Unsupported augmentation fill_mode: {fill_mode}")

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {image.shape}")
    coefficients = np.asarray(transform, dtype=np.float32).reshape(-1)
    if coefficients.size != 8:
        raise ValueError("Projective transform must contain 8 coefficients")

    height, width = image.shape[:2]
    output_y, output_x = np.indices((height, width), dtype=np.float32)
    denominator = coefficients[6] * output_x + coefficients[7] * output_y + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        source_x = (
            coefficients[0] * output_x + coefficients[1] * output_y + coefficients[2]
        ) / denominator
        source_y = (
            coefficients[3] * output_x + coefficients[4] * output_y + coefficients[5]
        ) / denominator
    finite_coordinates = np.isfinite(source_x) & np.isfinite(source_y)
    source_x = np.where(finite_coordinates, source_x, 0.0)
    source_y = np.where(finite_coordinates, source_y, 0.0)

    def map_indices(indices: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        if fill_mode == "CONSTANT":
            valid = (indices >= 0) & (indices < size)
            return np.clip(indices, 0, size - 1), valid
        if fill_mode == "NEAREST":
            return np.clip(indices, 0, size - 1), np.ones(indices.shape, dtype=bool)
        if fill_mode == "WRAP":
            return np.mod(indices, size), np.ones(indices.shape, dtype=bool)
        if size == 1:
            return np.zeros(indices.shape, dtype=np.int64), np.ones(
                indices.shape, dtype=bool
            )
        period = 2 * size
        reflected = np.mod(indices, period)
        reflected = np.where(reflected < size, reflected, period - 1 - reflected)
        return reflected, np.ones(indices.shape, dtype=bool)

    fill_value = np.float32(augmentations.get("cval", 0))

    def sample_pixels(x_indices: np.ndarray, y_indices: np.ndarray) -> np.ndarray:
        mapped_x, valid_x = map_indices(x_indices, width)
        mapped_y, valid_y = map_indices(y_indices, height)
        samples = image[mapped_y, mapped_x]
        if fill_mode == "CONSTANT":
            valid = valid_x & valid_y & finite_coordinates
            samples = np.where(valid[..., np.newaxis], samples, fill_value)
        return samples

    if is_mask or interpolation_order == 0:
        if fill_mode == "NEAREST":
            nearest_x = np.floor(source_x + 0.5).astype(np.int64)
            nearest_y = np.floor(source_y + 0.5).astype(np.int64)
        else:
            nearest_x = np.where(
                source_x < 0, np.ceil(source_x - 0.5), np.floor(source_x + 0.5)
            ).astype(np.int64)
            nearest_y = np.where(
                source_y < 0, np.ceil(source_y - 0.5), np.floor(source_y + 0.5)
            ).astype(np.int64)
        return sample_pixels(nearest_x, nearest_y).astype(np.float32, copy=False)

    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    weight_x = (source_x - x0)[..., np.newaxis]
    weight_y = (source_y - y0)[..., np.newaxis]
    top = sample_pixels(x0, y0) * (1.0 - weight_x) + sample_pixels(x1, y0) * weight_x
    bottom = sample_pixels(x0, y1) * (1.0 - weight_x) + sample_pixels(x1, y1) * weight_x
    return (top * (1.0 - weight_y) + bottom * weight_y).astype(np.float32)

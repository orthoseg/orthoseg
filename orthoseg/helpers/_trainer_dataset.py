"""Streaming Keras input sequence for paired training images and masks."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import keras
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


_TRAIN_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
}
_AUGMENTATION_KEYS = {
    "brightness_range",
    "cval",
    "fill_mode",
    "height_shift_range",
    "horizontal_flip",
    "interpolation_order",
    "rescale",
    "rotation_range",
    "shear_range",
    "vertical_flip",
    "width_shift_range",
    "zoom_range",
}
_GEOMETRIC_AUGMENTATION_KEYS = (
    "rotation_range",
    "width_shift_range",
    "height_shift_range",
    "shear_range",
    "zoom_range",
    "horizontal_flip",
    "vertical_flip",
)
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
) -> tuple[keras.utils.Sequence, int]:
    """Create a streaming Keras sequence of paired images and masks."""
    image_augment_dict = dict(image_augment_dict or {})
    mask_augment_dict = dict(mask_augment_dict or {})
    _validate_dataset_augmentations(image_augment_dict, mask_augment_dict)
    if mask_augment_dict.get("cval", 0) != 0:
        logger.warning(
            "cval for the mask should typically be 0 (background); configured value: "
            f"{mask_augment_dict['cval']}"
        )
    for augmentations in (image_augment_dict, mask_augment_dict):
        fill_mode = str(augmentations.get("fill_mode", "nearest")).upper()
        if fill_mode not in {"CONSTANT", "REFLECT", "WRAP", "NEAREST"}:
            raise ValueError(f"Unsupported augmentation fill_mode: {fill_mode}")
        if int(augmentations.get("interpolation_order", 1)) not in {0, 1}:
            raise ValueError(
                "interpolation_order must be 0 or 1 for the training pipeline"
            )
    if (
        nb_classes > 2
        and mask_augment_dict.get("rescale") is not None
        and mask_augment_dict["rescale"] != 1
    ):
        raise ValueError(f"With nb_classes > 2 ({nb_classes}), mask rescale must be 1")

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


def _list_image_mask_pairs(
    input_data_dir: Path, image_subdir: str, mask_subdir: str
) -> list[tuple[Path, Path]]:
    image_dir = input_data_dir / image_subdir
    mask_dir = input_data_dir / mask_subdir
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _TRAIN_IMAGE_EXTENSIONS
    )
    mask_paths = {
        path
        for path in mask_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _TRAIN_IMAGE_EXTENSIONS
    }

    if not image_paths:
        raise ValueError(f"No training images found in {image_dir}")

    pairs = []
    matched_masks: set[Path] = set()
    for image_path in image_paths:
        relative_path = image_path.relative_to(image_dir)
        mask_path = mask_dir / relative_path
        if not mask_path.exists() and image_path.suffix.lower() in {".jpg", ".jpeg"}:
            mask_path = mask_path.with_suffix(".png")
        if not mask_path.is_file():
            raise ValueError(f"No mask found for training image {image_path}")
        if mask_path not in mask_paths:
            raise ValueError(
                f"Mask for training image {image_path} has unsupported format: "
                f"{mask_path}"
            )
        if mask_path in matched_masks:
            raise ValueError(f"Mask {mask_path} matches multiple training images")
        pairs.append((image_path, mask_path))
        matched_masks.add(mask_path)

    unmatched_masks = sorted(mask_paths - matched_masks)
    if unmatched_masks:
        raise ValueError(
            f"Found masks without matching training images in {mask_dir}: "
            f"{unmatched_masks[:5]}"
        )
    return pairs


def _decode_with_pil(
    path_value: str | Path | np.ndarray,
    color_mode_value: str | np.ndarray,
) -> np.ndarray:
    path_string = np.asarray(path_value).item()
    if isinstance(path_string, bytes):
        path_string = path_string.decode("utf-8")
    color_mode = np.asarray(color_mode_value).item()
    if isinstance(color_mode, bytes):
        color_mode = color_mode.decode("ascii")
    pil_color_mode = {"grayscale": "L", "rgb": "RGB", "rgba": "RGBA"}[
        color_mode.lower()
    ]
    with Image.open(path_string) as image:
        decoded = np.asarray(image.convert(pil_color_mode), dtype=np.uint8)
    if decoded.ndim == 2:
        decoded = decoded[..., np.newaxis]
    return decoded


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


def _zoom_bounds(value: Any) -> tuple[float, float]:
    if value is None:
        return 1.0, 1.0
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a two-value zoom range, got {value}")
        return float(value[0]), float(value[1])
    zoom = abs(float(value))
    return 1.0 - zoom, 1.0 + zoom


def _has_spatial_augmentation(augmentations: dict[str, Any]) -> bool:
    for key in _GEOMETRIC_AUGMENTATION_KEYS:
        value = augmentations.get(key)
        if key in {"horizontal_flip", "vertical_flip"}:
            if value:
                return True
        elif isinstance(value, (list, tuple)):
            if key == "zoom_range":
                if tuple(value) != (1.0, 1.0):
                    return True
            elif any(float(item) != 0 for item in value):
                return True
        elif value is not None and float(value) != 0:
            return True
    return False


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


def _validate_dataset_augmentations(
    image_augmentations: dict[str, Any], mask_augmentations: dict[str, Any]
) -> None:
    if list(image_augmentations) != list(mask_augmentations):
        raise ValueError(
            "The same augmentation keys should be specified in the same order for "
            "images and masks."
        )
    for name, augmentations in (
        ("image_augmentations", image_augmentations),
        ("mask_augmentations", mask_augmentations),
    ):
        unsupported = set(augmentations) - _AUGMENTATION_KEYS
        if unsupported:
            raise ValueError(
                f"Unsupported {name} settings for the training pipeline: "
                f"{sorted(unsupported)}"
            )

    for key in _GEOMETRIC_AUGMENTATION_KEYS:
        if image_augmentations.get(key) != mask_augmentations.get(key):
            raise ValueError(f"{key} must have the same value for images and masks.")

    image_brightness = image_augmentations.get("brightness_range")
    mask_brightness = mask_augmentations.get("brightness_range")
    if (image_brightness is None) != (mask_brightness is None):
        raise ValueError("brightness_range must be specified for both images and masks")
    if image_brightness is not None and (
        not isinstance(image_brightness, (list, tuple))
        or len(image_brightness) != 2
        or min(image_brightness) < 0
        or image_brightness[0] > image_brightness[1]
    ):
        raise ValueError("image brightness_range must contain two non-negative values")
    if mask_brightness is not None and list(mask_brightness) != [1.0, 1.0]:
        raise ValueError(
            "brightness_range is applied to the image; the mask range must be [1, 1]"
        )


def _save_augmented_batch(
    image_batch: np.ndarray,
    mask_batch: np.ndarray,
    batch_index: np.ndarray,
    *,
    save_to_dir: Path,
    image_rescale: float | None,
) -> np.int64:
    batch_id = int(np.asarray(batch_index).item())
    for image_id, (image, mask) in enumerate(zip(image_batch, mask_batch, strict=True)):
        image_to_save = image
        if image_rescale is not None and image_rescale != 1:
            image_to_save = image_to_save / image_rescale
        image_to_save = np.clip(image_to_save, 0, 255).astype(np.uint8)
        if image_to_save.shape[-1] == 1:
            image_to_save = image_to_save[..., 0]

        output_dir = save_to_dir / f"{batch_id:0>4}"
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_to_save).save(output_dir / f"{image_id}_image.jpg")
        for channel_id in range(mask.shape[-1]):
            mask_to_save = np.clip(mask[..., channel_id] * 255, 0, 255).astype(np.uint8)
            Image.fromarray(mask_to_save).save(
                output_dir / f"{image_id}_mask_{channel_id}.png"
            )
    return np.int64(0)

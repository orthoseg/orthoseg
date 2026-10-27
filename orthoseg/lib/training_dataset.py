"""Streaming TensorFlow input pipeline for paired training images and masks."""

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
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
) -> tuple[tf.data.Dataset, int]:
    """Create a streaming TensorFlow dataset of paired images and masks."""
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
                "interpolation_order must be 0 or 1 for the tf.data pipeline"
            )
    if (
        nb_classes > 2
        and mask_augment_dict.get("rescale") is not None
        and mask_augment_dict["rescale"] != 1
    ):
        raise ValueError(f"With nb_classes > 2 ({nb_classes}), mask rescale must be 1")

    pairs = _list_image_mask_pairs(input_data_dir, image_subdir, mask_subdir)
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            [str(image_path) for image_path, _ in pairs],
            [str(mask_path) for _, mask_path in pairs],
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(pairs),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.enumerate()

    dataset = dataset.map(
        partial(
            _load_pair,
            image_color_mode=image_color_mode,
            mask_color_mode=mask_color_mode,
            target_size=target_size,
            image_augmentations=image_augment_dict,
            mask_augmentations=mask_augment_dict,
            seed=seed,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.enumerate()

    save_to_dir = None
    if save_to_subdir is not None:
        save_to_dir = input_data_dir / save_to_subdir
        save_to_dir.mkdir(parents=True, exist_ok=True)

    image_brightness = image_augment_dict.get("brightness_range")
    image_rescale = image_augment_dict.get("rescale")

    dataset = dataset.map(
        partial(
            _prepare_batch,
            image_brightness=image_brightness,
            seed=seed,
            nb_classes=nb_classes,
            save_to_dir=save_to_dir,
            image_rescale=image_rescale,
            model_preprocess_input=model_preprocess_input,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE), len(pairs)


def _load_pair(
    index: tf.Tensor,
    paths: tuple[tf.Tensor, tf.Tensor],
    *,
    image_color_mode: str,
    mask_color_mode: str,
    target_size: tuple[int, int],
    image_augmentations: dict[str, Any],
    mask_augmentations: dict[str, Any],
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
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
            height=tf.shape(image)[0],
            width=tf.shape(image)[1],
        )
        image = _apply_projective_transform(
            image, transform, image_augmentations, is_mask=False
        )
        mask = _apply_projective_transform(
            mask, transform, mask_augmentations, is_mask=True
        )

    image_rescale = image_augmentations.get("rescale")
    if image_rescale is not None:
        image *= tf.cast(image_rescale, tf.float32)
    mask_rescale = mask_augmentations.get("rescale")
    if mask_rescale is not None:
        mask *= tf.cast(mask_rescale, tf.float32)
    return image, mask


def _prepare_batch(
    batch_index: tf.Tensor,
    batch: tuple[tf.Tensor, tf.Tensor],
    *,
    image_brightness: list[float] | tuple[float, float] | None,
    seed: int,
    nb_classes: int,
    save_to_dir: Path | None,
    image_rescale: float | None,
    model_preprocess_input: Callable | None,
) -> tuple[tf.Tensor, tf.Tensor]:
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
        mask = tf.one_hot(
            tf.cast(tf.squeeze(mask, axis=-1), tf.int32),
            depth=nb_classes,
        )

    if save_to_dir is not None:
        save_token = tf.numpy_function(
            lambda images, masks, index: _save_augmented_batch(
                images,
                masks,
                index,
                save_to_dir=save_to_dir,
                image_rescale=image_rescale,
            ),
            [image, mask, batch_index],
            tf.int64,
        )
        with tf.control_dependencies([save_token]):
            image = tf.identity(image)
            mask = tf.identity(mask)

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


def _decode_tiff_with_pil(
    path_value: np.ndarray, color_mode_value: np.ndarray
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
    path: tf.Tensor,
    *,
    color_mode: str,
    target_size: tuple[int, int],
) -> tf.Tensor:
    channels_by_mode = {"grayscale": 1, "rgb": 3, "rgba": 4}
    try:
        channels = channels_by_mode[color_mode.lower()]
    except KeyError as ex:
        raise ValueError(f"Unsupported color mode: {color_mode}") from ex

    def decode_with_pil() -> tf.Tensor:
        decoded = tf.numpy_function(
            _decode_tiff_with_pil,
            [path, tf.constant(color_mode.lower())],
            tf.uint8,
        )
        decoded.set_shape([None, None, channels])
        return decoded

    def decode_with_tensorflow() -> tf.Tensor:
        decoded = tf.io.decode_image(
            tf.io.read_file(path),
            channels=channels,
            expand_animations=False,
        )
        decoded.set_shape([None, None, channels])
        return decoded

    path_lower = tf.strings.lower(path)
    is_pil_format = tf.strings.regex_full_match(path_lower, r".*\.(tif|tiff|ppm)")
    decoded = tf.cond(is_pil_format, decode_with_pil, decode_with_tensorflow)
    resized = tf.image.resize(decoded, target_size, method="nearest")
    resized.set_shape([target_size[0], target_size[1], channels])
    return tf.cast(resized, tf.float32)


def _stateless_uniform(
    seed: int,
    index: tf.Tensor,
    stream: int,
    minval: float,
    maxval: float,
) -> tf.Tensor:
    seed_pair = tf.stack(
        [
            tf.cast(seed, tf.int32),
            tf.cast(index, tf.int32) + tf.cast(stream, tf.int32),
        ]
    )
    if minval == maxval:
        return tf.constant(minval, dtype=tf.float32)
    return tf.random.stateless_uniform(
        [],
        seed=seed_pair,
        minval=minval,
        maxval=maxval,
        dtype=tf.float32,
    )


def _sample_range(
    value: Any,
    *,
    seed: int,
    index: tf.Tensor,
    stream: int,
    dimension: tf.Tensor | None = None,
) -> tf.Tensor:
    if value is None:
        return tf.constant(0.0, dtype=tf.float32)
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
        sample *= dimension
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
    index: tf.Tensor,
    height: tf.Tensor,
    width: tf.Tensor,
) -> tf.Tensor:
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
        dimension=tf.cast(width, tf.float32),
    )
    shift_y = _sample_range(
        augmentations.get("height_shift_range"),
        seed=seed,
        index=index,
        stream=4,
        dimension=tf.cast(height, tf.float32),
    )
    zoom_min, zoom_max = _zoom_bounds(augmentations.get("zoom_range"))
    zoom_x = _stateless_uniform(seed, index, 5, zoom_min, zoom_max)
    zoom_y = _stateless_uniform(seed, index, 6, zoom_min, zoom_max)
    flip_x = tf.where(
        tf.logical_and(
            tf.constant(bool(augmentations.get("horizontal_flip", False))),
            _stateless_uniform(seed, index, 7, 0.0, 1.0) < 0.5,
        ),
        -1.0,
        1.0,
    )
    flip_y = tf.where(
        tf.logical_and(
            tf.constant(bool(augmentations.get("vertical_flip", False))),
            _stateless_uniform(seed, index, 8, 0.0, 1.0) < 0.5,
        ),
        -1.0,
        1.0,
    )

    rotation_matrix = tf.stack(
        [
            tf.stack([tf.cos(angle), -tf.sin(angle)]),
            tf.stack([tf.sin(angle), tf.cos(angle)]),
        ]
    )
    shear_matrix = tf.stack(
        [
            tf.stack([tf.constant(1.0), -tf.sin(shear)]),
            tf.stack([tf.constant(0.0), tf.cos(shear)]),
        ]
    )
    zoom_matrix = tf.linalg.diag(tf.stack([zoom_x, zoom_y]))
    flip_matrix = tf.linalg.diag(tf.stack([flip_x, flip_y]))
    linear = tf.linalg.matmul(
        tf.linalg.matmul(tf.linalg.matmul(rotation_matrix, shear_matrix), zoom_matrix),
        flip_matrix,
    )

    center_x = (tf.cast(width, tf.float32) - 1.0) / 2.0
    center_y = (tf.cast(height, tf.float32) - 1.0) / 2.0
    offset_x = center_x - linear[0, 0] * center_x - linear[0, 1] * center_y + shift_x
    offset_y = center_y - linear[1, 0] * center_x - linear[1, 1] * center_y + shift_y
    return tf.stack(
        [
            linear[0, 0],
            linear[0, 1],
            offset_x,
            linear[1, 0],
            linear[1, 1],
            offset_y,
            0.0,
            0.0,
        ]
    )


def _apply_projective_transform(
    image: tf.Tensor,
    transform: tf.Tensor,
    augmentations: dict[str, Any],
    *,
    is_mask: bool,
) -> tf.Tensor:
    interpolation_order = int(augmentations.get("interpolation_order", 1))
    if interpolation_order not in {0, 1}:
        raise ValueError("interpolation_order must be 0 or 1 for the tf.data pipeline")
    interpolation = "NEAREST" if is_mask or interpolation_order == 0 else "BILINEAR"
    fill_mode = str(augmentations.get("fill_mode", "nearest")).upper()
    if fill_mode not in {"CONSTANT", "REFLECT", "WRAP", "NEAREST"}:
        raise ValueError(f"Unsupported augmentation fill_mode: {fill_mode}")
    result = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.reshape(transform, [1, 8]),
        output_shape=tf.shape(image)[:2],
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=tf.cast(augmentations.get("cval", 0), tf.float32),
    )[0]
    result.set_shape(image.shape)
    return result


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
                f"Unsupported {name} settings for the tf.data pipeline: "
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

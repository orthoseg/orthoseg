"""Backend-independent helpers for paired training images and masks."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


def _validate_train_dataset_options(
    image_augmentations: dict[str, Any],
    mask_augmentations: dict[str, Any],
    *,
    nb_classes: int,
    pipeline_name: str,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_augmentations = dict(image_augmentations or {})
    mask_augmentations = dict(mask_augmentations or {})
    _validate_dataset_augmentations(
        image_augmentations, mask_augmentations, pipeline_name=pipeline_name
    )
    if mask_augmentations.get("cval", 0) != 0:
        logger.warning(
            "cval for the mask should typically be 0 (background); configured value: "
            f"{mask_augmentations['cval']}"
        )
    for augmentations in (image_augmentations, mask_augmentations):
        fill_mode = str(augmentations.get("fill_mode", "nearest")).upper()
        if fill_mode not in {"CONSTANT", "REFLECT", "WRAP", "NEAREST"}:
            raise ValueError(f"Unsupported augmentation fill_mode: {fill_mode}")
        if int(augmentations.get("interpolation_order", 1)) not in {0, 1}:
            raise ValueError(
                f"interpolation_order must be 0 or 1 for the {pipeline_name} pipeline"
            )
    if (
        nb_classes > 2
        and mask_augmentations.get("rescale") is not None
        and mask_augmentations["rescale"] != 1
    ):
        raise ValueError(f"With nb_classes > 2 ({nb_classes}), mask rescale must be 1")
    return image_augmentations, mask_augmentations


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


def _validate_dataset_augmentations(
    image_augmentations: dict[str, Any],
    mask_augmentations: dict[str, Any],
    *,
    pipeline_name: str = "training",
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
                f"Unsupported {name} settings for the {pipeline_name} pipeline: "
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

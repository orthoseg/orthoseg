"""Tests for the streaming training dataset helpers."""

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from orthoseg.lib import training_dataset


@pytest.mark.parametrize(
    ("image_filenames", "mask_filenames", "message"),
    [
        (
            ("tile.png",),
            ("tile.png", "orphan.png"),
            "Found masks without matching training images",
        ),
        (
            ("tile.jpg", "tile.jpeg"),
            ("tile.png",),
            "matches multiple training images",
        ),
    ],
)
def test_list_image_mask_pairs_errors(
    tmp_path, image_filenames, mask_filenames, message
):
    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "mask"
    image_dir.mkdir()
    mask_dir.mkdir()
    for filename in image_filenames:
        Image.new("RGB", (2, 2)).save(image_dir / filename)
    for filename in mask_filenames:
        Image.new("L", (2, 2)).save(mask_dir / filename)

    with pytest.raises(ValueError, match=message):
        training_dataset._list_image_mask_pairs(tmp_path, "image", "mask")


@pytest.mark.parametrize(
    ("image_augmentations", "mask_augmentations", "message"),
    [
        (
            {"horizontal_flip": True, "fill_mode": "nearest"},
            {"fill_mode": "nearest", "horizontal_flip": True},
            "same order for images and masks",
        ),
        (
            {"unsupported_option": True},
            {"unsupported_option": True},
            "Unsupported image_augmentations settings",
        ),
        (
            {"rotation_range": 10},
            {"rotation_range": 20},
            "rotation_range must have the same value",
        ),
        (
            {"brightness_range": None},
            {"brightness_range": [1.0, 1.0]},
            "brightness_range must be specified for both",
        ),
        (
            {"brightness_range": [1.0]},
            {"brightness_range": [1.0]},
            "two non-negative values",
        ),
        (
            {"brightness_range": [2.0, 2.0]},
            {"brightness_range": [2.0, 2.0]},
            "mask range must be",
        ),
    ],
)
def test_validate_dataset_augmentations_errors(
    image_augmentations, mask_augmentations, message
):
    with pytest.raises(ValueError, match=message):
        training_dataset._validate_dataset_augmentations(
            image_augmentations, mask_augmentations
        )


@pytest.mark.parametrize(
    ("image_augmentations", "mask_augmentations", "nb_classes", "message"),
    [
        (
            {"fill_mode": "invalid"},
            {"fill_mode": "invalid"},
            1,
            "Unsupported augmentation fill_mode",
        ),
        (
            {"interpolation_order": 2},
            {"interpolation_order": 2},
            1,
            "interpolation_order must be 0 or 1",
        ),
        (
            {"rescale": 0.5},
            {"rescale": 0.5},
            3,
            "mask rescale must be 1",
        ),
    ],
)
def test_create_train_dataset_errors(
    tmp_path, image_augmentations, mask_augmentations, nb_classes, message
):
    with pytest.raises(ValueError, match=message):
        training_dataset.create_train_dataset(
            input_data_dir=tmp_path,
            image_subdir="image",
            mask_subdir="mask",
            image_augment_dict=image_augmentations,
            mask_augment_dict=mask_augmentations,
            model_preprocess_input=None,
            nb_classes=nb_classes,
        )


@pytest.mark.parametrize(
    ("value", "dimension", "expected"),
    [
        ([0.25, 0.25], 8.0, 2.0),
        (None, None, 0.0),
    ],
)
def test_sample_range_scales(value, dimension, expected):
    sample = training_dataset._sample_range(
        value,
        seed=3,
        index=tf.constant(0),
        stream=1,
        dimension=tf.constant(dimension) if dimension is not None else None,
    )
    assert float(sample.numpy()) == expected


def test_sample_range_error():
    with pytest.raises(ValueError, match="two-value augmentation range"):
        training_dataset._sample_range([0.1], seed=3, index=tf.constant(0), stream=1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (1.0, 1.0)),
        (0.2, (0.8, 1.2)),
        ([0.7, 1.3], (0.7, 1.3)),
    ],
)
def test_zoom_bounds(value, expected):
    assert training_dataset._zoom_bounds(value) == expected


def test_zoom_bounds_error():
    with pytest.raises(ValueError, match="two-value zoom range"):
        training_dataset._zoom_bounds([0.8])


@pytest.mark.parametrize(
    ("augmentations", "expected"),
    [
        ({}, False),
        ({"zoom_range": (1.0, 1.0)}, False),
        ({"width_shift_range": [0.0, 0.25]}, True),
        ({"horizontal_flip": True}, True),
    ],
)
def test_has_spatial_augmentation(augmentations, expected):
    assert training_dataset._has_spatial_augmentation(augmentations) is expected


def test_decode_and_resize_errors():
    with pytest.raises(ValueError, match="Unsupported color mode"):
        training_dataset._decode_and_resize(
            tf.constant("image.png"), color_mode="indexed", target_size=(2, 2)
        )


def test_decode_tiff_with_pil(tmp_path):
    image_path = tmp_path / "gray.tif"
    Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(image_path)

    decoded = training_dataset._decode_tiff_with_pil(
        np.asarray(str(image_path).encode("utf-8")), np.asarray(b"grayscale")
    )

    assert decoded.shape == (2, 2, 1)
    np.testing.assert_array_equal(decoded[:, :, 0], [[0, 1], [2, 3]])


def test_save_augmented_batch(tmp_path):
    images = np.full((1, 4, 4, 1), 40, dtype=np.float32)
    masks = np.zeros((1, 4, 4, 2), dtype=np.float32)
    masks[..., 1] = 1.0

    result = training_dataset._save_augmented_batch(
        images,
        masks,
        np.asarray(3),
        save_to_dir=tmp_path,
        image_rescale=0.5,
    )

    output_dir = tmp_path / "0003"
    assert result == 0
    with Image.open(output_dir / "0_image.jpg") as saved_image:
        assert saved_image.mode == "L"
        assert np.asarray(saved_image).mean() == pytest.approx(80, abs=2)
    with Image.open(output_dir / "0_mask_0.png") as saved_mask:
        assert np.asarray(saved_mask).max() == 0
    with Image.open(output_dir / "0_mask_1.png") as saved_mask:
        assert np.asarray(saved_mask).min() == 255

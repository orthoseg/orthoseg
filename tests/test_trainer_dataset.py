"""Tests for the streaming training dataset helpers."""

import numpy as np
import pytest
from PIL import Image

from orthoseg.helpers import _trainer_dataset


def test_apply_projective_transform():
    image = np.zeros((3, 3, 1), dtype=np.float32)
    image[1, 1, 0] = 1.0
    half_pixel_transform = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0])
    augmentations = {"interpolation_order": 1, "fill_mode": "constant"}

    bilinear_image = _trainer_dataset._apply_projective_transform(
        image, half_pixel_transform, augmentations, is_mask=False
    )
    nearest_mask = _trainer_dataset._apply_projective_transform(
        image, half_pixel_transform, augmentations, is_mask=True
    )
    nearest_image = _trainer_dataset._apply_projective_transform(
        image,
        half_pixel_transform,
        {"interpolation_order": 0, "fill_mode": "constant"},
        is_mask=False,
    )

    assert np.any((bilinear_image > 0) & (bilinear_image < 1))
    assert set(np.unique(nearest_mask)) <= {0.0, 1.0}
    np.testing.assert_array_equal(nearest_mask, nearest_image)


@pytest.mark.parametrize(
    ("augmentations", "message"),
    [
        ({"interpolation_order": 2}, "interpolation_order must be 0 or 1"),
        ({"fill_mode": "invalid"}, "Unsupported augmentation fill_mode"),
    ],
)
def test_apply_projective_transform_errors(augmentations, message):
    image = np.zeros((2, 2, 1), dtype=np.float32)
    identity_transform = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match=message):
        _trainer_dataset._apply_projective_transform(
            image, identity_transform, augmentations, is_mask=False
        )


@pytest.mark.parametrize(
    ("fill_mode", "expected"),
    [
        ("constant", 9.0),
        ("reflect", 0.0),
        ("wrap", 2.0),
        ("nearest", 0.0),
    ],
)
def test_apply_projective_transform_nearest_fill_modes(fill_mode, expected):
    image = np.array([[[0.0], [1.0], [2.0]]], dtype=np.float32)
    transform = np.array([1.0, 0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0])

    result = _trainer_dataset._apply_projective_transform(
        image,
        transform,
        {"fill_mode": fill_mode, "interpolation_order": 0, "cval": 9},
        is_mask=False,
    )

    assert result[0, 0, 0] == expected


def test_create_train_dataset(tmp_path):
    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "mask"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.fromarray(np.full((2, 2, 3), 80, dtype=np.uint8)).save(image_dir / "tile.png")
    Image.fromarray(np.full((2, 2), 2, dtype=np.uint8)).save(mask_dir / "tile.png")

    dataset, _ = _trainer_dataset.create_train_dataset(
        input_data_dir=tmp_path,
        image_subdir="image",
        mask_subdir="mask",
        image_augment_dict={"rescale": 0.5, "brightness_range": [1.0, 1.0]},
        mask_augment_dict={"rescale": 0.5, "brightness_range": [1.0, 1.0]},
        model_preprocess_input=None,
        batch_size=1,
        target_size=(2, 2),
        shuffle=False,
    )

    images, masks = next(iter(dataset))
    assert isinstance(images, np.ndarray)
    assert isinstance(masks, np.ndarray)
    np.testing.assert_array_equal(images, np.full((1, 2, 2, 3), 40.0))
    np.testing.assert_array_equal(masks, np.ones((1, 2, 2, 1)))


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
        (
            {"fill_mode": "nearest"},
            {"fill_mode": "invalid"},
            1,
            "Unsupported augmentation fill_mode",
        ),
        (
            {"interpolation_order": 1},
            {"interpolation_order": 2},
            1,
            "interpolation_order must be 0 or 1",
        ),
    ],
)
def test_create_train_dataset_errors(
    tmp_path, image_augmentations, mask_augmentations, nb_classes, message
):
    with pytest.raises(ValueError, match=message):
        _trainer_dataset.create_train_dataset(
            input_data_dir=tmp_path,
            image_subdir="image",
            mask_subdir="mask",
            image_augment_dict=image_augmentations,
            mask_augment_dict=mask_augmentations,
            model_preprocess_input=None,
            nb_classes=nb_classes,
        )


def test_create_train_dataset_warns(tmp_path, caplog):
    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "mask"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.fromarray(np.full((2, 2, 3), 12, dtype=np.uint8)).save(image_dir / "tile.png")
    Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(
        mask_dir / "tile.png"
    )

    dataset, dataset_size = _trainer_dataset.create_train_dataset(
        input_data_dir=tmp_path,
        image_subdir="image",
        mask_subdir="mask",
        image_augment_dict={"fill_mode": "constant", "cval": 0},
        mask_augment_dict={"fill_mode": "constant", "cval": 1},
        model_preprocess_input=None,
        batch_size=1,
        target_size=(2, 2),
        shuffle=True,
    )

    images, masks = next(iter(dataset))
    assert dataset_size == 1
    assert images.shape == (1, 2, 2, 3)
    assert masks.shape == (1, 2, 2, 1)
    assert "cval for the mask should typically be 0" in caplog.text


@pytest.mark.parametrize(
    ("extension", "color_mode", "image_values"),
    [
        (
            ".png",
            "rgb",
            np.array(
                [
                    [[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]],
                ],
                dtype=np.uint8,
            ),
        ),
        (
            ".ppm",
            "rgb",
            np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8),
        ),
        (
            ".png",
            "rgba",
            np.full((2, 2, 4), [10, 20, 30, 40], dtype=np.uint8),
        ),
    ],
)
def test_decode_and_resize(tmp_path, extension, color_mode, image_values):
    image_path = tmp_path / f"image{extension}"
    Image.fromarray(image_values).save(image_path)

    decoded = _trainer_dataset._decode_and_resize(
        image_path,
        color_mode=color_mode,
        target_size=(2, 2),
    )

    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, image_values.astype(np.float32))


def test_decode_and_resize_errors():
    with pytest.raises(ValueError, match="Unsupported color mode"):
        _trainer_dataset._decode_and_resize(
            "image.png", color_mode="indexed", target_size=(2, 2)
        )


def test_decode_and_resize_uses_centered_nearest_mapping(tmp_path):
    image_path = tmp_path / "image.png"
    image_values = np.arange(12, dtype=np.uint8).reshape((3, 4))
    Image.fromarray(image_values).save(image_path)

    decoded = _trainer_dataset._decode_and_resize(
        image_path,
        color_mode="grayscale",
        target_size=(5, 6),
    )

    expected = np.array(
        [
            [0, 1, 1, 2, 3, 3],
            [0, 1, 1, 2, 3, 3],
            [4, 5, 5, 6, 7, 7],
            [8, 9, 9, 10, 11, 11],
            [8, 9, 9, 10, 11, 11],
        ],
        dtype=np.float32,
    )[..., np.newaxis]
    np.testing.assert_array_equal(decoded, expected)


def test_decode_with_pil_grayscale(tmp_path):
    image_path = tmp_path / "gray.tif"
    Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(image_path)

    decoded = _trainer_dataset._decode_with_pil(
        np.asarray(str(image_path).encode("utf-8")), np.asarray(b"grayscale")
    )

    assert decoded.shape == (2, 2, 1)
    np.testing.assert_array_equal(decoded[:, :, 0], [[0, 1], [2, 3]])


def test_decode_with_pil_rgb(tmp_path):
    image_path = tmp_path / "rgb.tif"
    image_values = np.full((2, 2, 3), 37, dtype=np.uint8)
    Image.fromarray(image_values).save(image_path)

    decoded = _trainer_dataset._decode_with_pil(
        np.asarray(str(image_path)), np.asarray("rgb")
    )

    assert decoded.shape == (2, 2, 3)
    np.testing.assert_array_equal(decoded, image_values)


@pytest.mark.parametrize(
    ("augmentations", "expected"),
    [
        ({}, False),
        ({"zoom_range": (1.0, 1.0)}, False),
        ({"width_shift_range": [0.0, 0.25]}, True),
        ({"horizontal_flip": True}, True),
        ({"rotation_range": 0}, False),
        ({"rotation_range": 5}, True),
    ],
)
def test_has_spatial_augmentation(augmentations, expected):
    assert _trainer_dataset._has_spatial_augmentation(augmentations) is expected


@pytest.mark.parametrize(
    ("image_filenames", "mask_filenames", "message"),
    [
        ((), (), "No training images found"),
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
        _trainer_dataset._list_image_mask_pairs(tmp_path, "image", "mask")


@pytest.mark.parametrize("horizontal_flip", [False, True])
def test_load_pair(tmp_path, horizontal_flip):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    mask_values = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    image_values = np.repeat((mask_values * 100)[:, :, np.newaxis], 3, axis=2)
    Image.fromarray(image_values).save(image_path)
    Image.fromarray(mask_values).save(mask_path)
    augmentations = {"horizontal_flip": horizontal_flip, "rescale": 0.5}

    image, mask = _trainer_dataset._load_pair(
        0,
        (image_path, mask_path),
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        target_size=(2, 2),
        image_augmentations=augmentations,
        mask_augmentations=augmentations,
        seed=7,
    )

    np.testing.assert_array_equal(image[:, :, 0], mask[:, :, 0] * 100)
    assert set(np.unique(mask)) == {0.0, 0.5}


@pytest.mark.parametrize(
    ("augmentations", "expected"),
    [
        ({}, [1, 0, 0, 0, 1, 0, 0, 0]),
        (
            {
                "rotation_range": [0.0, 0.0],
                "shear_range": [0.0, 0.0],
                "width_shift_range": [0.25, 0.25],
                "height_shift_range": [0.5, 0.5],
                "zoom_range": [1.0, 1.0],
                "horizontal_flip": False,
                "vertical_flip": False,
            },
            [1, 0, 2, 0, 1, 2, 0, 0],
        ),
    ],
)
def test_make_projective_transform(augmentations, expected):
    transform = _trainer_dataset._make_projective_transform(
        augmentations,
        seed=3,
        index=0,
        height=4,
        width=8,
    )

    np.testing.assert_allclose(transform, expected, atol=1e-6)


def test_prepare_batch_applies_brightness_one_hot_and_preprocess():
    images = np.full((1, 2, 2, 1), 10.0, dtype=np.float32)
    masks = np.array([[[[0.0], [1.0]], [[1.0], [0.0]]]], dtype=np.float32)

    processed_images, one_hot_masks = _trainer_dataset._prepare_batch(
        0,
        (images, masks),
        image_brightness=[2.0, 2.0],
        seed=4,
        nb_classes=2,
        save_to_dir=None,
        image_rescale=None,
        model_preprocess_input=lambda image: image / 20.0,
    )

    np.testing.assert_array_equal(processed_images, np.ones((1, 2, 2, 1)))
    assert one_hot_masks.shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(one_hot_masks[0, :, :, 1], masks[0, :, :, 0])


def test_prepare_batch_saves_examples(tmp_path):
    images = np.full((1, 2, 2, 3), 80.0, dtype=np.float32)
    masks = np.ones((1, 2, 2, 1), dtype=np.float32)

    prepared_images, prepared_masks = _trainer_dataset._prepare_batch(
        5,
        (images, masks),
        image_brightness=None,
        seed=4,
        nb_classes=1,
        save_to_dir=tmp_path,
        image_rescale=0.5,
        model_preprocess_input=None,
    )

    assert prepared_images.shape == images.shape
    assert prepared_masks.shape == masks.shape
    assert (tmp_path / "0005" / "0_image.jpg").exists()
    assert (tmp_path / "0005" / "0_mask_0.png").exists()


def test_prepare_batch_without_optional_processing():
    images = np.ones((1, 2, 2, 1), dtype=np.float32)
    masks = np.zeros((1, 2, 2, 1), dtype=np.float32)

    prepared_images, prepared_masks = _trainer_dataset._prepare_batch(
        0,
        (images, masks),
        image_brightness=None,
        seed=4,
        nb_classes=1,
        save_to_dir=None,
        image_rescale=None,
        model_preprocess_input=None,
    )

    np.testing.assert_array_equal(prepared_images, images)
    np.testing.assert_array_equal(prepared_masks, masks)


def test_sample_range_error():
    with pytest.raises(ValueError, match="two-value augmentation range"):
        _trainer_dataset._sample_range([0.1], seed=3, index=0, stream=1)


def test_sample_range_integer_scalar_is_not_scaled_by_dimension():
    sample = _trainer_dataset._sample_range(
        1,
        seed=3,
        index=0,
        stream=1,
        dimension=8.0,
    )

    assert -1.0 <= sample <= 1.0


def test_sample_range_is_reproducible_for_seed_and_index():
    kwargs = {
        "seed": 12,
        "index": 4,
        "stream": 3,
    }

    first = _trainer_dataset._sample_range([0.2, 0.8], **kwargs)
    second = _trainer_dataset._sample_range([0.2, 0.8], **kwargs)

    assert first == second
    assert 0.2 <= first <= 0.8


@pytest.mark.parametrize(
    ("value", "dimension", "expected"),
    [
        ([0.25, 0.25], 8.0, 2.0),
        (None, None, 0.0),
    ],
)
def test_sample_range_scales(value, dimension, expected):
    sample = _trainer_dataset._sample_range(
        value,
        seed=3,
        index=0,
        stream=1,
        dimension=dimension,
    )
    assert sample == expected


def test_save_augmented_batch(tmp_path):
    images = np.full((1, 4, 4, 1), 40, dtype=np.float32)
    masks = np.zeros((1, 4, 4, 2), dtype=np.float32)
    masks[..., 1] = 1.0

    result = _trainer_dataset._save_augmented_batch(
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
        _trainer_dataset._validate_dataset_augmentations(
            image_augmentations, mask_augmentations
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (1.0, 1.0)),
        (0.2, (0.8, 1.2)),
        ([0.7, 1.3], (0.7, 1.3)),
    ],
)
def test_zoom_bounds(value, expected):
    assert _trainer_dataset._zoom_bounds(value) == expected


def test_zoom_bounds_error():
    with pytest.raises(ValueError, match="two-value zoom range"):
        _trainer_dataset._zoom_bounds([0.8])

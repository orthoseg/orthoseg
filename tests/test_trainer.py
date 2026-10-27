"""Tests for module trainer."""

from unittest.mock import patch

import keras
import numpy as np
import pytest
from PIL import Image

import orthoseg.model.model_helper as mh
from orthoseg.lib import trainer


def test_train_error_model_exists_no_preload(tmp_path):
    """Raise ValueError when a model exists but no preload file is specified."""
    augmentations = {"fill_mode": "constant", "cval": 0}
    hyperparams = mh.HyperParams(
        architecture=mh.ArchitectureParams(
            architecture="mobilenetv2+unet", classes=["background", "class1"]
        ),
        train=mh.TrainParams(
            image_augmentations=augmentations, mask_augmentations=augmentations
        ),
    )

    fake_best_model = mh.ModelInfo(
        filepath=tmp_path / "some_model.keras",
        filename="some_model",
        basefilename="test_01",
        segment_subject="test",
        traindata_id=1,
        architecture_id=0,
        trainparams_id=0,
        monitor_metric_accuracy=0.9,
        epoch=1,
        save_format="keras",
    )

    with patch("orthoseg.lib.trainer.mh.get_best_model", return_value=fake_best_model):
        with pytest.raises(
            ValueError, match="Model exists but preload model file not specified"
        ):
            trainer.train(
                traindata_dir=tmp_path,
                validationdata_dir=tmp_path,
                model_save_dir=tmp_path,
                segment_subject="test",
                traindata_id=1,
                hyperparams=hyperparams,
                model_preload_filepath=None,
            )


def test_create_train_dataset_pairs_files_and_one_hot_encodes_masks(tmp_path):
    """Load image/mask pairs by filename and produce one-hot mask batches."""
    data_dir = tmp_path / "training"
    image_dir = data_dir / "image"
    mask_dir = data_dir / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()

    image_a = np.full((2, 2, 3), 10, dtype=np.uint8)
    image_b = np.full((2, 2, 3), 20, dtype=np.uint8)
    mask_a = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    mask_b = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    Image.fromarray(image_a).save(image_dir / "a.png")
    Image.fromarray(image_b).save(image_dir / "b.png")
    Image.fromarray(mask_a).save(mask_dir / "a.png")
    Image.fromarray(mask_b).save(mask_dir / "b.png")

    augmentations = {"fill_mode": "constant", "cval": 0}
    dataset, dataset_size = trainer.create_train_dataset(
        input_data_dir=data_dir,
        image_subdir="image",
        mask_subdir="mask",
        image_augment_dict=augmentations,
        mask_augment_dict=augmentations,
        model_preprocess_input=lambda image: image / 10.0,
        batch_size=2,
        nb_classes=2,
        target_size=(2, 2),
        shuffle=False,
    )

    images, masks = next(iter(dataset))
    assert dataset_size == 2
    assert int(dataset.cardinality().numpy()) == 1
    assert images.shape == (2, 2, 2, 3)
    assert masks.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(images.numpy()[0], image_a.astype(np.float32) / 10.0)
    np.testing.assert_array_equal(masks.numpy()[0, :, :, 1], mask_a)
    np.testing.assert_array_equal(masks.numpy()[1, :, :, 1], mask_b)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(2, 2, 3)),
            keras.layers.Conv2D(2, kernel_size=1, activation="softmax"),
        ]
    )
    model.compile(optimizer="sgd", loss="categorical_crossentropy")
    history = model.fit(dataset, epochs=1, verbose=0, shuffle=False)
    assert len(history.history["loss"]) == 1


def test_create_train_dataset_applies_same_geometry_to_images_and_masks(tmp_path):
    """Spatial transforms stay aligned and brightness changes images only."""
    data_dir = tmp_path / "training"
    image_dir = data_dir / "image"
    mask_dir = data_dir / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()

    mask_values = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    image_values = np.repeat((mask_values * 50)[:, :, np.newaxis], 3, axis=2)
    Image.fromarray(image_values).save(image_dir / "tile.png")
    Image.fromarray(mask_values).save(mask_dir / "tile.png")

    image_augmentations = {
        "fill_mode": "constant",
        "cval": 0,
        "horizontal_flip": True,
        "vertical_flip": True,
        "brightness_range": [2.0, 2.0],
    }
    mask_augmentations = {
        "fill_mode": "constant",
        "cval": 0,
        "horizontal_flip": True,
        "vertical_flip": True,
        "brightness_range": [1.0, 1.0],
    }
    dataset, _ = trainer.create_train_dataset(
        input_data_dir=data_dir,
        image_subdir="image",
        mask_subdir="mask",
        image_augment_dict=image_augmentations,
        mask_augment_dict=mask_augmentations,
        model_preprocess_input=None,
        batch_size=1,
        nb_classes=2,
        target_size=(2, 2),
        shuffle=False,
        seed=7,
    )

    images, masks = next(iter(dataset))
    np.testing.assert_array_equal(
        images.numpy()[0, :, :, 0] > 0,
        masks.numpy()[0, :, :, 1] > 0,
    )
    assert images.numpy().max() == 100


def test_create_train_dataset_reports_missing_mask(tmp_path):
    """Fail clearly instead of pairing an image with the wrong mask."""
    image_dir = tmp_path / "training" / "image"
    image_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_dir / "tile.png")

    augmentations = {"fill_mode": "constant", "cval": 0}
    with pytest.raises(ValueError, match="No mask found"):
        trainer.create_train_dataset(
            input_data_dir=tmp_path / "training",
            image_subdir="image",
            mask_subdir="mask",
            image_augment_dict=augmentations,
            mask_augment_dict=augmentations,
            model_preprocess_input=None,
            target_size=(2, 2),
        )


def test_create_train_dataset_pairs_jpeg_with_png_mask_and_saves_output(tmp_path):
    """Support lossless PNG masks for JPEG inputs and preserve augmented saving."""
    data_dir = tmp_path / "training"
    image_dir = data_dir / "image"
    mask_dir = data_dir / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    Image.fromarray(np.full((2, 2, 3), 80, dtype=np.uint8)).save(image_dir / "tile.jpg")
    Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(
        mask_dir / "tile.png"
    )

    augmentations = {"fill_mode": "constant", "cval": 0}
    dataset, dataset_size = trainer.create_train_dataset(
        input_data_dir=data_dir,
        image_subdir="image",
        mask_subdir="mask",
        image_augment_dict=augmentations,
        mask_augment_dict=augmentations,
        model_preprocess_input=None,
        batch_size=1,
        save_to_subdir="augmented",
        nb_classes=2,
        target_size=(2, 2),
        shuffle=False,
    )

    next(iter(dataset))
    output_dir = data_dir / "augmented" / "0000"
    assert dataset_size == 1
    assert (output_dir / "0_image.jpg").exists()
    assert (output_dir / "0_mask_0.png").exists()
    assert (output_dir / "0_mask_1.png").exists()


def test_create_train_dataset_decodes_tiff_inputs(tmp_path):
    """Decode TIFF inputs through the TensorFlow dataset pipeline."""
    data_dir = tmp_path / "training"
    image_dir = data_dir / "image"
    mask_dir = data_dir / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    image_values = np.full((2, 2, 3), 30, dtype=np.uint8)
    mask_values = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    Image.fromarray(image_values).save(image_dir / "tile.tif")
    Image.fromarray(mask_values).save(mask_dir / "tile.tif")

    augmentations = {"fill_mode": "constant", "cval": 0}
    dataset, _ = trainer.create_train_dataset(
        input_data_dir=data_dir,
        image_subdir="image",
        mask_subdir="mask",
        image_augment_dict=augmentations,
        mask_augment_dict=augmentations,
        model_preprocess_input=None,
        batch_size=1,
        nb_classes=2,
        target_size=(2, 2),
        shuffle=False,
    )

    images, masks = next(iter(dataset))
    np.testing.assert_array_equal(images.numpy()[0], image_values.astype(np.float32))
    assert set(np.unique(masks.numpy())) == {0.0, 1.0}

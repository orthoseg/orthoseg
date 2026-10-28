import threading
from contextlib import AbstractContextManager, nullcontext

import numpy as np
import pytest
import rasterio as rio
import rasterio.transform as rio_transform

from orthoseg.lib import predicter


@pytest.mark.parametrize(
    "no_images_ok, exp_error",
    [
        (False, True),
        (True, False),
    ],
)
def test_predict_dir_input_image_dir_empty(
    tmp_path,
    no_images_ok: bool,
    exp_error: bool,
):
    input_image_dir = tmp_path / "input"
    input_image_dir.mkdir()
    output_image_dir = tmp_path / "output"
    output_image_dir.mkdir()

    handler: AbstractContextManager
    if exp_error:
        handler = pytest.raises(ValueError)
    else:
        handler = nullcontext()

    with handler:
        predicter.predict_dir(
            model=None,  # type: ignore
            preprocess_input=None,  # type: ignore
            input_image_dir=input_image_dir,
            output_image_dir=output_image_dir,
            output_vector_path=None,
            classes=[],
            no_images_ok=no_images_ok,
        )


def test_create_image_read_dataset_matches_read_image(tmp_path):
    image_path = tmp_path / "input.tif"
    image_data = np.arange(3 * 4 * 5, dtype=np.uint8).reshape((3, 4, 5))
    image_transform = rio_transform.from_origin(100, 200, 2, 2)
    with rio.open(
        image_path,
        "w",
        driver="GTiff",
        width=5,
        height=4,
        count=3,
        dtype=rio.uint8,
        crs="EPSG:31370",
        transform=image_transform,
    ) as image_ds:
        image_ds.write(image_data)

    def preprocess_input(image):
        return image / 255.0

    expected = predicter.read_image(
        image_path=image_path,
        preprocess_input=preprocess_input,
    )
    image_metadata = {}
    dataset = predicter._create_image_read_dataset(
        image_paths=[image_path],
        projection_if_missing=None,
        preprocess_input=preprocess_input,
        image_dtype=predicter.tf.float32,
        read_batch_size=1,
        image_metadata=image_metadata,
        image_metadata_lock=threading.Lock(),
    )

    (
        dataset_paths,
        actual_image_data_batch,
        image_shapes,
        read_errors,
        read_error_tracebacks,
    ) = next(iter(dataset))

    assert dataset_paths[0].numpy().decode("utf-8") == str(image_path)
    image_height, image_width, image_channels = image_shapes[0].numpy()
    actual_image_data = actual_image_data_batch[
        0, :image_height, :image_width, :image_channels
    ].numpy()
    assert actual_image_data.dtype == np.float32
    np.testing.assert_allclose(actual_image_data, expected["image_data"])
    actual_crs, actual_transform = image_metadata.pop(image_path)
    assert actual_crs == expected["image_crs"]
    assert actual_transform == expected["image_transform"]
    assert image_metadata == {}
    assert read_errors[0].numpy() == b""
    assert read_error_tracebacks[0].numpy() == b""


def test_create_image_read_dataset_reports_read_errors(tmp_path):
    image_path = tmp_path / "missing-crs.tif"
    with rio.open(
        image_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=rio.uint8,
    ) as image_ds:
        image_ds.write(np.zeros((1, 2, 2), dtype=np.uint8))

    image_metadata = {}
    dataset = predicter._create_image_read_dataset(
        image_paths=[image_path],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=predicter.tf.float32,
        read_batch_size=1,
        image_metadata=image_metadata,
        image_metadata_lock=threading.Lock(),
    )

    (
        _,
        _,
        _,
        read_errors,
        read_error_tracebacks,
    ) = next(iter(dataset))

    assert b"Image has no proj" in read_errors[0].numpy()
    assert b"read_image" in read_error_tracebacks[0].numpy()
    assert image_metadata == {}


def test_create_image_read_dataset_batches_read_errors(tmp_path):
    image_path = tmp_path / "input.tif"
    image_data = np.arange(2 * 3, dtype=np.uint8).reshape((1, 2, 3))
    with rio.open(
        image_path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype=rio.uint8,
        crs="EPSG:31370",
    ) as image_ds:
        image_ds.write(image_data)

    error_image_path = tmp_path / "missing-crs.tif"
    with rio.open(
        error_image_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=rio.uint8,
    ) as image_ds:
        image_ds.write(np.zeros((1, 2, 2), dtype=np.uint8))

    image_metadata = {}
    dataset = predicter._create_image_read_dataset(
        image_paths=[image_path, error_image_path],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=predicter.tf.float32,
        read_batch_size=2,
        image_metadata=image_metadata,
        image_metadata_lock=threading.Lock(),
    )

    paths, images, image_shapes, errors, traces = next(iter(dataset))
    records = {
        paths[index].numpy().decode("utf-8"): (
            images[index],
            image_shapes[index].numpy(),
            errors[index].numpy(),
            traces[index].numpy(),
        )
        for index in range(len(paths))
    }

    image_tensor, image_shape, image_error, _ = records[str(image_path)]
    image_height, image_width, image_channels = image_shape
    np.testing.assert_allclose(
        image_tensor[:image_height, :image_width, :image_channels].numpy(),
        image_data.transpose((1, 2, 0)),
    )
    assert image_error == b""

    _, error_shape, read_error, read_error_traceback = records[str(error_image_path)]
    assert error_shape.tolist() == [1, 1, 1]
    assert b"Image has no proj" in read_error
    assert b"read_image" in read_error_traceback
    assert image_metadata.keys() == {image_path}


def test_create_image_read_dataset_empty_paths():
    dataset = predicter._create_image_read_dataset(
        image_paths=[],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=predicter.tf.float32,
        read_batch_size=1,
        image_metadata={},
        image_metadata_lock=threading.Lock(),
    )

    assert list(dataset.as_numpy_iterator()) == []


def test_predict_image_batch_preserves_order_for_different_shapes():
    inputs = predicter.keras.layers.Input(shape=(None, None, 1))
    outputs = predicter.keras.layers.Activation("linear")(inputs)
    model = predicter.keras.models.Model(inputs=inputs, outputs=outputs)
    image_tensors = [
        predicter.tf.ones((2, 3, 1), dtype=predicter.tf.float32),
        predicter.tf.fill((1, 2, 1), 5.0),
        predicter.tf.fill((2, 3, 1), 3.0),
    ]

    predictions = predicter._predict_image_batch(model, image_tensors)

    assert [prediction.shape for prediction in predictions] == [
        (2, 3, 1),
        (1, 2, 1),
        (2, 3, 1),
    ]
    np.testing.assert_array_equal(predictions[0], np.ones((2, 3, 1)))
    np.testing.assert_array_equal(predictions[1], np.full((1, 2, 1), 5.0))
    np.testing.assert_array_equal(predictions[2], np.full((2, 3, 1), 3.0))


def test_predict_image_batch_retries_smaller_batches_on_oom(monkeypatch):
    inputs = predicter.keras.layers.Input(shape=(2, 2, 1))
    outputs = predicter.keras.layers.Activation("linear")(inputs)
    model = predicter.keras.models.Model(inputs=inputs, outputs=outputs)
    original_predict_on_batch = model.predict_on_batch
    observed_batch_sizes = []

    def predict_with_batch_limit(batch):
        batch_size = batch.shape[0]
        observed_batch_sizes.append(batch_size)
        if batch_size > 1:
            raise predicter.tf.errors.ResourceExhaustedError(
                None, None, "simulated out-of-memory error"
            )
        return original_predict_on_batch(batch)

    monkeypatch.setattr(model, "predict_on_batch", predict_with_batch_limit)
    image_tensors = [predicter.tf.fill((2, 2, 1), float(value)) for value in (1, 2, 3)]

    predictions = predicter._predict_image_batch(model, image_tensors)

    assert observed_batch_sizes == [3, 1, 2, 1, 1]
    for prediction, value in zip(predictions, (1, 2, 3), strict=True):
        np.testing.assert_array_equal(prediction, np.full((2, 2, 1), value))


def test_create_live_image_read_dataset_reads_batches_and_preserves_metadata(
    monkeypatch, tmp_path
):
    image_layer = {"name": "live-source"}
    image_files = [
        {
            "path": tmp_path / "tile-1.tif",
            "bbox": (1.0, 10.0, 4.0, 14.0),
            "size": (3, 4),
        },
        {
            "path": tmp_path / "tile-2.tif",
            "bbox": (2.0, 20.0, 4.0, 22.0),
            "size": (2, 2),
        },
        {
            "path": tmp_path / "tile-error.tif",
            "bbox": (3.0, 30.0, 4.0, 31.0),
            "size": (1, 1),
        },
    ]
    read_requests = []

    def fake_load_image(*, bbox, size, image_layer, ssl_verify):
        read_requests.append((bbox, size, image_layer, ssl_verify))
        if bbox[0] == 3.0:
            raise RuntimeError("live image read failed")
        image_width, image_height = size
        fill_value = bbox[0] / 10
        return {
            "image_data": np.full(
                (image_height, image_width, 2),
                fill_value,
                dtype=np.float64,
            ),
            "image_crs": f"crs-{int(bbox[0])}",
            "image_transform": f"transform-{int(bbox[0])}",
        }

    monkeypatch.setattr(predicter, "load_image", fake_load_image)
    image_metadata = {}
    dataset = predicter._create_live_image_read_dataset(
        image_files=image_files,
        image_layer=image_layer,
        ssl_verify=False,
        image_dtype=predicter.tf.float32,
        nb_parallel_read=2,
        read_batch_size=2,
        image_metadata=image_metadata,
        image_metadata_lock=threading.Lock(),
    )

    records = {}
    batch_sizes = []
    for paths, images, shapes, errors, traces in dataset.as_numpy_iterator():
        batch_sizes.append(len(paths))
        for index, path in enumerate(paths):
            records[path.decode("utf-8")] = (
                images[index],
                shapes[index],
                errors[index],
                traces[index],
            )

    assert sorted(batch_sizes) == [1, 2]
    assert {
        (bbox, size, ssl_verify) for bbox, size, _, ssl_verify in read_requests
    } == {
        (tuple(image_file["bbox"]), tuple(image_file["size"]), False)
        for image_file in image_files
    }
    assert all(
        requested_layer is image_layer for _, _, requested_layer, _ in read_requests
    )

    for image_file in image_files[:2]:
        path = str(image_file["path"])
        image, shape, error, error_trace = records[path]
        width, height = image_file["size"]
        expected_value = image_file["bbox"][0] / 10
        assert shape.tolist() == [height, width, 2]
        assert image.dtype == np.float32
        np.testing.assert_array_equal(
            image[:height, :width, :],
            np.full((height, width, 2), expected_value, dtype=np.float32),
        )
        assert error == b""
        assert error_trace == b""
        assert image_metadata.pop(image_file["path"]) == (
            f"crs-{int(image_file['bbox'][0])}",
            f"transform-{int(image_file['bbox'][0])}",
        )

    _, error_shape, read_error, read_error_trace = records[str(image_files[2]["path"])]
    assert error_shape.tolist() == [1, 1, 1]
    assert b"live image read failed" in read_error
    assert b"_read_live_image_for_dataset" in read_error_trace
    assert image_metadata == {}


def test_create_live_image_read_dataset_empty_files():
    dataset = predicter._create_live_image_read_dataset(
        image_files=[],
        image_layer={},
        ssl_verify=True,
        image_dtype=predicter.tf.float32,
        nb_parallel_read=2,
        read_batch_size=2,
        image_metadata={},
        image_metadata_lock=threading.Lock(),
    )

    assert list(dataset.as_numpy_iterator()) == []

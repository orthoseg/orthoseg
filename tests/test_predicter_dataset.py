import threading

import numpy as np
import pyproj
import pytest
import rasterio as rio
import rasterio.transform as rio_transform

from orthoseg.helpers import _predicter_dataset


def test_create_image_read_dataset_matches_read_image(tmp_path):
    image_path = tmp_path / "input.tif"
    image_data = np.arange(3 * 4 * 5, dtype=np.uint8).reshape((3, 4, 5))
    image_transform = rio_transform.from_origin(100, 200, 2, 2)
    with rio.open(
        str(image_path),
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

    expected = _predicter_dataset.read_image(
        image_path=image_path,
        preprocess_input=preprocess_input,
    )
    image_metadata = {}
    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[image_path],
        projection_if_missing=None,
        preprocess_input=preprocess_input,
        image_dtype=_predicter_dataset.tf.float32,
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


def test_iter_image_read_results_preserves_success_and_errors(tmp_path):
    image_path = tmp_path / "input.tif"
    image_data = np.arange(6, dtype=np.uint8).reshape((1, 2, 3))
    image_transform = rio_transform.from_origin(100, 200, 2, 2)
    with rio.open(
        str(image_path),
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype=rio.uint8,
        crs="EPSG:31370",
        transform=image_transform,
    ) as image_ds:
        image_ds.write(image_data)

    error_image_path = tmp_path / "missing-crs.tif"
    with rio.open(
        str(error_image_path),
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=rio.uint8,
    ) as image_ds:
        image_ds.write(np.zeros((1, 2, 2), dtype=np.uint8))

    image_metadata = {}
    image_metadata_lock = threading.Lock()
    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[image_path, error_image_path],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=_predicter_dataset.tf.float32,
        read_batch_size=2,
        image_metadata=image_metadata,
        image_metadata_lock=image_metadata_lock,
    )

    results = dict(
        _predicter_dataset.iter_image_read_results(
            dataset,
            image_metadata=image_metadata,
            image_metadata_lock=image_metadata_lock,
        )
    )

    successful_result = results[image_path]
    assert successful_result["image_data"].shape == (2, 3, 1)
    np.testing.assert_array_equal(
        successful_result["image_data"].numpy(),
        image_data.transpose((1, 2, 0)),
    )
    assert successful_result["image_crs"] == rio.crs.CRS.from_epsg(31370)
    assert successful_result["image_transform"] == image_transform
    assert successful_result["read_error"] == b""
    assert successful_result["read_error_traceback"] == b""

    failed_result = results[error_image_path]
    assert failed_result["image_data"] is None
    assert b"Image has no proj" in failed_result["read_error"]
    assert b"read_image" in failed_result["read_error_traceback"]
    assert image_metadata == {}


def test_create_image_read_dataset_reports_read_errors(tmp_path):
    image_path = tmp_path / "missing-crs.tif"
    with rio.open(
        str(image_path),
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=rio.uint8,
    ) as image_ds:
        image_ds.write(np.zeros((1, 2, 2), dtype=np.uint8))

    image_metadata = {}
    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[image_path],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=_predicter_dataset.tf.float32,
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
        str(image_path),
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
        str(error_image_path),
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=rio.uint8,
    ) as image_ds:
        image_ds.write(np.zeros((1, 2, 2), dtype=np.uint8))

    image_metadata = {}
    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[image_path, error_image_path],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=_predicter_dataset.tf.float32,
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
    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=_predicter_dataset.tf.float32,
        read_batch_size=1,
        image_metadata={},
        image_metadata_lock=threading.Lock(),
    )

    assert list(dataset.as_numpy_iterator()) == []


def test_read_image_dataset_item_returns_metadata_and_shape(tmp_path):
    image_path = tmp_path / "input.tif"
    image_pixels = np.arange(6, dtype=np.uint8).reshape((1, 2, 3))
    image_transform = rio_transform.from_origin(100, 200, 2, 2)
    with rio.open(
        str(image_path),
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype=rio.uint8,
        crs="EPSG:31370",
        transform=image_transform,
    ) as image_ds:
        image_ds.write(image_pixels)

    image_metadata = {}
    item = _predicter_dataset._read_image_dataset_item(
        _predicter_dataset.tf.constant(str(image_path)),
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=_predicter_dataset.tf.float32,
        image_metadata=image_metadata,
        image_metadata_lock=threading.Lock(),
    )

    path, image, image_shape, read_error, read_error_traceback = item
    assert path.numpy().decode("utf-8") == str(image_path)
    assert image_shape.numpy().tolist() == [2, 3, 1]
    np.testing.assert_array_equal(
        image.numpy(), image_pixels.transpose((1, 2, 0)).astype(np.float32)
    )
    assert image_metadata == {
        image_path: (rio.crs.CRS.from_epsg(31370), image_transform)
    }
    assert read_error.numpy() == b""
    assert read_error_traceback.numpy() == b""


@pytest.mark.parametrize(
    "projection_if_missing",
    ["EPSG:31370", rio.crs.CRS.from_epsg(31370)],
)
def test_read_image_for_dataset_uses_projection_if_missing(
    tmp_path, projection_if_missing
):
    image_path = tmp_path / "missing-crs.tif"
    image_pixels = np.arange(6, dtype=np.uint8).reshape((1, 2, 3))
    image_transform = rio_transform.from_origin(100, 200, 2, 2)
    with rio.open(
        str(image_path),
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype=rio.uint8,
        transform=image_transform,
    ) as image_ds:
        image_ds.write(image_pixels)

    image_metadata = {}
    image, read_error, read_error_traceback = (
        _predicter_dataset._read_image_for_dataset(
            np.asarray(str(image_path).encode("utf-8")),
            projection_if_missing=projection_if_missing,
            preprocess_input=None,
            image_dtype=np.float32,
            image_metadata=image_metadata,
            image_metadata_lock=threading.Lock(),
        )
    )

    np.testing.assert_array_equal(
        image, image_pixels.transpose((1, 2, 0)).astype(np.float32)
    )
    assert image_metadata == {
        image_path: (rio.crs.CRS.from_epsg(31370), image_transform)
    }
    assert read_error == b""
    assert read_error_traceback == b""


def test_read_image_for_dataset_returns_read_error_details(monkeypatch):
    def fail_read_image(**_kwargs):
        raise RuntimeError("image decoder failed")

    monkeypatch.setattr(_predicter_dataset, "read_image", fail_read_image)
    image_metadata = {}

    image, read_error, read_error_traceback = (
        _predicter_dataset._read_image_for_dataset(
            np.asarray(b"missing.tif"),
            projection_if_missing=None,
            preprocess_input=None,
            image_dtype=np.float32,
            image_metadata=image_metadata,
            image_metadata_lock=threading.Lock(),
        )
    )

    assert image.shape == (1, 1, 1)
    assert image.dtype == np.float32
    assert b"image decoder failed" in read_error
    assert b"RuntimeError" in read_error_traceback.item()
    assert image_metadata == {}


def test_read_live_image_dataset_item_returns_metadata_and_shape(monkeypatch, tmp_path):
    image_path = tmp_path / "live.tif"
    image_layer = {"name": "live-source", "projection": "EPSG:31370"}
    projection_crs = pyproj.CRS.from_epsg(31370)
    image_transform = rio_transform.from_origin(100, 200, 2, 2)
    observed_arguments = {}

    def fake_load_image(**kwargs):
        observed_arguments.update(kwargs)
        return {
            "image_data": np.full((2, 3, 1), 0.5, dtype=np.float64),
            "image_crs": projection_crs,
            "image_transform": image_transform,
        }

    monkeypatch.setattr(_predicter_dataset, "load_image", fake_load_image)
    image_metadata = {}
    item = _predicter_dataset._read_live_image_dataset_item(
        _predicter_dataset.tf.constant(str(image_path)),
        _predicter_dataset.tf.constant(
            [1.0, 2.0, 3.0, 4.0], dtype=_predicter_dataset.tf.float64
        ),
        _predicter_dataset.tf.constant([3, 2], dtype=_predicter_dataset.tf.int32),
        image_layer=image_layer,
        ssl_verify=False,
        image_dtype=_predicter_dataset.tf.float32,
        projection_crs=projection_crs,
        projection_switch_axes=True,
        projection_wms_srs="epsg:31370",
        image_metadata=image_metadata,
        image_metadata_lock=threading.Lock(),
    )

    path, image, image_shape, read_error, read_error_traceback = item
    assert path.numpy().decode("utf-8") == str(image_path)
    assert image_shape.numpy().tolist() == [2, 3, 1]
    np.testing.assert_array_equal(image.numpy(), np.full((2, 3, 1), 0.5))
    assert observed_arguments["bbox"] == (1.0, 2.0, 3.0, 4.0)
    assert observed_arguments["size"] == (3, 2)
    assert observed_arguments["projection_crs"] is projection_crs
    assert observed_arguments["projection_switch_axes"] is True
    assert observed_arguments["projection_wms_srs"] == "epsg:31370"
    assert image_metadata == {image_path: (projection_crs, image_transform)}
    assert read_error.numpy() == b""
    assert read_error_traceback.numpy() == b""


def test_read_live_image_for_dataset_returns_read_error_details(monkeypatch):
    def fail_load_image(**_kwargs):
        raise RuntimeError("live image read failed")

    monkeypatch.setattr(_predicter_dataset, "load_image", fail_load_image)
    image_metadata = {}

    image, read_error, read_error_traceback = (
        _predicter_dataset._read_live_image_for_dataset(
            np.asarray(b"live.tif"),
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            np.asarray([3, 2]),
            image_layer={"name": "live-source"},
            ssl_verify=True,
            image_dtype=np.float32,
            projection_crs=pyproj.CRS.from_epsg(31370),
            projection_switch_axes=False,
            projection_wms_srs="epsg:31370",
            image_metadata=image_metadata,
            image_metadata_lock=threading.Lock(),
        )
    )

    assert image.shape == (1, 1, 1)
    assert image.dtype == np.float32
    assert b"live image read failed" in read_error
    assert b"RuntimeError" in read_error_traceback.item()
    assert image_metadata == {}


@pytest.mark.parametrize(
    ("switch_axes", "expected_switch_axes"),
    [(None, False), (True, True)],
)
def test_create_live_image_read_dataset_reads_batches_and_preserves_metadata(
    monkeypatch, tmp_path, switch_axes, expected_switch_axes
):
    image_layer = {
        "name": "live-source",
        "projection": "EPSG:31370",
        "switch_axes": switch_axes,
    }
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
    epsg_lookup_thread_ids = []
    caller_thread_id = threading.get_ident()
    get_epsg_code = _predicter_dataset.image_util._get_epsg_code

    def track_epsg_lookup(crs):
        epsg_lookup_thread_ids.append(threading.get_ident())
        return get_epsg_code(crs)

    def fake_load_image(
        *,
        bbox,
        size,
        image_layer,
        ssl_verify,
        projection_crs,
        projection_switch_axes,
        projection_wms_srs,
    ):
        read_requests.append(
            (
                bbox,
                size,
                image_layer,
                ssl_verify,
                projection_crs,
                projection_switch_axes,
                projection_wms_srs,
            )
        )
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

    monkeypatch.setattr(
        _predicter_dataset.image_util, "_get_epsg_code", track_epsg_lookup
    )
    monkeypatch.setattr(_predicter_dataset, "load_image", fake_load_image)
    image_metadata = {}
    dataset = _predicter_dataset.create_live_image_read_dataset(
        image_files=image_files,
        image_layer=image_layer,
        ssl_verify=False,
        image_dtype=_predicter_dataset.tf.float32,
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
        (bbox, size, ssl_verify) for bbox, size, _, ssl_verify, _, _, _ in read_requests
    } == {
        (tuple(image_file["bbox"]), tuple(image_file["size"]), False)
        for image_file in image_files
    }
    assert all(
        requested_layer is image_layer
        for _, _, requested_layer, _, _, _, _ in read_requests
    )
    assert all(
        requested_crs is read_requests[0][4]
        and requested_switch_axes is expected_switch_axes
        and requested_wms_srs == "epsg:31370"
        for (
            _,
            _,
            _,
            _,
            requested_crs,
            requested_switch_axes,
            requested_wms_srs,
        ) in read_requests
    )
    assert epsg_lookup_thread_ids == [caller_thread_id]

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
    dataset = _predicter_dataset.create_live_image_read_dataset(
        image_files=[],
        image_layer={},
        ssl_verify=True,
        image_dtype=_predicter_dataset.tf.float32,
        nb_parallel_read=2,
        read_batch_size=2,
        image_metadata={},
        image_metadata_lock=threading.Lock(),
    )

    assert list(dataset.as_numpy_iterator()) == []


def test_load_image_uses_prepared_projection_and_normalizes(monkeypatch):
    projection_crs = pyproj.CRS.from_epsg(31370)
    image_layer = {
        "projection": "EPSG:4326",
        "layersources": [{"url": "https://example.test/wms"}],
        "image_pixels_ignore_border": 2,
    }
    image_pixels = np.arange(12, dtype=np.uint8).reshape((3, 2, 2))
    image_transform = rio_transform.from_origin(100, 200, 2, 2)
    observed_arguments = {}

    def fake_load_image(**kwargs):
        observed_arguments.update(kwargs)
        return image_pixels, {"transform": image_transform}

    monkeypatch.setattr(_predicter_dataset.image_util, "load_image", fake_load_image)

    result = _predicter_dataset.load_image(
        bbox=(1.0, 2.0, 3.0, 4.0),
        size=(2, 2),
        image_layer=image_layer,
        ssl_verify=False,
        projection_crs=projection_crs,
        projection_switch_axes=False,
        projection_wms_srs="epsg:31370",
    )

    assert observed_arguments == {
        "layersources": image_layer["layersources"],
        "crs": projection_crs,
        "bbox": (1.0, 2.0, 3.0, 4.0),
        "size": (2, 2),
        "ssl_verify": False,
        "image_format": _predicter_dataset.image_util.FORMAT_JPEG,
        "image_pixels_ignore_border": 2,
        "switch_axes": False,
        "wms_srs": "epsg:31370",
    }
    np.testing.assert_allclose(
        result["image_data"],
        image_pixels.transpose((1, 2, 0)) / 255.0,
    )
    assert result["image_crs"] is projection_crs
    assert result["image_transform"] == image_transform


def test_load_image_uses_layer_projection_when_not_precomputed(monkeypatch):
    image_layer = {
        "projection": "EPSG:4326",
        "switch_axes": True,
        "layersources": [{"url": "https://example.test/wms"}],
        "image_format": "image/png",
        "image_pixels_ignore_border": 0,
    }
    image_pixels = np.arange(4, dtype=np.uint8).reshape((1, 2, 2))
    image_transform = rio_transform.from_origin(100, 200, 2, 2)
    observed_arguments = {}

    def fake_load_image(**kwargs):
        observed_arguments.update(kwargs)
        return image_pixels, {"transform": image_transform}

    monkeypatch.setattr(_predicter_dataset.image_util, "load_image", fake_load_image)

    result = _predicter_dataset.load_image(
        bbox=(1.0, 2.0, 3.0, 4.0),
        size=(2, 2),
        image_layer=image_layer,
    )

    assert observed_arguments["crs"] == pyproj.CRS.from_epsg(4326)
    assert observed_arguments["switch_axes"] is True
    assert observed_arguments["wms_srs"] is None
    assert observed_arguments["image_format"] == "image/png"
    np.testing.assert_allclose(
        result["image_data"],
        image_pixels.transpose((1, 2, 0)) / 255.0,
    )
    assert result["image_crs"] == pyproj.CRS.from_epsg(4326)
    assert result["image_transform"] == image_transform

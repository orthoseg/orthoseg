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
    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[image_path],
        projection_if_missing=None,
        preprocess_input=preprocess_input,
        image_dtype=np.float32,
        max_parallel_reads=1,
    )

    actual_path, read_result = next(iter(dataset))
    assert actual_path == image_path
    actual_image_data = read_result["image_data"]
    assert actual_image_data.dtype == np.float32
    np.testing.assert_allclose(actual_image_data, expected["image_data"])
    assert read_result["image_crs"] == expected["image_crs"]
    assert read_result["image_transform"] == expected["image_transform"]
    assert read_result["read_error"] == b""
    assert read_result["read_error_traceback"] == b""


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

    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[image_path, error_image_path],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=np.float32,
        max_parallel_reads=2,
    )

    results = dict(_predicter_dataset.iter_image_read_results(dataset))

    successful_result = results[image_path]
    assert successful_result["image_data"].shape == (2, 3, 1)
    np.testing.assert_array_equal(
        successful_result["image_data"], image_data.transpose((1, 2, 0))
    )
    assert successful_result["image_crs"] == rio.crs.CRS.from_epsg(31370)
    assert successful_result["image_transform"] == image_transform
    assert successful_result["read_error"] == b""
    assert successful_result["read_error_traceback"] == b""

    failed_result = results[error_image_path]
    assert failed_result["image_data"] is None
    assert b"Image has no proj" in failed_result["read_error"]
    assert b"read_image" in failed_result["read_error_traceback"]


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

    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[tmp_path / "missing-crs.tif"],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=np.float32,
        max_parallel_reads=1,
    )

    _, read_result = next(iter(dataset))
    assert b"Image has no proj" in read_result["read_error"]
    assert b"read_image" in read_result["read_error_traceback"]


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

    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[image_path, error_image_path],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=np.float32,
        max_parallel_reads=2,
    )

    records = dict(dataset)
    success = records[image_path]
    np.testing.assert_allclose(success["image_data"], image_data.transpose((1, 2, 0)))
    assert success["read_error"] == b""

    failure = records[error_image_path]
    assert failure["image_data"] is None
    assert b"Image has no proj" in failure["read_error"]
    assert b"read_image" in failure["read_error_traceback"]


def test_create_image_read_dataset_empty_paths():
    dataset = _predicter_dataset.create_image_read_dataset(
        image_paths=[],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=np.float32,
        max_parallel_reads=1,
    )

    assert list(dataset) == []


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

    path, result = _predicter_dataset._read_image_dataset_item(
        image_path,
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=np.float32,
    )

    assert path == image_path
    image = result["image_data"]
    assert image.shape == (2, 3, 1)
    np.testing.assert_array_equal(
        image, image_pixels.transpose((1, 2, 0)).astype(np.float32)
    )
    assert result["image_crs"] == rio.crs.CRS.from_epsg(31370)
    assert result["image_transform"] == image_transform
    assert result["read_error"] == b""
    assert result["read_error_traceback"] == b""


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

    _, result = _predicter_dataset._read_image_for_dataset(
        image_path,
        projection_if_missing=projection_if_missing,
        preprocess_input=None,
        image_dtype=np.float32,
    )

    np.testing.assert_array_equal(
        result["image_data"], image_pixels.transpose((1, 2, 0)).astype(np.float32)
    )
    assert result["image_crs"] == rio.crs.CRS.from_epsg(31370)
    assert result["image_transform"] == image_transform
    assert result["read_error"] == b""
    assert result["read_error_traceback"] == b""


def test_read_image_for_dataset_returns_read_error_details(monkeypatch):
    def fail_read_image(**_kwargs):
        raise RuntimeError("image decoder failed")

    monkeypatch.setattr(_predicter_dataset, "read_image", fail_read_image)
    _, result = _predicter_dataset._read_image_for_dataset(
        "missing.tif",
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=np.float32,
    )

    assert result["image_data"] is None
    assert b"image decoder failed" in result["read_error"]
    assert b"RuntimeError" in result["read_error_traceback"]


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
    path, result = _predicter_dataset._read_live_image_dataset_item(
        {
            "path": image_path,
            "bbox": (1.0, 2.0, 3.0, 4.0),
            "size": (3, 2),
        },
        image_layer=image_layer,
        ssl_verify=False,
        image_dtype=np.float32,
        projection_crs=projection_crs,
        projection_switch_axes=True,
        projection_wms_srs="epsg:31370",
    )

    assert path == image_path
    np.testing.assert_array_equal(result["image_data"], np.full((2, 3, 1), 0.5))
    assert observed_arguments["bbox"] == (1.0, 2.0, 3.0, 4.0)
    assert observed_arguments["size"] == (3, 2)
    assert observed_arguments["projection_crs"] is projection_crs
    assert observed_arguments["projection_switch_axes"] is True
    assert observed_arguments["projection_wms_srs"] == "epsg:31370"
    assert result["image_crs"] is projection_crs
    assert result["image_transform"] == image_transform
    assert result["read_error"] == b""
    assert result["read_error_traceback"] == b""


def test_read_live_image_for_dataset_returns_read_error_details(monkeypatch):
    def fail_load_image(**_kwargs):
        raise RuntimeError("live image read failed")

    monkeypatch.setattr(_predicter_dataset, "load_image", fail_load_image)
    _, result = _predicter_dataset._read_live_image_for_dataset(
        {
            "path": "live.tif",
            "bbox": (1.0, 2.0, 3.0, 4.0),
            "size": (3, 2),
        },
        image_layer={"name": "live-source"},
        ssl_verify=True,
        image_dtype=np.float32,
        projection_crs=pyproj.CRS.from_epsg(31370),
        projection_switch_axes=False,
        projection_wms_srs="epsg:31370",
    )

    assert result["image_data"] is None
    assert b"live image read failed" in result["read_error"]
    assert b"RuntimeError" in result["read_error_traceback"]


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
    dataset = _predicter_dataset.create_live_image_read_dataset(
        image_files=image_files,
        image_layer=image_layer,
        ssl_verify=False,
        image_dtype=np.float32,
        max_parallel_reads=2,
    )

    records = dict(dataset)

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
        result = records[image_file["path"]]
        image = result["image_data"]
        width, height = image_file["size"]
        expected_value = image_file["bbox"][0] / 10
        assert image.shape == (height, width, 2)
        assert image.dtype == np.float32
        np.testing.assert_array_equal(
            image,
            np.full((height, width, 2), expected_value, dtype=np.float32),
        )
        assert result["read_error"] == b""
        assert result["read_error_traceback"] == b""
        assert result["image_crs"] == f"crs-{int(image_file['bbox'][0])}"
        assert result["image_transform"] == f"transform-{int(image_file['bbox'][0])}"

    error_result = records[image_files[2]["path"]]
    assert error_result["image_data"] is None
    assert b"live image read failed" in error_result["read_error"]
    assert b"_read_live_image_for_dataset" in error_result["read_error_traceback"]


def test_create_image_read_dataset_limits_parallel_reads(monkeypatch, tmp_path):
    image_paths = [tmp_path / "first.tif", tmp_path / "second.tif"]
    barrier = threading.Barrier(2)
    active_reads = 0
    max_active_reads = 0
    active_reads_lock = threading.Lock()

    def fake_read_image(*, image_path, **_kwargs):
        nonlocal active_reads, max_active_reads
        with active_reads_lock:
            active_reads += 1
            max_active_reads = max(max_active_reads, active_reads)
        try:
            barrier.wait(timeout=2)
            return {
                "image_data": np.zeros((1, 1, 1), dtype=np.float32),
                "image_crs": None,
                "image_transform": None,
                "image_path": image_path,
            }
        finally:
            with active_reads_lock:
                active_reads -= 1

    monkeypatch.setattr(_predicter_dataset, "read_image", fake_read_image)
    results = list(
        _predicter_dataset.create_image_read_dataset(
            image_paths=image_paths,
            projection_if_missing=None,
            preprocess_input=None,
            image_dtype=np.float32,
            max_parallel_reads=2,
        )
    )

    assert len(results) == 2
    assert max_active_reads == 2


def test_create_image_read_dataset_reads_ahead_while_consumer_waits(
    monkeypatch, tmp_path
):
    image_paths = [tmp_path / f"tile-{index}.tif" for index in range(5)]
    later_read_started = threading.Event()

    def fake_read_image(*, image_path, **_kwargs):
        if image_path == image_paths[2]:
            later_read_started.set()
        return {
            "image_data": np.zeros((1, 1, 1), dtype=np.float32),
            "image_crs": None,
            "image_transform": None,
            "image_path": image_path,
        }

    monkeypatch.setattr(_predicter_dataset, "read_image", fake_read_image)
    read_results = _predicter_dataset.create_image_read_dataset(
        image_paths=image_paths,
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=np.float32,
        max_parallel_reads=2,
    )

    try:
        next(read_results)
        assert later_read_started.wait(timeout=2)
    finally:
        read_results.close()


def test_create_live_image_read_dataset_empty_files():
    dataset = _predicter_dataset.create_live_image_read_dataset(
        image_files=[],
        image_layer={},
        ssl_verify=True,
        image_dtype=np.float32,
        max_parallel_reads=2,
    )

    assert list(dataset) == []


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

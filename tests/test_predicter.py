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
        nb_parallel_read=1,
        image_metadata=image_metadata,
        image_metadata_lock=threading.Lock(),
    )

    (
        dataset_path,
        actual_image_data,
        read_error,
        read_error_traceback,
    ) = next(iter(dataset.as_numpy_iterator()))

    assert dataset_path.decode("utf-8") == str(image_path)
    assert actual_image_data.dtype == np.float32
    np.testing.assert_allclose(actual_image_data, expected["image_data"])
    actual_crs, actual_transform = image_metadata.pop(image_path)
    assert actual_crs == expected["image_crs"]
    assert actual_transform == expected["image_transform"]
    assert image_metadata == {}
    assert read_error == b""
    assert read_error_traceback == b""


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
        nb_parallel_read=1,
        image_metadata=image_metadata,
        image_metadata_lock=threading.Lock(),
    )

    (
        _,
        _,
        read_error,
        read_error_traceback,
    ) = next(iter(dataset.as_numpy_iterator()))

    assert b"Image has no proj" in read_error
    assert b"read_image" in read_error_traceback
    assert image_metadata == {}


def test_create_image_read_dataset_empty_paths():
    dataset = predicter._create_image_read_dataset(
        image_paths=[],
        projection_if_missing=None,
        preprocess_input=None,
        image_dtype=predicter.tf.float32,
        nb_parallel_read=1,
        image_metadata={},
        image_metadata_lock=threading.Lock(),
    )

    assert list(dataset.as_numpy_iterator()) == []

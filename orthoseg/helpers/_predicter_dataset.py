"""Internal helpers for reading prediction images with tf.data."""

import logging
import traceback
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import rasterio as rio
import rasterio.crs as rio_crs
import rasterio.plot as rio_plot
import tensorflow as tf

from orthoseg.util import image_util

logger = logging.getLogger(__name__)


def create_image_read_dataset(
    image_paths: list[Path],
    *,
    projection_if_missing: str | rio_crs.CRS | None,
    preprocess_input: Callable | None,
    image_dtype: tf.dtypes.DType,
    read_batch_size: int,
    image_metadata: dict[Path, tuple[Any, Any]],
    image_metadata_lock: Any,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant([str(path) for path in image_paths], dtype=tf.string)
    )

    dataset = dataset.map(
        partial(
            _read_image_dataset_item,
            projection_if_missing=projection_if_missing,
            preprocess_input=preprocess_input,
            image_dtype=image_dtype,
            image_metadata=image_metadata,
            image_metadata_lock=image_metadata_lock,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return _batch_image_read_dataset(
        dataset,
        image_dtype=image_dtype,
        read_batch_size=read_batch_size,
    )


def create_live_image_read_dataset(
    image_files: list[dict[str, Any]],
    *,
    image_layer: dict[str, Any],
    ssl_verify: bool | str,
    image_dtype: tf.dtypes.DType,
    nb_parallel_read: int,
    read_batch_size: int,
    image_metadata: dict[Path, tuple[Any, Any]],
    image_metadata_lock: Any,
) -> tf.data.Dataset:
    image_paths = [str(image_file["path"]) for image_file in image_files]
    image_bboxes = np.asarray(
        [image_file["bbox"] for image_file in image_files],
        dtype=np.float64,
    ).reshape((-1, 4))
    image_sizes = np.asarray(
        [image_file["size"] for image_file in image_files],
        dtype=np.int32,
    ).reshape((-1, 2))
    if not image_files:
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant([], dtype=tf.string),
                tf.zeros((0, 1, 1, 1), dtype=image_dtype),
                tf.zeros((0, 3), dtype=tf.int32),
                tf.constant([], dtype=tf.string),
                tf.constant([], dtype=tf.string),
            )
        )
        return _batch_image_read_dataset(
            dataset,
            image_dtype=image_dtype,
            read_batch_size=read_batch_size,
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(image_paths, dtype=tf.string),
            tf.constant(image_bboxes, dtype=tf.float64),
            tf.constant(image_sizes, dtype=tf.int32),
        )
    )

    projection_crs = pyproj.CRS.from_user_input(image_layer["projection"])
    projection_switch_axes = image_layer.get("switch_axes")
    if projection_switch_axes is None:
        projection_switch_axes = image_util.has_switched_axes(projection_crs)
    projection_wms_srs = f"epsg:{image_util._get_epsg_code(projection_crs)}"

    dataset = dataset.map(
        partial(
            _read_live_image_dataset_item,
            image_layer=image_layer,
            ssl_verify=ssl_verify,
            image_dtype=image_dtype,
            projection_crs=projection_crs,
            projection_switch_axes=projection_switch_axes,
            projection_wms_srs=projection_wms_srs,
            image_metadata=image_metadata,
            image_metadata_lock=image_metadata_lock,
        ),
        num_parallel_calls=nb_parallel_read,
        deterministic=False,
    )
    return _batch_image_read_dataset(
        dataset,
        image_dtype=image_dtype,
        read_batch_size=read_batch_size,
    )


def read_image(
    image_path: Path,
    projection_if_missing: str | rio_crs.CRS | None = None,
    preprocess_input: Callable | None = None,
) -> dict:
    """Read image file.

    Args:
        image_path: file path.
        projection_if_missing: the projection to use if the
            image to read does not contain projection information. Defaults to None.
        preprocess_input: the preprocessing function to
            apply to the image being read. Defaults to None.

    Returns:
        the data read from the image file.
    """
    # Read input file
    # Because sometimes a read seems to fail, retry up to 3 times...
    retry_count = 0
    while True:
        try:
            with rio.open(str(image_path)) as image_ds:
                # Read geo info
                image_crs = image_ds.profile["crs"]
                image_transform = image_ds.transform

                # Read pixels
                image_data = image_ds.read()

            # change from (channels, width, height) to
            # (width, height, channels) + normalize to between 0 and 1
            image_data = rio_plot.reshape_as_image(image_data)
            if preprocess_input is not None:
                image_data = preprocess_input(image_data)

            # Read worked, so jump out of the loop...
            break
        except Exception as ex:  # pragma: no cover
            retry_count += 1
            logger.warning(f"Read failed, retry nb {retry_count} for {image_path}")
            if retry_count >= 3:
                message = f"Read failed {retry_count} times for {image_path}: {ex}"
                logger.error(message)
                raise RuntimeError(message) from ex

    # The read was successfull, now check if there was a projection in the
    # file and/or if one was provided
    if image_crs is None:
        if projection_if_missing is not None:
            if isinstance(projection_if_missing, rio_crs.CRS):
                image_crs = projection_if_missing
            else:
                image_crs = rio_crs.CRS.from_user_input(projection_if_missing)
        else:
            message = (
                f"Image has no proj and projection_if_missing is None: {image_path}"
            )
            logger.error(message)
            raise ValueError(message)

    # Now return the result
    image = {
        "image_data": image_data,
        "image_crs": image_crs,
        "image_transform": image_transform,
        "image_path": image_path,
    }

    return image


def load_image(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    image_layer: dict,
    ssl_verify: bool | str = True,
    *,
    projection_crs: pyproj.CRS | None = None,
    projection_switch_axes: bool | None = None,
    projection_wms_srs: str | None = None,
) -> dict:
    """Load an image from the image_layer specified.

    Args:
        bbox: bounding box of the image to load.
        size: size of the image to load.
        image_layer: layer configuration to load the image.
        ssl_verify: True to use the default certificate bundle
            as installed on your system. False disables certificate validation
            (NOT recommended!). If a path to a certificate bundle file (.pem) is passed,
            this will be used. In corporate networks using a proxy server this is often
            needed to avoid CERTIFICATE_VERIFY_FAILED errors. Defaults to True.
        projection_crs: CRS prepared before parallel live-image reads.
        projection_switch_axes: Whether coordinates need their axes switched.
        projection_wms_srs: Precomputed WMS CRS identifier.

    Returns:
        the image and its properties.
    """
    # Load image
    crs = (
        projection_crs
        if projection_crs is not None
        else pyproj.CRS.from_user_input(image_layer["projection"])
    )
    if projection_switch_axes is None:
        projection_switch_axes = image_layer.get("switch_axes")
    image_data, profile = image_util.load_image(
        layersources=image_layer["layersources"],
        crs=crs,
        bbox=bbox,
        size=size,
        ssl_verify=ssl_verify,
        image_format=image_layer.get("image_format", image_util.FORMAT_JPEG),
        # transparent=transparent,
        image_pixels_ignore_border=image_layer["image_pixels_ignore_border"],
        switch_axes=projection_switch_axes,
        wms_srs=projection_wms_srs,
    )

    # change from (channels, width, height) to
    # (width, height, channels) + normalize to between 0 and 1
    image_data = rio_plot.reshape_as_image(image_data)
    image_data = image_data / 255.0

    # Now return the result
    image = {
        "image_data": image_data,
        "image_crs": crs,
        "image_transform": profile["transform"],
    }

    return image


def _read_image_dataset_item(
    image_path: tf.Tensor,
    *,
    projection_if_missing: str | rio_crs.CRS | None,
    preprocess_input: Callable | None,
    image_dtype: tf.dtypes.DType,
    image_metadata: dict[Path, tuple[Any, Any]],
    image_metadata_lock: Any,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    read_outputs = tf.numpy_function(
        func=lambda path_value: _read_image_for_dataset(
            path_value,
            projection_if_missing=projection_if_missing,
            preprocess_input=preprocess_input,
            image_dtype=image_dtype.as_numpy_dtype,
            image_metadata=image_metadata,
            image_metadata_lock=image_metadata_lock,
        ),
        inp=[image_path],
        Tout=(image_dtype, tf.string, tf.string),
    )
    image_data, read_error, read_error_traceback = read_outputs
    image_data.set_shape([None, None, None])
    return (
        image_path,
        image_data,
        tf.shape(image_data, out_type=tf.int32),
        read_error,
        read_error_traceback,
    )


def _read_live_image_dataset_item(
    image_path: tf.Tensor,
    bbox: tf.Tensor,
    size: tf.Tensor,
    *,
    image_layer: dict[str, Any],
    ssl_verify: bool | str,
    image_dtype: tf.dtypes.DType,
    projection_crs: pyproj.CRS,
    projection_switch_axes: bool,
    projection_wms_srs: str,
    image_metadata: dict[Path, tuple[Any, Any]],
    image_metadata_lock: Any,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    read_outputs = tf.numpy_function(
        func=lambda path_value, bbox_value, size_value: _read_live_image_for_dataset(
            path_value,
            bbox_value,
            size_value,
            image_layer=image_layer,
            ssl_verify=ssl_verify,
            image_dtype=image_dtype.as_numpy_dtype,
            projection_crs=projection_crs,
            projection_switch_axes=projection_switch_axes,
            projection_wms_srs=projection_wms_srs,
            image_metadata=image_metadata,
            image_metadata_lock=image_metadata_lock,
        ),
        inp=[image_path, bbox, size],
        Tout=(image_dtype, tf.string, tf.string),
    )
    image_data, read_error, read_error_traceback = read_outputs
    image_data.set_shape([None, None, None])
    return (
        image_path,
        image_data,
        tf.shape(image_data, out_type=tf.int32),
        read_error,
        read_error_traceback,
    )


def _batch_image_read_dataset(
    dataset: tf.data.Dataset,
    *,
    image_dtype: tf.dtypes.DType,
    read_batch_size: int,
) -> tf.data.Dataset:
    return dataset.padded_batch(
        read_batch_size,
        padded_shapes=([], [None, None, None], [3], [], []),
        padding_values=(
            tf.constant(b"", dtype=tf.string),
            tf.constant(0, dtype=image_dtype),
            tf.constant(0, dtype=tf.int32),
            tf.constant(b"", dtype=tf.string),
            tf.constant(b"", dtype=tf.string),
        ),
    ).prefetch(tf.data.AUTOTUNE)


def iter_image_read_results(
    dataset: tf.data.Dataset,
    *,
    image_metadata: dict[Path, tuple[Any, Any]],
    image_metadata_lock: Any,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    for (
        image_paths_batch,
        image_data_batch,
        image_shapes_batch,
        read_errors_batch,
        read_error_tracebacks_batch,
    ) in dataset:
        image_path_values = image_paths_batch.numpy()
        image_shapes = image_shapes_batch.numpy()
        read_errors = read_errors_batch.numpy()
        read_error_tracebacks = read_error_tracebacks_batch.numpy()
        for batch_index, image_path_value in enumerate(image_path_values):
            image_path = Path(image_path_value.decode("utf-8"))
            read_error = read_errors[batch_index]
            read_result: dict[str, Any] = {
                "image_data": None,
                "read_error": read_error,
                "read_error_traceback": read_error_tracebacks[batch_index],
            }
            if not read_error:
                image_height, image_width, image_channels = (
                    int(value) for value in image_shapes[batch_index]
                )
                read_result["image_data"] = image_data_batch[
                    batch_index,
                    :image_height,
                    :image_width,
                    :image_channels,
                ]
                with image_metadata_lock:
                    (
                        read_result["image_crs"],
                        read_result["image_transform"],
                    ) = image_metadata.pop(image_path)

            yield image_path, read_result


def _image_read_error_placeholder(image_dtype: Any) -> np.ndarray:
    # TensorFlow 2.21 segfaults when tf.numpy_function returns a zero-sized array.
    return np.zeros((1, 1, 1), dtype=image_dtype)


def _read_live_image_for_dataset(
    image_path_value: np.ndarray,
    bbox_value: np.ndarray,
    size_value: np.ndarray,
    *,
    image_layer: dict[str, Any],
    ssl_verify: bool | str,
    image_dtype: Any,
    projection_crs: pyproj.CRS,
    projection_switch_axes: bool,
    projection_wms_srs: str,
    image_metadata: dict[Path, tuple[Any, Any]],
    image_metadata_lock: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_path = np.asarray(image_path_value).item()
    if isinstance(image_path, bytes):
        image_path = image_path.decode("utf-8")
    image_path = Path(image_path)

    bbox_values = np.asarray(bbox_value).tolist()
    bbox = (
        float(bbox_values[0]),
        float(bbox_values[1]),
        float(bbox_values[2]),
        float(bbox_values[3]),
    )
    size_values = np.asarray(size_value).tolist()
    size = (int(size_values[0]), int(size_values[1]))

    try:
        image_info = load_image(
            bbox=bbox,
            size=size,
            image_layer=image_layer,
            ssl_verify=ssl_verify,
            projection_crs=projection_crs,
            projection_switch_axes=projection_switch_axes,
            projection_wms_srs=projection_wms_srs,
        )
    except Exception as ex:
        return (
            _image_read_error_placeholder(image_dtype),
            np.asarray(str(ex).encode("utf-8", errors="replace"), dtype=np.bytes_),
            np.asarray(
                traceback.format_exc().encode("utf-8", errors="replace"),
                dtype=np.bytes_,
            ),
        )

    image_data = np.asarray(image_info["image_data"], dtype=image_dtype)
    with image_metadata_lock:
        image_metadata[image_path] = (
            image_info["image_crs"],
            image_info["image_transform"],
        )

    return (
        image_data,
        np.asarray(b"", dtype=np.bytes_),
        np.asarray(b"", dtype=np.bytes_),
    )


def _read_image_for_dataset(
    image_path_value: np.ndarray,
    *,
    projection_if_missing: str | rio_crs.CRS | None,
    preprocess_input: Callable | None,
    image_dtype: Any,
    image_metadata: dict[Path, tuple[Any, Any]],
    image_metadata_lock: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_path = np.asarray(image_path_value).item()
    if isinstance(image_path, bytes):
        image_path = image_path.decode("utf-8")

    try:
        image_info = read_image(
            image_path=Path(image_path),
            projection_if_missing=projection_if_missing,
            preprocess_input=preprocess_input,
        )
    except Exception as ex:
        return (
            _image_read_error_placeholder(image_dtype),
            np.asarray(str(ex).encode("utf-8", errors="replace"), dtype=np.bytes_),
            np.asarray(
                traceback.format_exc().encode("utf-8", errors="replace"),
                dtype=np.bytes_,
            ),
        )

    image_data = np.asarray(image_info["image_data"], dtype=image_dtype)
    with image_metadata_lock:
        image_metadata[Path(image_path)] = (
            image_info["image_crs"],
            image_info["image_transform"],
        )

    return (
        image_data,
        np.asarray(b"", dtype=np.bytes_),
        np.asarray(b"", dtype=np.bytes_),
    )

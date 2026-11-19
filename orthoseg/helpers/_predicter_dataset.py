"""Internal helpers for reading prediction images concurrently.

keras and tensorflow are optional dependencies only used for prefetching image tensors
to improve throughput. If they are not available, the prefetching will be a no-op.
"""

import logging
import queue
import threading
import traceback
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyproj
import rasterio as rio
import rasterio.crs as rio_crs
import rasterio.plot as rio_plot

from orthoseg.util import image_util

logger = logging.getLogger(__name__)


def create_image_read_dataset(
    image_paths: list[Path],
    *,
    projection_if_missing: str | rio_crs.CRS | None,
    preprocess_input: Callable | None,
    image_dtype: Any,
    max_parallel_reads: int,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    def read_item(image_path: Path) -> tuple[Path, dict[str, Any]]:
        return _read_image_for_dataset(
            image_path,
            projection_if_missing=projection_if_missing,
            preprocess_input=preprocess_input,
            image_dtype=image_dtype,
        )

    return _iter_parallel_reads(image_paths, read_item, max_parallel_reads)


def create_live_image_read_dataset(
    image_files: list[dict[str, Any]],
    *,
    image_layer: dict[str, Any],
    ssl_verify: bool | str,
    image_dtype: Any,
    max_parallel_reads: int,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    if not image_files:
        return iter(())
    projection_crs = pyproj.CRS.from_user_input(image_layer["projection"])
    projection_switch_axes = image_layer.get("switch_axes")
    if projection_switch_axes is None:
        projection_switch_axes = image_util.has_switched_axes(projection_crs)
    projection_wms_srs = f"epsg:{image_util._get_epsg_code(projection_crs)}"

    def read_item(image_file: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        return _read_live_image_for_dataset(
            image_file,
            image_layer=image_layer,
            ssl_verify=ssl_verify,
            image_dtype=image_dtype,
            projection_crs=projection_crs,
            projection_switch_axes=projection_switch_axes,
            projection_wms_srs=projection_wms_srs,
        )

    return _iter_parallel_reads(image_files, read_item, max_parallel_reads)


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
    image_path: Path,
    *,
    projection_if_missing: str | rio_crs.CRS | None,
    preprocess_input: Callable | None,
    image_dtype: Any,
) -> tuple[Path, dict[str, Any]]:
    return _read_image_for_dataset(
        image_path,
        projection_if_missing=projection_if_missing,
        preprocess_input=preprocess_input,
        image_dtype=image_dtype,
    )


def _read_live_image_dataset_item(
    image_file: dict[str, Any],
    *,
    image_layer: dict[str, Any],
    ssl_verify: bool | str,
    image_dtype: Any,
    projection_crs: pyproj.CRS,
    projection_switch_axes: bool,
    projection_wms_srs: str,
) -> tuple[Path, dict[str, Any]]:
    return _read_live_image_for_dataset(
        image_file,
        image_layer=image_layer,
        ssl_verify=ssl_verify,
        image_dtype=image_dtype,
        projection_crs=projection_crs,
        projection_switch_axes=projection_switch_axes,
        projection_wms_srs=projection_wms_srs,
    )


def iter_image_read_results(
    read_results: Iterable[tuple[Path, dict[str, Any]]],
) -> Iterator[tuple[Path, dict[str, Any]]]:
    yield from read_results


def prefetch_image_read_results(
    read_results: Iterator[tuple[Path, dict[str, Any]]],
    image_dtype: Any,
    *,
    prefetch_buffer_size: int = -1,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Prefetch image tensors with a configurable buffer.

    Note: this will only work if keras and tensorflow are available. If not, this is a
    no-op and will yield the read results as-is.

    Args:
        read_results: An iterator of tuples containing the image path and the read
            result dictionary.
        image_dtype: The data type of the image tensors.
        prefetch_buffer_size: The size of the prefetch buffer. Defaults to -1, which
            uses tf.data.AUTOTUNE.

    Yields:
        Tuples of the image path and the read result dictionary, with the image data
        prefetched as a tensor.
    """
    from orthoseg._compat import (  # noqa: PLC0415
        _module_is_available,
        get_keras_backend,
    )

    if not _module_is_available("tensorflow"):
        yield from read_results
        return

    try:
        from orthoseg._keras import keras  # noqa: PLC0415
    except ModuleNotFoundError as ex:
        if ex.name not in {"keras", "tensorflow"}:
            raise
        yield from read_results
        return

    if get_keras_backend(keras) != "tensorflow":
        yield from read_results
        return

    try:
        import tensorflow as tf  # noqa: PLC0415
    except ModuleNotFoundError as ex:
        if ex.name != "tensorflow":
            raise
        yield from read_results
        return

    tensor_dtype = tf.as_dtype(image_dtype)
    metadata_queue: queue.Queue[tuple[Path, dict[str, Any], bool]] = queue.Queue()

    def generate_image_data() -> Iterator[np.ndarray]:
        for image_path, read_result in read_results:
            read_result = dict(read_result)
            image_data = read_result.pop("image_data", None)
            has_image_data = image_data is not None
            metadata_queue.put((image_path, read_result, has_image_data))
            if not has_image_data:
                image_data = np.empty((0, 0, 0), dtype=tensor_dtype.as_numpy_dtype)
            yield image_data

    if prefetch_buffer_size < 1:
        prefetch_buffer_size = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_generator(
        generate_image_data,
        output_signature=tf.TensorSpec(
            shape=(None, None, None),
            dtype=tensor_dtype,
        ),
    ).prefetch(prefetch_buffer_size)

    for image_data in dataset:
        image_path, read_result, has_image_data = metadata_queue.get()
        read_result["image_data"] = image_data if has_image_data else None
        yield image_path, read_result


def _iter_parallel_reads(
    items: Iterable[Any],
    read_item: Callable[[Any], tuple[Path, dict[str, Any]]],
    max_parallel_reads: int,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    if max_parallel_reads < 1:
        raise ValueError("max_parallel_reads must be at least 1")

    def generate() -> Iterator[tuple[Path, dict[str, Any]]]:
        results_queue: queue.Queue[object] = queue.Queue(maxsize=max_parallel_reads)
        stop_producer = threading.Event()
        finished = object()

        def enqueue(result: object) -> bool:
            while not stop_producer.is_set():
                try:
                    results_queue.put(result, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                item_iterator = iter(items)
                with ThreadPoolExecutor(max_workers=max_parallel_reads) as executor:
                    pending: set[Future[tuple[Path, dict[str, Any]]]] = set()
                    for _ in range(max_parallel_reads):
                        try:
                            item = next(item_iterator)
                        except StopIteration:
                            break
                        pending.add(executor.submit(read_item, item))

                    while pending and not stop_producer.is_set():
                        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            try:
                                result: object = future.result()
                            except BaseException as ex:
                                result = ex
                            if not enqueue(result):
                                return
                            try:
                                item = next(item_iterator)
                            except StopIteration:
                                continue
                            pending.add(executor.submit(read_item, item))
            except BaseException as ex:
                enqueue(ex)
            finally:
                enqueue(finished)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            while True:
                result = results_queue.get()
                if result is finished:
                    break
                if isinstance(result, BaseException):
                    raise result
                yield cast("tuple[Path, dict[str, Any]]", result)
        finally:
            stop_producer.set()
            producer.join()

    return generate()


def _numpy_dtype(image_dtype: Any) -> np.dtype:
    return np.dtype(getattr(image_dtype, "as_numpy_dtype", image_dtype))


def _read_live_image_for_dataset(
    image_file: dict[str, Any],
    *,
    image_layer: dict[str, Any],
    ssl_verify: bool | str,
    image_dtype: Any,
    projection_crs: pyproj.CRS,
    projection_switch_axes: bool,
    projection_wms_srs: str,
) -> tuple[Path, dict[str, Any]]:
    image_path = Path(image_file["path"])

    try:
        image_info = load_image(
            bbox=image_file["bbox"],
            size=image_file["size"],
            image_layer=image_layer,
            ssl_verify=ssl_verify,
            projection_crs=projection_crs,
            projection_switch_axes=projection_switch_axes,
            projection_wms_srs=projection_wms_srs,
        )
    except Exception as ex:
        return image_path, {
            "image_data": None,
            "read_error": str(ex).encode("utf-8", errors="replace"),
            "read_error_traceback": traceback.format_exc().encode(
                "utf-8", errors="replace"
            ),
        }

    return image_path, {
        "image_data": np.asarray(
            image_info["image_data"], dtype=_numpy_dtype(image_dtype)
        ),
        "image_crs": image_info["image_crs"],
        "image_transform": image_info["image_transform"],
        "read_error": b"",
        "read_error_traceback": b"",
    }


def _read_image_for_dataset(
    image_path_value: Path,
    *,
    projection_if_missing: str | rio_crs.CRS | None,
    preprocess_input: Callable | None,
    image_dtype: Any,
) -> tuple[Path, dict[str, Any]]:
    image_path = Path(image_path_value)

    try:
        image_info = read_image(
            image_path=image_path,
            projection_if_missing=projection_if_missing,
            preprocess_input=preprocess_input,
        )
    except Exception as ex:
        return image_path, {
            "image_data": None,
            "read_error": str(ex).encode("utf-8", errors="replace"),
            "read_error_traceback": traceback.format_exc().encode(
                "utf-8", errors="replace"
            ),
        }

    return image_path, {
        "image_data": np.asarray(
            image_info["image_data"], dtype=_numpy_dtype(image_dtype)
        ),
        "image_crs": image_info["image_crs"],
        "image_transform": image_info["image_transform"],
        "read_error": b"",
        "read_error_traceback": b"",
    }

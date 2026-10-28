from contextlib import AbstractContextManager, nullcontext

import numpy as np
import pytest

from orthoseg.lib import predicter


class _CompletedFuture:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class _ImmediateExecutor:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def submit(self, function, *args, **kwargs):
        return _CompletedFuture(function(*args, **kwargs))


class _NoopProgressLogger:
    def __init__(self, **_kwargs):
        pass

    def update(self, **_kwargs):
        pass


def _identity_model():
    inputs = predicter.keras.layers.Input(shape=(2, 2, 1))
    outputs = predicter.keras.layers.Activation("linear")(inputs)
    return predicter.keras.models.Model(inputs=inputs, outputs=outputs)


def _patch_prediction_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(predicter.futures, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(predicter, "ProgressLogger", _NoopProgressLogger)
    monkeypatch.setattr(
        predicter._processing_util, "setprocessnice", lambda _priority: None
    )
    monkeypatch.setattr(
        predicter.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(tmp_path / "prediction-tmp"),
    )


def _successful_read_result(image_path, value):
    return image_path, {
        "image_data": predicter.tf.fill((2, 2, 1), float(value)),
        "image_crs": "EPSG:31370",
        "image_transform": "transform",
        "read_error": b"",
        "read_error_traceback": b"",
    }


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


@pytest.mark.parametrize(
    ("provide_directory", "provide_image_layer", "message"),
    [
        (False, False, "input_image_dir or image_layer should be provided"),
        (True, True, "cannot be provided together"),
    ],
)
def test_predict_layer_requires_exactly_one_input_source(
    tmp_path, provide_directory, provide_image_layer, message
):
    with pytest.raises(ValueError, match=message):
        predicter._predict_layer(
            model=None,  # type: ignore[arg-type]
            preprocess_input=None,
            input_image_dir=tmp_path / "input" if provide_directory else None,
            image_layer={"projection": "EPSG:31370"} if provide_image_layer else None,
            output_image_dir=tmp_path / "output",
            output_vector_path=None,
            classes=[],
            image_files=[],
        )


def test_predict_layer_uses_file_dataset_and_processes_final_partial_batch(
    monkeypatch, tmp_path
):
    _patch_prediction_runtime(monkeypatch, tmp_path)
    input_image_dir = tmp_path / "input"
    output_image_dir = tmp_path / "output"
    output_image_dir.mkdir()
    done_log_path = output_image_dir / "images_done.txt"
    done_log_path.write_text("already-done.tif\n")

    already_done_path = input_image_dir / "already-done.tif"
    first_image_path = input_image_dir / "nested" / "first.tif"
    second_image_path = input_image_dir / "nested" / "second.tif"
    image_files = [
        {"path": path}
        for path in (
            already_done_path,
            first_image_path,
            second_image_path,
        )
    ]
    read_results = [
        _successful_read_result(first_image_path, 1),
        _successful_read_result(second_image_path, 2),
    ]
    read_dataset = object()
    dataset_arguments = {}
    postprocess_arguments = []

    def create_image_dataset(**kwargs):
        dataset_arguments.update(kwargs)
        return read_dataset

    def iter_read_results(dataset, **_kwargs):
        assert dataset is read_dataset
        return iter(read_results)

    def postprocess_prediction(**kwargs):
        postprocess_arguments.append(kwargs)

    monkeypatch.setattr(
        predicter._predicter_dataset, "create_image_read_dataset", create_image_dataset
    )
    monkeypatch.setattr(
        predicter._predicter_dataset, "iter_image_read_results", iter_read_results
    )
    monkeypatch.setattr(
        predicter.postp, "postprocess_prediction_to_file", postprocess_prediction
    )

    predicter._predict_layer(
        model=_identity_model(),
        preprocess_input=None,
        input_image_dir=input_image_dir,
        image_layer=None,
        output_image_dir=output_image_dir,
        output_vector_path=None,
        classes=["background", "road"],
        image_files=image_files,
        batch_size=3,
        projection_if_missing="EPSG:31370",
        nb_parallel_read=-1,
        nb_parallel_postprocess=1,
    )

    assert dataset_arguments["image_paths"] == [first_image_path, second_image_path]
    assert dataset_arguments["projection_if_missing"] == (
        predicter.rio_crs.CRS.from_user_input("EPSG:31370")
    )
    assert dataset_arguments["read_batch_size"] == 3
    assert done_log_path.read_text().splitlines() == [
        "already-done.tif",
        "first.tif",
        "second.tif",
    ]
    assert len(postprocess_arguments) == 2
    for call, image_path, value in zip(
        postprocess_arguments,
        (first_image_path, second_image_path),
        (1, 2),
        strict=True,
    ):
        assert call["input_image_filepath"] == image_path
        assert call["output_image_dir"] == output_image_dir / "nested"
        np.testing.assert_array_equal(call["image_pred_arr"], np.full((2, 2, 1), value))


def test_predict_layer_uses_live_dataset_and_force_reprocesses_in_evaluate_mode(
    monkeypatch, tmp_path
):
    _patch_prediction_runtime(monkeypatch, tmp_path)
    output_image_dir = tmp_path / "output"
    evaluate_image_dir = tmp_path / "output_eval"
    evaluate_image_dir.mkdir()
    done_log_path = evaluate_image_dir / "images_done.txt"
    done_log_path.write_text("tile.tif\n")
    image_path = tmp_path / "tile.tif"
    image_layer = {"name": "live-source", "projection": "EPSG:31370"}
    image_files = [
        {
            "path": image_path,
            "bbox": (1.0, 2.0, 3.0, 4.0),
            "size": (2, 2),
        }
    ]
    read_dataset = object()
    dataset_arguments = {}
    postprocess_arguments = []

    def create_live_image_dataset(**kwargs):
        dataset_arguments.update(kwargs)
        return read_dataset

    def iter_read_results(dataset, **_kwargs):
        assert dataset is read_dataset
        return iter([_successful_read_result(image_path, 3)])

    def postprocess_prediction(**kwargs):
        postprocess_arguments.append(kwargs)

    monkeypatch.setattr(
        predicter._predicter_dataset,
        "create_live_image_read_dataset",
        create_live_image_dataset,
    )
    monkeypatch.setattr(
        predicter._predicter_dataset, "iter_image_read_results", iter_read_results
    )
    monkeypatch.setattr(
        predicter.postp,
        "postprocess_prediction_to_file",
        postprocess_prediction,
    )

    predicter._predict_layer(
        model=_identity_model(),
        preprocess_input=None,
        input_image_dir=None,
        image_layer=image_layer,
        output_image_dir=output_image_dir,
        output_vector_path=None,
        classes=["background", "road"],
        image_files=image_files,
        batch_size=1,
        evaluate_mode=True,
        nb_parallel_read=2,
        nb_parallel_postprocess=1,
        ssl_verify=False,
        force=True,
    )

    assert dataset_arguments["image_files"] == image_files
    assert dataset_arguments["image_layer"] is image_layer
    assert dataset_arguments["read_batch_size"] == 1
    assert dataset_arguments["ssl_verify"] is False
    assert done_log_path.read_text().splitlines() == ["tile.tif", "tile.tif"]
    assert len(postprocess_arguments) == 1
    assert postprocess_arguments[0]["evaluate_mode"] is True
    assert postprocess_arguments[0]["input_image_filepath"] == image_path
    assert postprocess_arguments[0]["output_image_dir"] == evaluate_image_dir
    np.testing.assert_array_equal(
        postprocess_arguments[0]["image_pred_arr"], np.full((2, 2, 1), 3)
    )


def test_predict_layer_logs_read_errors_and_stops_at_error_limit(monkeypatch, tmp_path):
    _patch_prediction_runtime(monkeypatch, tmp_path)
    input_image_dir = tmp_path / "input"
    output_image_dir = tmp_path / "output"
    image_path = input_image_dir / "broken.tif"
    read_dataset = object()

    monkeypatch.setattr(
        predicter._predicter_dataset,
        "create_image_read_dataset",
        lambda **_kwargs: read_dataset,
    )
    monkeypatch.setattr(
        predicter._predicter_dataset,
        "iter_image_read_results",
        lambda _dataset, **_kwargs: iter(
            [
                (
                    image_path,
                    {
                        "read_error": b"corrupt image",
                        "read_error_traceback": b"reader traceback",
                    },
                )
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="corrupt image"):
        predicter._predict_layer(
            model=_identity_model(),
            preprocess_input=None,
            input_image_dir=input_image_dir,
            image_layer=None,
            output_image_dir=output_image_dir,
            output_vector_path=None,
            classes=["background", "road"],
            image_files=[{"path": image_path}],
            batch_size=2,
            nb_parallel_read=1,
            nb_parallel_postprocess=1,
            max_prediction_errors=1,
        )

    error_log = (output_image_dir / "images_error.csv").read_text()
    assert "filename,error,traceback" in error_log
    assert "broken.tif" in error_log
    assert "corrupt image" in error_log
    assert "reader traceback" in error_log


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


@pytest.mark.parametrize("return_tensor", [False, True])
def test_predict_tensor_batch_converts_non_tensorflow_backend_input(
    monkeypatch, return_tensor
):
    inputs = predicter.keras.layers.Input(shape=(2, 2, 1))
    outputs = predicter.keras.layers.Activation("linear")(inputs)
    model = predicter.keras.models.Model(inputs=inputs, outputs=outputs)

    def predict_on_batch(image_batch):
        assert isinstance(image_batch, np.ndarray)
        predictions = image_batch + 1
        if return_tensor:
            return predicter.tf.convert_to_tensor(predictions)
        return predictions

    monkeypatch.setattr(predicter.keras.backend, "backend", lambda: "jax")
    monkeypatch.setattr(model, "predict_on_batch", predict_on_batch)

    image_batch = predicter.tf.ones((2, 2, 2, 1), dtype=predicter.tf.float32)
    predictions = predicter._predict_tensor_batch(model, image_batch)

    assert isinstance(predictions, np.ndarray)
    np.testing.assert_array_equal(predictions, np.full((2, 2, 2, 1), 2.0))


def test_predict_tensor_batch_reraises_oom_for_single_image(monkeypatch):
    inputs = predicter.keras.layers.Input(shape=(2, 2, 1))
    outputs = predicter.keras.layers.Activation("linear")(inputs)
    model = predicter.keras.models.Model(inputs=inputs, outputs=outputs)

    def raise_oom(_image_batch):
        raise predicter.tf.errors.ResourceExhaustedError(
            None, None, "single-image batch is too large"
        )

    monkeypatch.setattr(model, "predict_on_batch", raise_oom)

    with pytest.raises(predicter.tf.errors.ResourceExhaustedError):
        predicter._predict_tensor_batch(
            model,
            predicter.tf.ones((1, 2, 2, 1), dtype=predicter.tf.float32),
        )


def test_predict_image_batch_with_no_images_returns_empty_list():
    inputs = predicter.keras.layers.Input(shape=(2, 2, 1))
    outputs = predicter.keras.layers.Activation("linear")(inputs)
    model = predicter.keras.models.Model(inputs=inputs, outputs=outputs)

    assert predicter._predict_image_batch(model, []) == []

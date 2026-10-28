from contextlib import AbstractContextManager, nullcontext

import numpy as np
import pytest

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

    def raise_oom(image_batch):
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

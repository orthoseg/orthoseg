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

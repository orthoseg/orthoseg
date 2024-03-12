from contextlib import nullcontext
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

    if exp_error:
        handler = pytest.raises(ValueError)
    else:
        handler = nullcontext()

    with handler:
        predicter.predict_dir(
            model=None,  # type: ignore  # noqa: PGH003
            input_image_dir=input_image_dir,
            output_image_dir=output_image_dir,
            output_vector_path=None,
            classes=[],
            no_images_ok=no_images_ok,
        )

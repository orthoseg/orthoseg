"""Tests for module predict."""

import importlib
import os
import re
import shutil
from contextlib import nullcontext
from pathlib import Path

import geofileops as gfo
import geopandas as gpd
import pytest
from PIL import Image

from orthoseg import predict
from orthoseg.helpers import config_helper as conf
from orthoseg.predict import _predict_args
from tests import test_helper
from tests.test_helper import SportsFields

predict_module = importlib.import_module("orthoseg.predict")


def _write_cache_jpg(cache_dir: Path, pixel_x_size: float, pixel_y_size: float):
    image_dir = cache_dir / "first"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "tile.jpg"
    Image.new("RGB", (2, 2)).save(image_path)
    image_path.with_suffix(".jgw").write_text(
        f"{pixel_x_size}\n0\n0\n{-pixel_y_size}\n0\n0\n"
    )


def test_validate_input_image_dir(tmp_path, caplog):
    configured_dir = tmp_path / "512x512_0pxOverlap"
    fallback_dir = tmp_path / "1024x1024_0pxOverlap"
    fallback_dir.mkdir()
    _write_cache_jpg(fallback_dir, pixel_x_size=0.25, pixel_y_size=0.25)
    (tmp_path / "not_a_cache_dir").mkdir()

    result = predict_module._validate_input_image_dir(
        configured_dir,
        image_pixel_x_size=0.25,
        image_pixel_y_size=0.25,
        use_cache="yes",
    )

    assert result == fallback_dir
    assert "Using compatible cache directory" in caplog.text


def test_validate_input_image_dir_multiple_candidates(tmp_path, caplog):
    configured_dir = tmp_path / "512x512_0pxOverlap"
    candidate_dirs = [
        tmp_path / "1024x1024_0pxOverlap",
        tmp_path / "2048x2048_0pxOverlap",
    ]
    for candidate_dir in candidate_dirs:
        candidate_dir.mkdir()

    candidate_order = [path for path in tmp_path.iterdir() if path in candidate_dirs]
    _write_cache_jpg(candidate_order[0], pixel_x_size=0.5, pixel_y_size=0.25)
    _write_cache_jpg(candidate_order[1], pixel_x_size=0.25, pixel_y_size=0.25)

    result = predict_module._validate_input_image_dir(
        configured_dir,
        image_pixel_x_size=0.25,
        image_pixel_y_size=0.25,
        use_cache="yes",
    )

    assert result == candidate_order[1]
    assert f"Using compatible cache directory: {candidate_order[1]}" in caplog.text


@pytest.mark.parametrize("create_parent", [True, False])
@pytest.mark.parametrize(
    "use_cache, should_raise",
    [("yes", True), ("ifavailable", False), ("no", False)],
)
def test_validate_input_image_dir_without_candidates(
    tmp_path, create_parent, use_cache, should_raise
):
    configured_dir = tmp_path / "cache" / "512x512_0pxOverlap"
    if create_parent:
        configured_dir.parent.mkdir()

    if should_raise:
        with pytest.raises(ValueError, match="Input image directory does not exist"):
            predict_module._validate_input_image_dir(
                configured_dir,
                image_pixel_x_size=0.25,
                image_pixel_y_size=0.25,
                use_cache=use_cache,
            )
    else:
        assert (
            predict_module._validate_input_image_dir(
                configured_dir,
                image_pixel_x_size=0.25,
                image_pixel_y_size=0.25,
                use_cache=use_cache,
            )
            == configured_dir
        )


@pytest.mark.parametrize("use_cache", ["no", "ifavailable"])
def test_validate_input_image_dir_skips_incompatible_cache(tmp_path, caplog, use_cache):
    configured_dir = tmp_path / "512x512_0pxOverlap"
    incompatible_dir = tmp_path / "1024x1024_0pxOverlap"
    incompatible_dir.mkdir()
    _write_cache_jpg(incompatible_dir, pixel_x_size=0.5, pixel_y_size=0.5)

    assert (
        predict_module._validate_input_image_dir(
            configured_dir,
            image_pixel_x_size=0.25,
            image_pixel_y_size=0.25,
            use_cache=use_cache,
        )
        == configured_dir
    )
    if use_cache == "ifavailable":
        assert "fetching images directly" in caplog.text


@pytest.mark.parametrize(
    "cached_pixel_x_size, cached_pixel_y_size, is_compatible",
    [
        (0.25, 0.5, True),
        (0.5, 0.5, False),
        (0.25, 0.75, False),
    ],
)
def test_validate_input_image_dir_checks_pixel_sizes(
    tmp_path, cached_pixel_x_size, cached_pixel_y_size, is_compatible
):
    input_image_dir = tmp_path / "512x512_0pxOverlap"
    fallback_dir = tmp_path / "1024x1024_0pxOverlap"
    fallback_dir.mkdir()
    _write_cache_jpg(
        fallback_dir,
        pixel_x_size=cached_pixel_x_size,
        pixel_y_size=cached_pixel_y_size,
    )

    if is_compatible:
        assert (
            predict_module._validate_input_image_dir(
                input_image_dir,
                image_pixel_x_size=0.25,
                image_pixel_y_size=0.5,
                use_cache="yes",
            )
            == fallback_dir
        )
    else:
        with pytest.raises(ValueError, match="No compatible cache directory"):
            predict_module._validate_input_image_dir(
                input_image_dir,
                image_pixel_x_size=0.25,
                image_pixel_y_size=0.5,
                use_cache="yes",
            )


@pytest.mark.parametrize(
    "args",
    [
        (["--config", "test.ini", "predict.image_layer=abc", "--no-postprocess"]),
        (["--config", "test.ini", "predict.image_layer=abc", "predict.batch_size=1"]),
    ],
)
def test_predict_args(args):
    valid_args = _predict_args(args=args)

    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None
    assert valid_args.no_postprocess is not None

    if "--no-postprocess" in args:
        assert valid_args.no_postprocess
        nb_overrules = len(args) - 3
    else:
        assert not valid_args.no_postprocess
        nb_overrules = len(args) - 2

    assert len(valid_args.config_overrules) == nb_overrules


@pytest.mark.parametrize("config_path, exp_error", [(Path("INVALID"), True)])
def test_predict_invalid_config(config_path, exp_error):
    if exp_error:
        handler = pytest.raises(ValueError)
    else:
        handler = nullcontext()
    with handler:
        predict(config_path=config_path)


def test_predict_error_handling():
    """Force an error so the general error handler in predict is tested."""
    with pytest.raises(
        RuntimeError,
        match=re.escape("ERROR in predict for sportsfields.ini on UNEXISTING"),
    ):
        predict(
            config_path=SportsFields.config_path,
            config_overrules=["predict.image_layer=UNEXISTING"],
        )


@pytest.mark.parametrize(
    "use_cache, skip_images", [(True, False), (True, True), (False, False)]
)
def test_predict_use_cache_skip(tmp_path, use_cache, skip_images):
    if not use_cache and os.name == "nt":
        pytest.skip("Test to predict without cache crashes on windows")

    if skip_images and not use_cache:
        raise ValueError("skip_images=True is only possible in test if use_cache=True")

    # Init
    testprojects_dir = tmp_path / "sample_projects"
    # Use sportsfields sample project
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)
    project_dir = testprojects_dir / SportsFields.subject

    config_path = project_dir / SportsFields.config_path.name
    conf.read_orthoseg_config(config_path=config_path)
    image_cache_dir = conf.dirs.getpath("predict_image_input_dir")

    # Remove the output vector dir to force a new prediction
    result_vector_dir = conf.dirs.getpath("output_vector_dir")
    shutil.rmtree(result_vector_dir, ignore_errors=True)

    # If use cache or skip_images is True, keep the cache. Cache is always needed when
    # skip_images is True to be able to determine which images to skip.
    if use_cache:
        assert image_cache_dir.exists()
    else:
        shutil.rmtree(image_cache_dir)

    # With skip_images, create a done file in the image prediction output dir
    # to skip all images but the last one. This will reduce the number of features in
    # the output.
    if skip_images:
        predict_image_output_basedir = Path(
            f"{conf.dirs['predict_image_output_basedir']}_sportsfields_01"
        )
        predict_image_output_basedir.mkdir(parents=True, exist_ok=True)
        images = [image_path.name for image_path in image_cache_dir.rglob("*.jpg")]
        images_to_skip = sorted(images)[:-1]
        done_path = predict_image_output_basedir / "images_done.txt"
        with done_path.open("w") as f:
            for image_name in images_to_skip:
                f.write(f"{image_name}\n")

        # If no cache should be used, remove the cache again
        if not use_cache:
            if image_cache_dir.exists():
                shutil.rmtree(image_cache_dir)

    # Run predict
    predict(config_path=config_path)

    # Check output results
    result_vector_dir = conf.dirs.getpath("output_vector_dir")
    result_vector_path = (
        result_vector_dir / "sportsfields_01_BEFL-2025-sportsfields.gpkg"
    )

    # The area of the output should be within a 10% margin of the expected area.
    assert result_vector_path.exists()
    result_gdf = gpd.read_file(result_vector_path)
    assert result_gdf is not None
    assert result_gdf.crs.to_epsg() == 31370
    exp_area = 33177 if not skip_images else 5723
    assert exp_area * 0.9 < sum(result_gdf.geometry.area) < exp_area * 1.1


@pytest.mark.parametrize("postprocess", [False, True])
def test_predict_postprocess(tmp_path, postprocess):
    # Init test project and keep the existing prediction output file in place so
    # predict can reuse it and skip actual inference.
    testprojects_dir = tmp_path / "sample_projects"
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)
    project_dir = testprojects_dir / SportsFields.subject
    config_path = project_dir / SportsFields.config_path.name

    result_path = predict(config_path=config_path, postprocess=postprocess)

    # Check result
    result_count = gfo.get_layerinfo(result_path).featurecount
    if postprocess:
        assert result_count == SportsFields.expected_postprocess_count
        assert not result_path.stem.endswith("_predict")
    else:
        assert result_count == SportsFields.expected_output_count
        assert result_path.stem.endswith("_predict")


@pytest.mark.parametrize(
    "dtype_policy_raw, nb_gpu, compute_capability, expected_calls",
    [
        ("float32", 0, None, ["float32"]),
        ("", 0, None, []),
        ("", 2, (7, 0), ["mixed_float16"]),
        ("", 2, (6, 1), []),
        ("", 1, None, []),
        (None, 0, None, []),
        (None, 1, (7, 5), ["mixed_float16"]),
    ],
)
def test_set_dtype_policy_for_predict(
    monkeypatch, dtype_policy_raw, nb_gpu, compute_capability, expected_calls
):
    calls = []

    monkeypatch.setattr(
        predict_module.mh,
        "get_number_gpus",
        lambda: nb_gpu,
    )
    monkeypatch.setattr(
        predict_module.mh,
        "get_min_gpu_compute_capability",
        lambda: compute_capability,
    )
    monkeypatch.setattr(
        predict_module.mf,
        "set_dtype_policy",
        lambda policy: calls.append(policy),
    )

    predict_module._set_dtype_policy_for_predict(dtype_policy_raw)

    assert calls == expected_calls

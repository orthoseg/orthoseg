"""Tests for module predict."""

import os
import re
import shutil
from contextlib import nullcontext
from pathlib import Path

import geopandas as gpd
import pytest

from orthoseg import predict
from orthoseg.helpers import config_helper as conf
from orthoseg.predict import _predict_args
from tests import test_helper
from tests.test_helper import SportsFields


@pytest.mark.parametrize(
    "args",
    [
        (
            [
                "--config",
                "X:/Monitoring/OrthoSeg/test/test.ini",
                "predict.image_layer=LT-2023",
            ]
        )
    ],
)
def test_predict_args(args):
    valid_args = _predict_args(args=args)
    assert valid_args is not None
    assert valid_args.config is not None
    assert valid_args.config_overrules is not None


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
            f"{conf.dirs['predict_image_output_basedir']}_sportsfields_01_276"
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
        result_vector_dir / "sportsfields_01_276_BEFL-2025-sportsfields.gpkg"
    )

    # The area of the output should be within a 10% margin of the expected area.
    assert result_vector_path.exists()
    result_gdf = gpd.read_file(result_vector_path)
    assert result_gdf is not None
    assert result_gdf.crs.to_epsg() == 31370
    exp_area = 33177 if not skip_images else 5723
    assert exp_area * 0.9 < sum(result_gdf.geometry.area) < exp_area * 1.1

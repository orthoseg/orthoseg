"""Tests for module predict."""

import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import geopandas as gpd
import pytest

import orthoseg
from orthoseg import predict
from orthoseg.helpers import config_helper as conf
from orthoseg.predict import _predict_args
from tests import test_helper


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


@pytest.mark.parametrize("config_path, exp_error", [("INVALID", True)])
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
        match="ERROR in predict for footballfields_BEFL-2019_test on UNEXISTING",
    ):
        predict(
            config_path=test_helper.SampleProjectFootball.predict_config_path,
            config_overrules=["predict.image_layer=UNEXISTING"],
        )


@pytest.mark.skipif(os.name == "nt", reason="crashes on windows")
@pytest.mark.parametrize(
    "use_cache, skip_images, exp_area",
    [
        (True, False, 30000),
        (True, True, 12585),
        (False, False, 30000),
        (False, True, 12585),
    ],
)
def test_predict_use_cache_skip(tmp_path, use_cache, skip_images, exp_area):
    # Init
    testprojects_dir = tmp_path / "sample_projects"
    # Use footballfields sample project
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)

    footballfields_dir = testprojects_dir / "footballfields"

    config_path = footballfields_dir / "footballfields_BEFL-2019_test.ini"
    conf.read_orthoseg_config(config_path=config_path)
    image_cache_dir = conf.dirs.getpath("predict_image_input_dir")

    # Load images if use cache or skip_images is True. Cache is always needed when
    # skip_images is True to be able to determine which images to skip.
    assert not image_cache_dir.exists()
    if use_cache or skip_images:
        orthoseg.load_images(config_path=config_path)
        assert image_cache_dir.exists()

    # With skip_images, create a done file in the image prediction output dir
    # to skip all images but the last one. This will reduce the number of features in
    # the output.
    if skip_images:
        predict_image_output_basedir = Path(
            f"{conf.dirs['predict_image_output_basedir']}_footballfields_01_201"
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

    # Download the model
    model_dir = conf.dirs.getpath("model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    test_helper.SampleProjectFootball.download_model(model_dir)

    # Run predict
    predict(config_path=config_path)

    # Check output results
    result_vector_dir = conf.dirs.getpath("output_vector_dir")
    result_vector_path = result_vector_dir / "footballfields_01_201_BEFL-2019.gpkg"

    # The area of the output should be within a 10% margin of the expected area.
    assert result_vector_path.exists()
    result_gdf = gpd.read_file(result_vector_path)
    assert exp_area * 0.9 < sum(result_gdf.geometry.area) < exp_area * 1.1

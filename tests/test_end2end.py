"""End to end tests for the entire orthoseg process."""

import shutil
import tempfile
from pathlib import Path

import geofileops as gfo
import geopandas as gpd
import pytest
import shapely

import orthoseg
import orthoseg.model.model_helper as mh
from orthoseg.helpers import config_helper as conf
from tests import test_helper
from tests.test_helper import ProjectTemplate, SportsFields

testprojects_dir = Path(tempfile.gettempdir()) / "orthoseg_test_end2end/sample_projects"
sportsfields_dir = testprojects_dir / SportsFields.subject
projecttemplate_dir = testprojects_dir / ProjectTemplate.project_dir.name


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def test_1_init_testproject():
    # Use sportsfields sample project for these end to end tests
    shutil.rmtree(testprojects_dir, ignore_errors=True)
    shutil.copytree(test_helper.sampleprojects_dir, testprojects_dir)


@pytest.mark.order(after="test_1_init_testproject")
def test_2_load_images():
    # Load project config to init some vars.
    config_path = sportsfields_dir / SportsFields.config_path.name
    conf.read_orthoseg_config(config_path)
    image_cache_dir = conf.dirs.getpath("predict_image_input_dir")

    # Clean result if it isn't empty yet.
    # Remark: by default, disable clean to improve test speed.
    clean_cache = False
    if clean_cache and image_cache_dir.exists():
        shutil.rmtree(image_cache_dir)
        assert not image_cache_dir.exists()

    # Run task to load images
    orthoseg.load_images(config_path)

    # Check if the right number of files was loaded
    assert image_cache_dir.exists()
    files = list(image_cache_dir.glob("**/*.jpg"))
    assert len(files) == 4


@pytest.mark.order(after="test_1_init_testproject")
def test_3_train():
    # Load project config to init some vars.
    config_path = sportsfields_dir / SportsFields.config_path.name

    # Overrule config so small images are used, only one epoch is ran,... to speed up
    # the test.
    overrules = [
        "train.image_pixel_width=32",
        "train.image_pixel_height=32",
        "train.preload_with_previous_traindata=False",
        "train.force_model_traindata_id=-1",
        "train.resume_train=False",
        "train.force_train=False",
        "train.batch_size_fit=1",
        "train.batch_size_predict=1",
        "train.save_best_only=False",
        "train.save_min_accuracy=0",
        "train.nb_epoch_with_freeze=0",
        "train.max_epoch=1",
    ]
    conf.read_orthoseg_config(config_path, overrules=overrules)

    # Replace label files with minimal test data to speed up the tests.
    labels_dir = conf.dirs.getpath("labels_dir")
    shutil.rmtree(labels_dir, ignore_errors=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_locations_df = gpd.GeoDataFrame(
        data={"traindata_type": ["train", "validation"]},
        geometry=[
            shapely.box(100_000, 100_000, 100_128, 100_128),
            shapely.box(101_000, 100_000, 101_128, 100_128),
        ],
        crs="EPSG:31370",
    )
    label_locations_df.to_file(labels_dir / "sportsfields_BEFL-2025_locations.gpkg")
    label_polygons_df = gpd.GeoDataFrame(
        data={"classname": ["football_field", "tennis_court"]},
        geometry=[
            shapely.box(100_010, 100_010, 100_020, 100_020),
            shapely.box(100_030, 100_010, 100_040, 100_020),
        ],
        crs="EPSG:31370",
    )
    label_polygons_df.to_file(labels_dir / "sportsfields_BEFL-2025_polygons.gpkg")

    # Init + cleanup result dirs
    traindata_id_result = 2
    training_dir = conf.dirs.getpath("training_dir")
    training_id_dir = training_dir / f"{traindata_id_result:02d}"
    if training_id_dir.exists():
        shutil.rmtree(training_id_dir)
    model_dir = conf.dirs.getpath("model_dir")
    if model_dir.exists():
        modelfile_paths = model_dir.glob(
            f"{SportsFields.subject}_{traindata_id_result:02d}_*"
        )
        for modelfile_path in modelfile_paths:
            modelfile_path.unlink()

    # Run train session
    orthoseg.train(config_path, config_overrules=overrules)

    # Check if the training (image) data was created
    assert training_id_dir.exists()

    # Check if the new model was created
    best_model = mh.get_best_model(
        model_dir=conf.dirs.getpath("model_dir"),
        segment_subject=conf.general["segment_subject"],
        traindata_id=traindata_id_result,
    )

    assert best_model is not None
    assert best_model["traindata_id"] == traindata_id_result
    assert best_model["epoch"] == 0


@pytest.mark.order(after="test_2_load_images")
def test_4_predict():
    # Load project config to init some vars.
    config_path = sportsfields_dir / SportsFields.config_path.name
    overrules = ["train.force_model_traindata_id=1"]

    conf.read_orthoseg_config(config_path, overrules=overrules)

    # Cleanup result if it isn't empty yet
    predict_image_output_dir = Path(
        f"{conf.dirs['predict_image_output_basedir']}_{SportsFields.subject}_02_0"
    )
    if predict_image_output_dir.exists():
        shutil.rmtree(predict_image_output_dir)
        # Make sure it is deleted now!
        assert not predict_image_output_dir.exists()
    result_vector_dir = conf.dirs.getpath("output_vector_dir")
    if result_vector_dir.exists():
        shutil.rmtree(result_vector_dir)
        # Make sure it is deleted now!
        assert not result_vector_dir.exists()

    # Run task to predict
    orthoseg.predict(config_path, config_overrules=overrules)

    # Check results
    result_vector_path = (
        result_vector_dir / f"{SportsFields.subject}_01_276_BEFL-2025-sportsfields.gpkg"
    )
    assert result_vector_path.exists()
    result_gdf = gfo.read_file(result_vector_path)
    expected_count = 27
    assert len(result_gdf) == expected_count


@pytest.mark.order(after="test_4_predict")
def test_5_postprocess():
    # Load project config to init some vars.
    config_path = sportsfields_dir / SportsFields.config_path.name
    overrules = ["train.force_model_traindata_id=1"]

    conf.read_orthoseg_config(config_path, overrules=overrules)

    # Run task to postprocess
    output_vector_path = orthoseg.postprocess(config_path, config_overrules=overrules)

    # Check results
    assert output_vector_path.exists()
    result_gdf = gfo.read_file(output_vector_path)
    expected_count = 22
    assert len(result_gdf) == expected_count

    # The output file should contain the style from the project dir
    styles = gfo.get_layerstyles(output_vector_path)
    assert len(styles) == 1
    layer_name = gfo.get_only_layer(output_vector_path)
    assert styles.iloc[0]["f_table_name"] == layer_name
    assert styles.iloc[0]["styleName"] == SportsFields.subject

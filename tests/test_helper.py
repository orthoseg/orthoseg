"""
Helper functions for all tests.
"""

import logging
from pathlib import Path
import tempfile
from typing import Optional

import geopandas as gpd
from shapely import geometry as sh_geom


class TestData:
    testdata_dir = Path(__file__).resolve().parent / "data"
    sampleprojects_dir = Path(__file__).resolve().parent.parent / "sample_projects"

    classes = {
        "background": {
            "labelnames": ["ignore_for_train", "background"],
            "weight": 1,
            "burn_value": 0,
        },
        "test_classname1": {"labelnames": ["testlabel1"], "weight": 1, "burn_value": 1},
        "test_classname2": {"labelnames": ["testlabel2"], "weight": 1, "burn_value": 1},
    }
    image_pixel_x_size = 0.25
    image_pixel_y_size = 0.25
    image_pixel_width = 512
    image_pixel_height = 512
    image_crs_width = image_pixel_width * image_pixel_x_size
    image_crs_height = image_pixel_height * image_pixel_y_size
    crs_xmin = 150000
    crs_ymin = 150000
    crs = "EPSG:31370"
    location = sh_geom.box(
        crs_xmin,
        crs_ymin,
        crs_xmin + (image_pixel_width * image_pixel_x_size),
        crs_ymin + (image_pixel_height * image_pixel_y_size),
    )
    location_invalid = sh_geom.Polygon(
        [
            (crs_xmin, crs_ymin),
            (crs_xmin + image_crs_width, crs_ymin),
            (crs_xmin + image_crs_width, crs_ymin + image_crs_height),
            (crs_xmin, crs_ymin + image_crs_height),
            (
                crs_xmin + image_pixel_x_size,
                crs_ymin + image_crs_height + image_pixel_y_size,
            ),
            (crs_xmin, crs_ymin),
        ]
    )
    polygon = location
    polygon_invalid = location_invalid
    locations_gdf = gpd.GeoDataFrame(
        {
            "geometry": [location, location, location, location],
            "traindata_type": ["train", "validation", "test", "todo"],
            "path": "/tmp/locations.gdf",
        },
        crs="epsg:31370",
    )  # type: ignore
    polygons_gdf = gpd.GeoDataFrame(
        {
            "geometry": [polygon, polygon],
            "classname": ["testlabel1", "testlabel2"],
            "path": "/tmp/polygons.gdf",
        },
        crs="epsg:31370",
    )  # type: ignore


def create_tempdir(base_dirname: str, parent_dir: Optional[Path] = None) -> Path:
    # Parent
    if parent_dir is None:
        parent_dir = Path(tempfile.gettempdir())

    for i in range(1, 999999):
        try:
            tempdir = parent_dir / f"{base_dirname}_{i:06d}"
            tempdir.mkdir(parents=True)
            return Path(tempdir)
        except FileExistsError:
            continue

    raise Exception(
        "Wasn't able to create a temporary dir with basedir: "
        f"{parent_dir / base_dirname}"
    )


def init_test_for_debug(test_module_name: str) -> Path:
    # Init logging
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    # Prepare tmpdir
    tmp_basedir = Path(tempfile.gettempdir()) / test_module_name
    tmpdir = create_tempdir(parent_dir=tmp_basedir, base_dirname="debugrun")

    """
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    """

    return tmpdir

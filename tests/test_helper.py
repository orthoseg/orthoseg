# -*- coding: utf-8 -*-
"""
Helper functions for all tests.
"""

import logging
from pathlib import Path
import sys
import tempfile
from typing import Optional

import geopandas as gpd
from shapely import geometry as sh_geom

# Add path so the local geofileops packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestData:
    testdata_dir = Path(__file__).resolve().parent / "data"
    testprojects_dir = Path(__file__).resolve().parent / "test_projects"

    classes = {
        "background": {
            "labelnames": ["ignore_for_train", "background"],
            "weight": 1,
            "burn_value": 0,
        },
        "test": {"labelnames": ["testlabel"], "weight": 1, "burn_value": 1},
    }
    locations_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                sh_geom.box(150000, 170000, 150128, 170128),
                sh_geom.box(150000, 180000, 150128, 180128),
                sh_geom.box(150000, 190000, 150128, 190128),
            ],
            "traindata_type": ["train", "train", "validation"],
        },
        crs="epsg:31370",
    )  # type: ignore
    polygons_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                sh_geom.box(150030, 170030, 150060, 170060),
                sh_geom.box(150030, 180030, 150060, 180060),
            ],
            "label_name": ["testlabel", "testlabel"],
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

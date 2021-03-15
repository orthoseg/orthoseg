# -*- coding: utf-8 -*-
"""
Helper functions for all tests.
"""

import logging
from pathlib import Path
import sys
import tempfile

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def get_testprojects_dir() -> Path:
    return Path(__file__).resolve().parent / 'test_projects'

def create_tempdir(
        base_dirname: str,
        parent_dir: Path = None) -> Path:
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

    raise Exception(f"Wasn't able to create a temporary dir with basedir: {parent_dir / base_dirname}") 

def init_test_for_debug(test_module_name: str) -> Path:
    # Init logging
    logging.basicConfig(
            format="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s", 
            datefmt="%H:%M:%S", level=logging.INFO)

    # Prepare tmpdir
    tmp_basedir = Path(tempfile.gettempdir()) / test_module_name
    tmpdir = create_tempdir(parent_dir=tmp_basedir, base_dirname='debugrun')
    
    """
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    """

    return tmpdir

# -*- coding: utf-8 -*-
"""
Test if backwards compatibility for old API still works.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import orthoseg


def test_version():
    assert "\n" not in orthoseg.__version__
    assert "." in orthoseg.__version__

"""
Test if backwards compatibility for old API still works.
"""

import orthoseg


def test_version():
    assert "\n" not in orthoseg.__version__
    assert "." in orthoseg.__version__

"""Shared pytest configuration."""

import os

if os.name == "nt" and "GITHUB_ACTIONS" in os.environ:
    # Load TensorFlow before test modules import native geospatial libraries.
    import tensorflow  # noqa: F401

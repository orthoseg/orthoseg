"""Shared pytest configuration."""

import gc
import os


def pytest_runtest_teardown(item, nextitem):  # noqa: ARG001
    """Run garbage collection after each test on Windows.

    This prevents C-extension segfaults caused by accumulated cyclic references.
    """
    if os.name == "nt":
        gc.collect()

"""Conftest.py pytest setup."""

import shutil

import pytest


def pytest_configure(config):
    import matplotlib as mpl

    mpl.use("Agg")


def pytest_runtest_setup(item):
    """Called for each test."""
    markers = [value.name for value in item.iter_markers()]

    # pytest.mark.requires_ert:
    if "requires_ert" in markers and not shutil.which("ert"):
        pytest.skip("Skip test if not ERT present (executable 'ert' is missing)")

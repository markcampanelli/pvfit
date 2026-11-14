"""
PVfit testing: Getting-started demo for single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import importlib
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Do not show figures.


def test_getting_started_matrix_script():
    """Test that getting started script runs without error."""

    runpy.run_path(
        importlib.resources.files("pvfit.demos.sdm")
        / Path(__file__).name.replace("_test.py", ".py")
    )

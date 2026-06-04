"""
PVfit testing: Getting-started demo for single-diode equation (SDE).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import importlib
from pathlib import Path
import runpy

import matplotlib

matplotlib.use("Agg")  # Do not show figures.


def test_getting_started_script():
    """Test that getting started script runs without error."""

    runpy.run_path(
        importlib.resources.files("pvfit.demos.sde")
        / Path(__file__).name.replace("_test.py", ".py")
    )

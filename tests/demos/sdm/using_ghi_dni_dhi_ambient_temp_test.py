"""
PVfit testing: Using meteorological (MET) station data in single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import importlib
from pathlib import Path
import runpy

import matplotlib

matplotlib.use("Agg")  # Do not show figures.


def test_using_ghi_dni_dhi_ambient_temp_script():
    """Test that F and T_degC from MET data script runs without error."""

    runpy.run_path(
        importlib.resources.files("pvfit.demos.sdm")
        / Path(__file__).name.replace("_test.py", ".py")
    )

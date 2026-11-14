"""
PVfit testing: Using meteorological (MET) station data in single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import importlib
import runpy
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # Do not show figures.


@pytest.mark.skipif(
    sys.version_info >= (3, 15), reason="pvlib requires Python less than 3.15"
)
def test_using_ghi_dni_dhi_ambient_temp_script():
    """Test that F and T_degC from MET data script runs without error."""

    runpy.run_path(
        importlib.resources.files("pvfit.demos.sdm")
        / Path(__file__).name.replace("_test.py", ".py")
    )

"""
PVfit testing: Using meteorological (MET) station data in single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import os
import runpy


def test_using_ghi_dni_dhi_ambient_temp_script():
    """Test that F and T_degC from MET data script runs without error."""
    runpy.run_path(os.path.abspath(__file__).replace("_test.py", ".py"))

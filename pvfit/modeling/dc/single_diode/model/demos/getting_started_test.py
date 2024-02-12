"""
PVfit testing: Getting-started demo for single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import os
import runpy


def test_getting_started_script():
    """Test that getting started script runs without error."""
    runpy.run_path(os.path.abspath(__file__).replace("_test.py", ".py"))

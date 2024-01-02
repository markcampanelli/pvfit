"""
PVfit testing: Getting-started demo for spectral corrections.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import importlib.resources


def test_getting_started():
    """Test that getting started script runs without error."""
    exec(
        importlib.resources.files("pvfit.measurement.spectral_correction.demos")
        .joinpath("getting_started.py")
        .read_text()
    )

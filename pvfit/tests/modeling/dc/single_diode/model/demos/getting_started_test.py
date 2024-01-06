"""
PVfit testing: Getting-started demo for single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import importlib.resources


def test_getting_started_model():
    """Test that getting started script runs without error."""
    exec(
        importlib.resources.files("pvfit.modeling.dc.single_diode.model.demos")
        .joinpath("getting_started.py")
        .read_text()
    )

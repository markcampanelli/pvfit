"""
PVfit testing: Getting-started demo for spectral corrections.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import importlib.resources
import os


def test_getting_started():
    """Test that getting started script runs without error."""
    exec(
        open(
            os.path.join(
                str(
                    importlib.resources.files(
                        "pvfit.measurement.spectral_correction.demos"
                    )
                ),
                "getting_started.py",
            )
        ).read()
    )

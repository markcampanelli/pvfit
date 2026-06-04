"""
PVfit testing: Getting-started demo for spectral corrections.

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
        importlib.resources.files("pvfit.demos.spectral_correction")
        / Path(__file__).name.replace("_test.py", ".py")
    )


# FIXME Assert result by refactoring this test.
# numpy.testing.assert_allclose(
#     computation.M(
#         S_TD_OC=data.S_TD_NIST,
#         E_TD_OC=data.E_sim_NIST,
#         S_TD_RC=data.S_TD_NIST,
#         E_TD_RC=data.E_G173_global_tilt,
#         S_RD_OC=data.S_RD_NIST,
#         E_RD_OC=data.E_sim_NIST,
#         S_RD_RC=data.S_RD_NIST,
#         E_RD_RC=data.E_G173_global_tilt,
#     ),
#     0.9982571553509618,
# )

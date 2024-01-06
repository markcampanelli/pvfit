"""
PVfit testing: Sample measurement data for example spectral corrections.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import pvfit.measurement.spectral_correction.computation as computation
import pvfit.measurement.spectral_correction.demos.data as data


def test_nist_M():
    """Test sample data provided by NIST."""
    # Scalar conversions to spectral response from responsivity cancel out in this
    # computation.
    assert (
        computation.M(
            S_TD_OC=data.S_TD_NIST,
            E_TD_OC=data.E_sim_NIST,
            S_TD_RC=data.S_TD_NIST,
            E_TD_RC=data.E_G173_global_tilt,
            S_RD_OC=data.S_RD_NIST,
            E_RD_OC=data.E_sim_NIST,
            S_RD_RC=data.S_RD_NIST,
            E_RD_RC=data.E_G173_global_tilt,
        )
        == 0.9982571553509605
    )

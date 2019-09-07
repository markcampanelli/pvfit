import pvfit.measurement.spectral_correction.api


def test_nist_M():
    """Test sample data provided by NIST."""
    # Scalar conversions to spectral response from responsivity cancel out in this computation.
    assert pvfit.measurement.spectral_correction.api.M(
        S_TD_OC=pvfit.measurement.spectral_correction.data.S_TD_NIST,
        E_TD_OC=pvfit.measurement.spectral_correction.data.E_sim_NIST,
        S_TD_RC=pvfit.measurement.spectral_correction.data.S_TD_NIST,
        E_TD_RC=pvfit.measurement.spectral_correction.data.E_G173_global_tilt,
        S_RD_OC=pvfit.measurement.spectral_correction.data.S_RD_NIST,
        E_RD_OC=pvfit.measurement.spectral_correction.data.E_sim_NIST,
        S_RD_RC=pvfit.measurement.spectral_correction.data.S_RD_NIST,
        E_RD_RC=pvfit.measurement.spectral_correction.data.E_G173_global_tilt) == 0.9982571553509605

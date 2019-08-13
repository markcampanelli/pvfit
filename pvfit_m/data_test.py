import pvfit_m


def test_nist_m():
    """Test sample data provided by NIST."""
    # Scalar conversions to spectral response from responsivity cancel out in this computation.
    assert pvfit_m.api.compute_m(sr_td=pvfit_m.data.sr_td_NIST, si_td=pvfit_m.data.si_sim_NIST,
                                 sr_rd=pvfit_m.data.sr_rd_NIST, si_rd=pvfit_m.data.si_sim_NIST,
                                 si_0=pvfit_m.data.si_G173_global_tilt) == 0.9982571553509605

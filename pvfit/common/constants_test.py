import numpy

import pvfit.common.constants as constants


# Test constants for value changes. Constants are python scalars.

def test_q_C():
    assert isinstance(constants.q_C, float)
    assert constants.q_C == 1.602176634e-19


def test_k_B_J_per_K():
    assert isinstance(constants.k_B_J_per_K, float)
    assert constants.k_B_J_per_K == 1.380649e-23


def test_k_B_eV_per_K():
    assert isinstance(constants.k_B_eV_per_K, float)
    assert constants.k_B_eV_per_K == 8.617333262e-05


def test_T_stc_degC():
    assert isinstance(constants.T_stc_degC, float)
    assert constants.T_stc_degC == 25.


def test_T_abs_zero_degC():
    assert isinstance(constants.T_abs_zero_degC, float)
    assert constants.T_abs_zero_degC == -273.15


def test_T_stc_K():
    assert isinstance(constants.T_stc_K, float)
    assert constants.T_stc_K == 298.15


def test_c_m_per_s():
    assert constants.c_m_per_s == 299792458.0


def test_h_J_s():
    assert constants.h_J_s == 6.62607015e-34


def test_G_hemi_stc_W_per_m2():
    assert isinstance(constants.G_hemi_stc_W_per_m2, float)
    numpy.testing.assert_array_equal(constants.G_hemi_stc_W_per_m2, 1000.)


def test_materials():
    assert isinstance(constants.materials['CIGS']['E_g_stc_eV'], float)
    numpy.testing.assert_array_equal(constants.materials['CIGS']['E_g_stc_eV'], 1.15)

    assert isinstance(constants.materials['CIS']['E_g_stc_eV'], float)
    numpy.testing.assert_array_equal(constants.materials['CIS']['E_g_stc_eV'], 1.010)

    assert isinstance(constants.materials['CdTe']['E_g_stc_eV'], float)
    numpy.testing.assert_array_equal(constants.materials['CdTe']['E_g_stc_eV'], 1.475)

    assert isinstance(constants.materials['GaAs']['E_g_stc_eV'], float)
    numpy.testing.assert_array_equal(constants.materials['GaAs']['E_g_stc_eV'], 1.43)

    assert isinstance(constants.materials['x-Si']['E_g_stc_eV'], float)
    numpy.testing.assert_array_equal(constants.materials['x-Si']['E_g_stc_eV'], 1.121)


def test_newton_options_default():
    numpy.testing.assert_array_equal(constants.newton_options_default['maxiter'], 50)
    numpy.testing.assert_array_equal(constants.newton_options_default['tol'], 1.48e-08)


def test_minimize_scalar_bounded_options_default():
    numpy.testing.assert_array_equal(constants.minimize_scalar_bounded_options_default['maxiter'], 500)
    numpy.testing.assert_array_equal(constants.minimize_scalar_bounded_options_default['xatol'], 1e-05)

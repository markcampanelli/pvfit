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


def test_T_degC_stc():
    assert isinstance(constants.T_degC_stc, float)
    assert constants.T_degC_stc == 25.0


def test_T_degC_abs_zero():
    assert isinstance(constants.T_degC_abs_zero, float)
    assert constants.T_degC_abs_zero == -273.15


def test_T_K_stc():
    assert isinstance(constants.T_K_stc, float)
    assert constants.T_K_stc == 298.15


def test_c_m_per_s():
    assert constants.c_m_per_s == 299792458.0


def test_h_J_s():
    assert constants.h_J_s == 6.62607015e-34


def test_G_hemi_W_per_m2_stc():
    assert isinstance(constants.G_hemi_W_per_m2_stc, float)
    numpy.testing.assert_array_equal(constants.G_hemi_W_per_m2_stc, 1000.0)


def test_materials():
    assert isinstance(constants.materials["CIGS"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(constants.materials["CIGS"]["E_g_eV_stc"], 1.15)

    assert isinstance(constants.materials["CIS"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(constants.materials["CIS"]["E_g_eV_stc"], 1.010)

    assert isinstance(constants.materials["CdTe"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(constants.materials["CdTe"]["E_g_eV_stc"], 1.475)

    assert isinstance(constants.materials["GaAs"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(constants.materials["GaAs"]["E_g_eV_stc"], 1.43)

    assert isinstance(constants.materials["x-Si"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(constants.materials["x-Si"]["E_g_eV_stc"], 1.121)


def test_I_sum_A_atol():
    assert constants.I_sum_A_atol == 1e-12

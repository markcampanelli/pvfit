import numpy as np

import pvfit.common.constants as constants


# Test constants for value changes. Constants are python scalars.

def test_q_C():
    assert isinstance(constants.q_C, float)
    np.testing.assert_array_equal(constants.q_C, 1.6021766208e-19)


def test_k_B_J_per_K():
    assert isinstance(constants.k_B_J_per_K, float)
    np.testing.assert_array_equal(constants.k_B_J_per_K, 1.38064852e-23)


def test_k_B_eV_per_K():
    assert isinstance(constants.k_B_eV_per_K, float)
    np.testing.assert_array_equal(constants.k_B_eV_per_K, 8.6173303e-5)


def test_T_degC_stc():
    assert isinstance(constants.T_degC_stc, float)
    np.testing.assert_array_equal(constants.T_degC_stc, 25.)


def test_T_K_stc():
    assert isinstance(constants.T_K_stc, float)
    np.testing.assert_array_equal(constants.T_K_stc, 298.15)


def test_G_hemi_W_per_m2_stc():
    assert isinstance(constants.G_hemi_W_per_m2_stc, float)
    np.testing.assert_array_equal(constants.G_hemi_W_per_m2_stc, 1000.)


def test_materials():
    assert isinstance(constants.materials['CIGS']['E_g_eV_stc'], float)
    np.testing.assert_array_equal(constants.materials['CIGS']['E_g_eV_stc'], 1.15)

    assert isinstance(constants.materials['CIS']['E_g_eV_stc'], float)
    np.testing.assert_array_equal(constants.materials['CIS']['E_g_eV_stc'], 1.010)

    assert isinstance(constants.materials['CdTe']['E_g_eV_stc'], float)
    np.testing.assert_array_equal(constants.materials['CdTe']['E_g_eV_stc'], 1.475)

    assert isinstance(constants.materials['GaAs']['E_g_eV_stc'], float)
    np.testing.assert_array_equal(constants.materials['GaAs']['E_g_eV_stc'], 1.43)

    assert isinstance(constants.materials['x-Si']['E_g_eV_stc'], float)
    np.testing.assert_array_equal(constants.materials['x-Si']['E_g_eV_stc'], 1.121)

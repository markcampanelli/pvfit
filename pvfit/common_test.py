"""
PVfit testing: Common items.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import pvfit.common as common


def test_q_C():
    assert isinstance(common.q_C, float)
    assert common.q_C == 1.602176634e-19


def test_k_B_J_per_K():
    assert isinstance(common.k_B_J_per_K, float)
    assert common.k_B_J_per_K == 1.380649e-23


def test_k_B_eV_per_K():
    assert isinstance(common.k_B_eV_per_K, float)
    assert common.k_B_eV_per_K == 8.617333262e-05


def test_c_m_per_s():
    assert common.c_m_per_s == 299792458.0


def test_h_J_s():
    assert common.h_J_s == 6.62607015e-34


def test_T_degC_abs_zero():
    assert isinstance(common.T_degC_abs_zero, float)
    assert common.T_degC_abs_zero == -273.15


def test_T_degC_stc():
    assert isinstance(common.T_degC_stc, float)
    assert common.T_degC_stc == 25.0


def test_T_K_stc():
    assert isinstance(common.T_K_stc, float)
    assert common.T_K_stc == 298.15


def test_E_hemispherical_tilted_W_per_m2_stc():
    assert isinstance(common.E_hemispherical_tilted_W_per_m2_stc, float)
    assert common.E_hemispherical_tilted_W_per_m2_stc == 1000.0

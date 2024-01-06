"""
PVfit testing: Constants.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy
import pytest
from scipy.constants import convert_temperature

import pvfit.modeling.dc.common as common


def test_T_degC_stc():
    assert isinstance(common.T_degC_stc, float)
    assert common.T_degC_stc == 25.0


def test_T_K_stc():
    assert isinstance(common.T_K_stc, float)
    assert common.T_K_stc == 298.15


def test_G_hemi_W_per_m2_stc():
    assert isinstance(common.G_hemi_W_per_m2_stc, float)
    numpy.testing.assert_array_equal(common.G_hemi_W_per_m2_stc, 1000.0)


def test_materials():
    assert isinstance(common.MATERIALS["CIGS"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(common.MATERIALS["CIGS"]["E_g_eV_stc"], 1.15)

    assert isinstance(common.MATERIALS["CIS"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(common.MATERIALS["CIS"]["E_g_eV_stc"], 1.010)

    assert isinstance(common.MATERIALS["CdTe"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(common.MATERIALS["CdTe"]["E_g_eV_stc"], 1.475)

    assert isinstance(common.MATERIALS["GaAs"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(common.MATERIALS["GaAs"]["E_g_eV_stc"], 1.43)

    assert isinstance(common.MATERIALS["mono-Si"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(common.MATERIALS["mono-Si"]["E_g_eV_stc"], 1.121)

    assert isinstance(common.MATERIALS["multi-Si"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(common.MATERIALS["multi-Si"]["E_g_eV_stc"], 1.121)

    assert isinstance(common.MATERIALS["poly-Si"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(common.MATERIALS["poly-Si"]["E_g_eV_stc"], 1.121)

    assert isinstance(common.MATERIALS["x-Si"]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(common.MATERIALS["x-Si"]["E_g_eV_stc"], 1.121)


def test_N_IC_MIN():
    assert isinstance(common.N_IC_MIN, float)
    numpy.testing.assert_array_equal(common.N_IC_MIN, 1.0)


def test_N_IC_MAX():
    assert isinstance(common.N_IC_MAX, float)
    numpy.testing.assert_array_equal(common.N_IC_MAX, 2.0)


@pytest.fixture(
    params=[
        {
            "given": {
                "N_s": 72,
                "T_degC": common.T_degC_stc,
            },
            "expected": {
                "scaled_thermal_voltage": numpy.array(
                    72
                    * (
                        common.k_B_J_per_K
                        * convert_temperature(common.T_degC_stc, "Celsius", "Kelvin")
                        / common.q_C
                    )
                )
            },
        },
        {
            "given": {
                "N_s": 60,
                "T_degC": numpy.array(
                    [common.T_degC_stc - 1, common.T_degC_stc, common.T_degC_stc + 1]
                ),
            },
            "expected": {
                "scaled_thermal_voltage": 60
                * (
                    common.k_B_J_per_K
                    * convert_temperature(
                        numpy.array(
                            [
                                common.T_degC_stc - 1,
                                common.T_degC_stc,
                                common.T_degC_stc + 1,
                            ]
                        ),
                        "Celsius",
                        "Kelvin",
                    )
                    / common.q_C
                )
            },
        },
        {
            "given": {"N_s": numpy.array([60, 72, 192]), "T_degC": common.T_degC_stc},
            "expected": {
                "scaled_thermal_voltage": numpy.array([60, 72, 192])
                * (
                    common.k_B_J_per_K
                    * convert_temperature(
                        common.T_degC_stc,
                        "Celsius",
                        "Kelvin",
                    )
                    / common.q_C
                )
            },
        },
        {
            "given": {
                "N_s": numpy.array([60, 72, 192]),
                "T_degC": numpy.array(
                    [
                        common.T_degC_stc - 1,
                        common.T_degC_stc,
                        common.T_degC_stc + 1,
                    ]
                ),
            },
            "expected": {
                "scaled_thermal_voltage": numpy.array([60, 72, 192])
                * (
                    common.k_B_J_per_K
                    * convert_temperature(
                        numpy.array(
                            [
                                common.T_degC_stc - 1,
                                common.T_degC_stc,
                                common.T_degC_stc + 1,
                            ]
                        ),
                        "Celsius",
                        "Kelvin",
                    )
                    / common.q_C
                )
            },
        },
    ],
)
def get_scaled_thermal_voltage_fixture(request):
    return request.param


def test_get_scaled_thermal_voltage(get_scaled_thermal_voltage_fixture):
    given = get_scaled_thermal_voltage_fixture["given"]
    expected = get_scaled_thermal_voltage_fixture["expected"]

    scaled_thermal_voltage_got = common.get_scaled_thermal_voltage(
        N_s=given["N_s"], T_degC=given["T_degC"]
    )
    scaled_thermal_voltage_expected = expected["scaled_thermal_voltage"]

    assert isinstance(scaled_thermal_voltage_got, type(scaled_thermal_voltage_expected))
    assert scaled_thermal_voltage_got.shape == scaled_thermal_voltage_expected.shape
    assert scaled_thermal_voltage_got.dtype == scaled_thermal_voltage_expected.dtype
    numpy.testing.assert_allclose(
        scaled_thermal_voltage_got, scaled_thermal_voltage_expected
    )

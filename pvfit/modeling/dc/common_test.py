"""
PVfit testing: Common items for DC modeling.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy
import pytest
from scipy.constants import convert_temperature

from pvfit.common import T_degC_stc
from pvfit.modeling.dc import common


def test_materials():
    assert isinstance(common.MATERIALS_INFO[common.Material.CIGS]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(
        common.MATERIALS_INFO[common.Material.CIGS]["E_g_eV_stc"], 1.15
    )

    assert isinstance(common.MATERIALS_INFO[common.Material.CIS]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(
        common.MATERIALS_INFO[common.Material.CIS]["E_g_eV_stc"], 1.010
    )

    assert isinstance(common.MATERIALS_INFO[common.Material.CdTe]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(
        common.MATERIALS_INFO[common.Material.CdTe]["E_g_eV_stc"], 1.475
    )

    assert isinstance(common.MATERIALS_INFO[common.Material.GaAs]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(
        common.MATERIALS_INFO[common.Material.GaAs]["E_g_eV_stc"], 1.43
    )

    assert isinstance(
        common.MATERIALS_INFO[common.Material.monoSi]["E_g_eV_stc"], float
    )
    numpy.testing.assert_array_equal(
        common.MATERIALS_INFO[common.Material.monoSi]["E_g_eV_stc"], 1.121
    )

    assert isinstance(
        common.MATERIALS_INFO[common.Material.multiSi]["E_g_eV_stc"], float
    )
    numpy.testing.assert_array_equal(
        common.MATERIALS_INFO[common.Material.multiSi]["E_g_eV_stc"], 1.121
    )

    assert isinstance(
        common.MATERIALS_INFO[common.Material.polySi]["E_g_eV_stc"], float
    )
    numpy.testing.assert_array_equal(
        common.MATERIALS_INFO[common.Material.polySi]["E_g_eV_stc"], 1.121
    )

    assert isinstance(common.MATERIALS_INFO[common.Material.xSi]["E_g_eV_stc"], float)
    numpy.testing.assert_array_equal(
        common.MATERIALS_INFO[common.Material.xSi]["E_g_eV_stc"], 1.121
    )


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
                "T_degC": T_degC_stc,
            },
            "expected": {
                "scaled_thermal_voltage": numpy.array(
                    72
                    * (
                        common.k_B_J_per_K
                        * convert_temperature(T_degC_stc, "Celsius", "Kelvin")
                        / common.q_C
                    )
                )
            },
        },
        {
            "given": {
                "N_s": 60,
                "T_degC": numpy.array([T_degC_stc - 1, T_degC_stc, T_degC_stc + 1]),
            },
            "expected": {
                "scaled_thermal_voltage": 60
                * (
                    common.k_B_J_per_K
                    * convert_temperature(
                        numpy.array(
                            [
                                T_degC_stc - 1,
                                T_degC_stc,
                                T_degC_stc + 1,
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
            "given": {"N_s": numpy.array([60, 72, 192]), "T_degC": T_degC_stc},
            "expected": {
                "scaled_thermal_voltage": numpy.array([60, 72, 192])
                * (
                    common.k_B_J_per_K
                    * convert_temperature(
                        T_degC_stc,
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
                        T_degC_stc - 1,
                        T_degC_stc,
                        T_degC_stc + 1,
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
                                T_degC_stc - 1,
                                T_degC_stc,
                                T_degC_stc + 1,
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

    assert isinstance(
        scaled_thermal_voltage_got, type(expected["scaled_thermal_voltage"])
    )
    assert scaled_thermal_voltage_got.shape == expected["scaled_thermal_voltage"].shape
    assert scaled_thermal_voltage_got.dtype == expected["scaled_thermal_voltage"].dtype
    numpy.testing.assert_allclose(
        scaled_thermal_voltage_got, expected["scaled_thermal_voltage"]
    )

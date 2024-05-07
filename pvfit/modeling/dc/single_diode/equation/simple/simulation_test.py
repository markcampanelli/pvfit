"""
PVfit testing: Single-diode equation (SDE) simulation.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy
import pytest
from scipy.constants import convert_temperature

from pvfit.common import T_K_stc, T_degC_stc, k_B_J_per_K, q_C
from pvfit.measurement.iv.types import IVCurveParametersArray, IVData
from pvfit.modeling.dc.common import get_scaled_thermal_voltage
import pvfit.modeling.dc.single_diode.equation.simple.simulation as simulation
from pvfit.modeling.dc.single_diode.equation.simple.types import ModelParameters


@pytest.fixture(
    # Not necessarily I-V curve solutions.
    params=[
        {  # Can handle all python scalar inputs.
            "given": {
                "iv_curve": IVData(V_V=0.5, I_A=3.0),
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=7.0,
                    I_rs_A=6.0e-7,
                    n=1.25,
                    R_s_Ohm=0.1,
                    G_p_S=0.005,
                ),
            },
            "expected": {
                "I_sum_A": numpy.array(
                    7.0
                    - 6.0e-7
                    * numpy.expm1(
                        q_C * (0.5 + 3.0 * 0.1) / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                    )
                    - 0.005 * (0.5 + 3.0 * 0.1)
                    - 3.0,
                ),
            },
        },
        {  # Can handle all rank-0 array inputs.
            "given": {
                "iv_curve": IVData(V_V=numpy.array(0.5), I_A=numpy.array(3.0)),
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=numpy.array(7.0),
                    I_rs_A=numpy.array(6.0e-7),
                    n=numpy.array(1.25),
                    R_s_Ohm=numpy.array(0.1),
                    G_p_S=numpy.array(0.005),
                ),
            },
            "expected": {
                "I_sum_A": numpy.array(
                    7.0
                    - 6.0e-7
                    * numpy.expm1(
                        q_C * (0.5 + 3.0 * 0.1) / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                    )
                    - 0.005 * (0.5 + 3.0 * 0.1)
                    - 3.0
                ),
            },
        },
        {  # Can handle all rank-1 singleton array inputs.
            "given": {
                "iv_curve": IVData(
                    V_V=numpy.array([0.5]),
                    I_A=numpy.array([3.0]),
                ),
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=numpy.array([7.0]),
                    I_rs_A=numpy.array([6.0e-7]),
                    n=numpy.array([1.25]),
                    R_s_Ohm=numpy.array([0.1]),
                    G_p_S=numpy.array([0.005]),
                ),
            },
            "expected": {
                "I_sum_A": numpy.array(
                    [
                        7.0
                        - 6.0e-7
                        * numpy.expm1(
                            q_C * (0.5 + 3.0 * 0.1) / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                        )
                        - 0.005 * (0.5 + 3.0 * 0.1)
                        - 3.0
                    ]
                ),
            },
        },
        {  # Can handle all rank-1 non-singleton array inputs.
            "given": {
                "iv_curve": IVData(
                    V_V=numpy.array([0.5, 0.0, 0.0]),
                    I_A=numpy.array([0.0, 0.0, 3.0]),
                ),
                "model_parameters": ModelParameters(
                    N_s=numpy.array([1, 60, 96]),
                    T_degC=numpy.array([T_degC_stc / 2, T_degC_stc, 2 * T_degC_stc]),
                    I_ph_A=numpy.array([7.0, 7.0, 7.0]),
                    I_rs_A=numpy.array([6.0e-7, 6.0e-7, 6.0e-7]),
                    n=numpy.array([1.25, 1.25, 1.25]),
                    R_s_Ohm=numpy.array([0.1, 0.1, 0.1]),
                    G_p_S=numpy.array([0.005, 0.005, 0.005]),
                ),
            },
            "expected": {
                "I_sum_A": numpy.array(
                    [
                        7.0
                        - 6.0e-7
                        * numpy.expm1(
                            q_C
                            * 0.5
                            / (
                                1
                                * 1.25
                                * k_B_J_per_K
                                * convert_temperature(
                                    T_degC_stc / 2, "Celsius", "Kelvin"
                                )
                            )
                        )
                        - 0.005 * 0.5,
                        7.0,
                        7.0
                        - 6.0e-7
                        * numpy.expm1(
                            q_C
                            * 3.0
                            * 0.1
                            / (
                                96
                                * 1.25
                                * k_B_J_per_K
                                * convert_temperature(
                                    2 * T_degC_stc, "Celsius", "Kelvin"
                                )
                            )
                        )
                        - 0.005 * 3.0 * 0.1
                        - 3.0,
                    ]
                ),
            },
        },
        {  # Can handle mixed inputs with python floats.
            "given": {
                "iv_curve": IVData(
                    V_V=numpy.array([0.5, 0.0, 0.0]), I_A=numpy.array([0.0, 0.0, 3.0])
                ),
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=7.0,
                    I_rs_A=6.0e-7,
                    n=1.25,
                    R_s_Ohm=0.1,
                    G_p_S=0.005,
                ),
            },
            "expected": {
                "I_sum_A": numpy.array(
                    [
                        7.0
                        - 6.0e-7
                        * numpy.expm1(q_C * 0.5 / (1 * 1.25 * k_B_J_per_K * T_K_stc))
                        - 0.005 * 0.5,
                        7.0,
                        7.0
                        - 6.0e-7
                        * numpy.expm1(
                            q_C * 3.0 * 0.1 / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                        )
                        - 0.005 * 3.0 * 0.1
                        - 3.0,
                    ]
                ),
            },
        },
        {  # Can handle mixed inputs with rank-2 arrays.
            "given": {
                "iv_curve": IVData(
                    V_V=numpy.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.5]]),
                    I_A=numpy.array([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]),
                ),
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=7.0,
                    I_rs_A=numpy.full((1, 3), 6.0e-7),
                    n=numpy.array(1.25),
                    R_s_Ohm=numpy.array([0.1]),
                    G_p_S=numpy.full((2, 3), 0.005),
                ),
            },
            "expected": {
                "I_sum_A": numpy.array(
                    [
                        [
                            7.0
                            - 6.0e-7
                            * numpy.expm1(
                                q_C * 0.5 / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                            )
                            - 0.005 * 0.5,
                            7.0,
                            7.0
                            - 6.0e-7
                            * numpy.expm1(
                                q_C * 3.0 * 0.1 / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                            )
                            - 0.005 * 3.0 * 0.1
                            - 3.0,
                        ],
                        [
                            7.0
                            - 6.0e-7
                            * numpy.expm1(
                                q_C * 3.0 * 0.1 / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                            )
                            - 0.005 * 3.0 * 0.1
                            - 3.0,
                            7.0,
                            7.0
                            - 6.0e-7
                            * numpy.expm1(
                                q_C * 0.5 / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                            )
                            - 0.005 * 0.5,
                        ],
                    ]
                ),
            },
        },
        {  # Can handle mixed inputs and zero shunt conductance with positive series resistance.
            "given": {
                "iv_curve": IVData(V_V=0.5, I_A=3.0),
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=7.0,
                    I_rs_A=6.0e-7,
                    n=numpy.array(1.25),
                    R_s_Ohm=0.1,
                    G_p_S=0.0,
                ),
            },
            "expected": {
                "I_sum_A": numpy.array(
                    7.0
                    - 6.0e-7
                    * numpy.expm1(
                        q_C * (0.5 + 3.0 * 0.1) / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                    )
                    - 3.0
                ),
            },
        },
        {  # Can handle mixed inputs and zero shunt conductance with zero series resistance
            "given": {
                "iv_curve": IVData(V_V=0.5, I_A=3.0),
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=7.0,
                    I_rs_A=6.0e-7,
                    n=numpy.array([1.25]),
                    R_s_Ohm=0.0,
                    G_p_S=0.0,
                ),
            },
            "expected": {
                "I_sum_A": numpy.array(
                    [
                        7.0
                        - 6.0e-7
                        * numpy.expm1(q_C * 0.5 / (1 * 1.25 * k_B_J_per_K * T_K_stc))
                        - 3.0
                    ]
                )
            },
        },
    ]
)
def current_sum_at_diode_node_fixture(request):
    return request.param


def test_I_sum_diode_anode_at_I_V(current_sum_at_diode_node_fixture):
    # Note: The computation of this function is so straight forward that  we do NOT
    # extensively verify ufunc behavior.
    given = current_sum_at_diode_node_fixture["given"]
    expected = current_sum_at_diode_node_fixture["expected"]

    I_sum_diode_anode_A_got = simulation.I_sum_diode_anode_at_I_V(
        iv_data=given["iv_curve"],
        model_parameters=given["model_parameters"],
    )["I_sum_diode_anode_A"]
    I_sum_diode_anode_A_expected = expected["I_sum_A"]

    assert isinstance(I_sum_diode_anode_A_got, type(I_sum_diode_anode_A_expected))
    assert I_sum_diode_anode_A_got.shape == I_sum_diode_anode_A_expected.shape
    assert I_sum_diode_anode_A_got.dtype == I_sum_diode_anode_A_expected.dtype
    numpy.testing.assert_array_almost_equal(
        I_sum_diode_anode_A_got, I_sum_diode_anode_A_expected
    )


def test_I_at_V_explicit():
    # Can handle zero series resistance.
    V_V = 0.35
    N_s = 1
    T_degC = T_degC_stc
    I_ph_A = 0.125
    I_rs_A = 9.24e-7
    n = 1.5
    R_s_Ohm = 0.0
    G_p_S = 0.001
    model_parameters = ModelParameters(
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_A=I_rs_A,
        n=n,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )

    I_A_got = simulation.I_at_V(V_V=V_V, model_parameters=model_parameters)["I_A"]
    I_A_expected = numpy.array(
        I_ph_A
        - I_rs_A * numpy.expm1(q_C * V_V / (N_s * n * k_B_J_per_K * T_K_stc))
        - G_p_S * V_V
    )

    assert isinstance(I_A_got, type(I_A_expected))
    assert I_A_got.shape == I_A_expected.shape
    assert I_A_got.dtype == I_A_expected.dtype
    numpy.testing.assert_array_almost_equal(I_A_got, I_A_expected)


def test_I_at_V_implicit():
    # Implicit computation checks out when chaining with inverse function.
    V_V = numpy.array(0.35)
    N_s = 1
    T_degC = T_degC_stc
    I_ph_A = 0.125
    I_rs_A = 9.24e-7
    n = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.001
    model_parameters = ModelParameters(
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_A=I_rs_A,
        n=n,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )

    I_A_got = simulation.I_at_V(V_V=V_V, model_parameters=model_parameters)["I_A"]
    I_A_expected = numpy.array(0.09301675954356346)

    assert isinstance(I_A_got, type(I_A_expected))
    assert I_A_got.shape == I_A_expected.shape
    assert I_A_got.dtype == I_A_expected.dtype
    numpy.testing.assert_array_almost_equal(I_A_got, I_A_expected)

    # Check "full circle" computation back to V_V.
    V_V_got = simulation.V_at_I(I_A=I_A_got, model_parameters=model_parameters)["V_V"]

    numpy.testing.assert_array_almost_equal(V_V_got, V_V)


def test_V_at_I_explicit():
    # Explicit solution when G_p_S==0.
    I_A = 0.1
    N_s = 1
    T_degC = T_degC_stc
    I_ph_A = 0.125
    I_rs_A = 9.24e-7
    n = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.0
    model_parameters = ModelParameters(
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_A=I_rs_A,
        n=n,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )

    V_V_got = simulation.V_at_I(I_A=I_A, model_parameters=model_parameters)["V_V"]
    V_V_expected = numpy.array(
        N_s
        * n
        * k_B_J_per_K
        * T_K_stc
        / q_C
        * (numpy.log(I_ph_A - I_A + I_rs_A) - numpy.log(I_rs_A))
        - I_A * R_s_Ohm
    )

    assert isinstance(V_V_got, type(V_V_expected))
    assert V_V_got.shape == V_V_expected.shape
    assert V_V_got.dtype == V_V_expected.dtype
    numpy.testing.assert_array_equal(V_V_got, V_V_expected)


def test_V_at_I_implicit():
    # Implicit computation checks out when chaining with inverse function.
    I_A = numpy.array(0.1)
    I_ph_A = 0.125
    N_s = 1
    T_degC = T_degC_stc
    I_rs_A = 9.24e-7
    n = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.001
    model_parameters = ModelParameters(
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_A=I_rs_A,
        n=n,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )

    V_V_got = simulation.V_at_I(I_A=I_A, model_parameters=model_parameters)["V_V"]
    V_V_expected = numpy.array(0.33645640565779594)

    assert isinstance(V_V_got, type(V_V_expected))
    assert V_V_got.shape == V_V_expected.shape
    assert V_V_got.dtype == V_V_expected.dtype
    numpy.testing.assert_array_equal(V_V_got, V_V_expected)

    I_A_got = simulation.I_at_V(V_V=V_V_got, model_parameters=model_parameters)["I_A"]

    # Check "full circle" computation back to I_A.
    numpy.testing.assert_array_almost_equal(I_A_got, I_A)


def test_dV_dI_at_I_explicit():
    # Explicit solution when G_p_S==0.
    I_A = numpy.array(0.1)
    N_s = 1
    T_degC = T_degC_stc
    I_ph_A = 0.125
    I_rs_A = 9.24e-7
    n = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.0
    model_parameters = ModelParameters(
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_A=I_rs_A,
        n=n,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )

    got = simulation.dV_dI_at_I(I_A=I_A, model_parameters=model_parameters)
    V_V_expected = numpy.array(
        N_s
        * n
        * k_B_J_per_K
        * T_K_stc
        / q_C
        * (numpy.log(I_ph_A - I_A + I_rs_A) - numpy.log(I_rs_A))
        - I_A * R_s_Ohm
    )
    n_1_mod_V = model_parameters["n"] * get_scaled_thermal_voltage(
        N_s=N_s, T_degC=T_degC
    )
    # Compute first derivative of voltage with respect to current at specified current.
    dV_dI_Ohm_expected = numpy.array(
        -1.0
        / (
            I_rs_A / n_1_mod_V * numpy.exp((V_V_expected + I_A * R_s_Ohm) / n_1_mod_V)
            + G_p_S
        )
        - R_s_Ohm
    )

    assert isinstance(got["dV_dI_Ohm"], type(dV_dI_Ohm_expected))
    assert got["dV_dI_Ohm"].shape == dV_dI_Ohm_expected.shape
    assert got["dV_dI_Ohm"].dtype == dV_dI_Ohm_expected.dtype
    numpy.testing.assert_array_equal(got["dV_dI_Ohm"], dV_dI_Ohm_expected)

    assert isinstance(got["V_V"], type(V_V_expected))
    assert got["V_V"].shape == V_V_expected.shape
    assert got["V_V"].dtype == V_V_expected.dtype
    numpy.testing.assert_array_equal(got["V_V"], V_V_expected)


def test_P_at_V_explicit():
    # Can handle zero series resistance.
    V_V = numpy.array(0.35)
    N_s = 1
    T_degC = T_degC_stc
    I_ph_A = 0.125
    I_rs_A = 9.24e-7
    n = 1.5
    R_s_Ohm = 0.0
    G_p_S = 0.001
    model_parameters = ModelParameters(
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_A=I_rs_A,
        n=n,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )

    got = simulation.P_at_V(V_V=V_V, model_parameters=model_parameters)
    P_W_expected = numpy.array(
        V_V
        * (
            I_ph_A
            - I_rs_A * numpy.expm1(q_C * V_V / (N_s * n * k_B_J_per_K * T_K_stc))
            - G_p_S * V_V
        )
    )

    assert isinstance(got["P_W"], type(P_W_expected))
    assert got["P_W"].shape == P_W_expected.shape
    assert got["P_W"].dtype == P_W_expected.dtype
    numpy.testing.assert_array_equal(got["P_W"], P_W_expected)


@pytest.fixture(
    params=[
        {  # Happy path for a function that touches many others, scalar case.
            "given": {
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=0.125,
                    I_rs_A=9.24e-7,
                    n=1.5,
                    R_s_Ohm=0.5625,
                    G_p_S=0.001,
                )
            },
            "expected": {
                "iv_curve_parameters": IVCurveParametersArray(
                    I_sc_A=numpy.array(0.12492493),
                    R_sc_Ohm=numpy.array(871.28357523),
                    V_x_V=numpy.array(0.22760037),
                    I_x_A=numpy.array(0.12267153),
                    V_mp_V=numpy.array(0.31597361),
                    P_mp_W=numpy.array(0.03421952),
                    I_mp_A=numpy.array(0.10829866),
                    V_xx_V=numpy.array(0.38558718),
                    I_xx_A=numpy.array(0.06876379),
                    R_oc_Ohm=numpy.array(0.87183978),
                    V_oc_V=numpy.array(0.45520074),
                    FF=numpy.array(0.6017579),
                )
            },
        },
        {  # Happy path for a function that touches many others, rank-0 array case.
            "given": {
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=0.125,
                    I_rs_A=1.23e-6,
                    n=2.0,
                    R_s_Ohm=0.5625,
                    G_p_S=numpy.array(0.005),
                )
            },
            "expected": {
                "iv_curve_parameters": IVCurveParametersArray(
                    I_sc_A=numpy.array(0.12464585),
                    R_sc_Ohm=numpy.array(196.88421221),
                    V_x_V=numpy.array(0.29559654),
                    I_x_A=numpy.array(0.12171252),
                    V_mp_V=numpy.array(0.42294383),
                    P_mp_W=numpy.array(0.04550328),
                    I_mp_A=numpy.array(0.10758704),
                    V_xx_V=numpy.array(0.50706846),
                    I_xx_A=numpy.array(0.07075575),
                    R_oc_Ohm=numpy.array(0.98264912),
                    V_oc_V=numpy.array(0.59119309),
                    FF=numpy.array(0.61749791),
                )
            },
        },
        {  # Happy path for a function that touches many others, array case.
            "given": {
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=0.125,
                    I_rs_A=numpy.array([9.24e-7, 1.23e-6, 7.54e-11]),
                    n=1.75,
                    R_s_Ohm=0.5625,
                    G_p_S=0.001,
                )
            },
            "expected": {
                "iv_curve_parameters": IVCurveParametersArray(
                    I_sc_A=numpy.array([0.12492624, 0.12492509, 0.12492973]),
                    R_sc_Ohm=numpy.array([911.24231276, 885.07975543, 1000.55449615]),
                    V_x_V=numpy.array([0.26552007, 0.25909159, 0.47707191]),
                    I_x_A=numpy.array([0.12308449, 0.12285332, 0.12443842]),
                    V_mp_V=numpy.array([0.37612483, 0.36492711, 0.76227265]),
                    P_mp_W=numpy.array([0.04101074, 0.03961548, 0.08894652]),
                    I_mp_A=numpy.array([0.10903491, 0.10855723, 0.11668596]),
                    V_xx_V=numpy.array([0.45358249, 0.44155515, 0.85820824]),
                    I_xx_A=numpy.array([0.0706868, 0.07009548, 0.08274347]),
                    R_oc_Ohm=numpy.array([0.92359761, 0.92355944, 0.92483151]),
                    V_oc_V=numpy.array([0.53104015, 0.51818319, 0.95414383]),
                    FF=numpy.array([0.61818227, 0.6119725, 0.74618981]),
                )
            },
        },
        {  # Happy path for a function that touches many others, zero power case.
            "given": {
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC=T_degC_stc,
                    I_ph_A=0.0,
                    I_rs_A=9.24e-7,
                    n=1.75,
                    R_s_Ohm=0.5625,
                    G_p_S=0.001,
                )
            },
            "expected": {
                "iv_curve_parameters": IVCurveParametersArray(
                    I_sc_A=numpy.array(0.0),
                    R_sc_Ohm=numpy.array(980.42564499),
                    V_x_V=numpy.array(0.0),
                    I_x_A=numpy.array(0.0),
                    V_mp_V=numpy.array(0.0),
                    P_mp_W=numpy.array(0.0),
                    I_mp_A=numpy.array(0.0),
                    V_xx_V=numpy.array(0.0),
                    I_xx_A=numpy.array(0.0),
                    R_oc_Ohm=numpy.array(980.42564499),
                    V_oc_V=numpy.array(0.0),
                    FF=numpy.array(float("nan")),
                )
            },
        },
    ]
)
def iv_parameters_fixture(request):
    return request.param


def test_iv_parameters(iv_parameters_fixture):
    given = iv_parameters_fixture["given"]
    expected = iv_parameters_fixture["expected"]

    iv_curve_parameters_got = simulation.iv_curve_parameters(
        model_parameters=given["model_parameters"]
    )
    iv_curve_parameters_expected = expected["iv_curve_parameters"]

    assert set(iv_curve_parameters_got.keys()) == set(
        iv_curve_parameters_expected.keys()
    )

    for key in iv_curve_parameters_got:
        assert isinstance(
            iv_curve_parameters_got[key], type(iv_curve_parameters_expected[key])
        )
        assert (
            iv_curve_parameters_got[key].shape
            == iv_curve_parameters_expected[key].shape
        )
        assert (
            iv_curve_parameters_got[key].dtype
            == iv_curve_parameters_expected[key].dtype
        )
        numpy.testing.assert_allclose(
            iv_curve_parameters_got[key], iv_curve_parameters_expected[key]
        )

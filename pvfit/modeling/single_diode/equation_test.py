import numpy
import pytest
from scipy.constants import convert_temperature

from pvfit.common.constants import T_K_stc, T_degC_stc, k_B_J_per_K, q_C
import pvfit.modeling.single_diode.equation as equation


@pytest.fixture(
    # Not necessarily I-V curve solutions.
    params=[
        {  # Can handle all python scalar inputs.
            "V_V": 0.5,
            "I_A": 3.0,
            "I_ph_A": 7.0,
            "I_rs_1_A": 6.0e-7,
            "n_1": 1.25,
            "R_s_Ohm": 0.1,
            "G_p_S": 0.005,
            "N_s": 1,
            "T_degC": T_degC_stc,
            "I_sum_A_expected": 7.0
            - 6.0e-7
            * numpy.expm1(q_C * (0.5 + 3.0 * 0.1) / (1 * 1.25 * k_B_J_per_K * T_K_stc))
            - 0.005 * (0.5 + 3.0 * 0.1)
            - 3.0,
            "T_K_expected": T_K_stc,
            "V_1_V_expected": numpy.float64(0.5 + 3.0 * 0.1),
            "n_1_mod_V_expected": numpy.float64(
                (1 * 1.25 * k_B_J_per_K * T_K_stc) / q_C
            ),
        },
        {  # Can handle all rank-0 array inputs.
            "V_V": numpy.array(0.5),
            "I_A": numpy.array(3.0),
            "I_ph_A": numpy.array(7.0),
            "I_rs_1_A": numpy.array(6.0e-7),
            "n_1": numpy.array(1.25),
            "R_s_Ohm": numpy.array(0.1),
            "G_p_S": numpy.array(0.005),
            "N_s": 1,
            "T_degC": T_degC_stc,
            "I_sum_A_expected": numpy.float64(
                7.0
                - 6.0e-7
                * numpy.expm1(
                    q_C * (0.5 + 3.0 * 0.1) / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                )
                - 0.005 * (0.5 + 3.0 * 0.1)
                - 3.0
            ),
            "T_K_expected": numpy.float64(T_K_stc),
            "V_1_V_expected": numpy.float64(0.5 + 3.0 * 0.1),
            "n_1_mod_V_expected": numpy.float64(
                (1 * 1.25 * k_B_J_per_K * T_K_stc) / q_C
            ),
        },
        {  # Can handle all rank-1 singleton array inputs.
            "V_V": numpy.array([0.5]),
            "I_A": numpy.array([3.0]),
            "I_ph_A": numpy.array([7.0]),
            "I_rs_1_A": numpy.array([6.0e-7]),
            "n_1": numpy.array([1.25]),
            "R_s_Ohm": numpy.array([0.1]),
            "G_p_S": numpy.array([0.005]),
            "N_s": 1,
            "T_degC": T_degC_stc,
            "I_sum_A_expected": numpy.array(
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
            "T_K_expected": numpy.float64(T_K_stc),
            "V_1_V_expected": numpy.array([0.5 + 3.0 * 0.1]),
            "n_1_mod_V_expected": numpy.array(
                [(1 * 1.25 * k_B_J_per_K * T_K_stc) / q_C]
            ),
        },
        {  # Can handle all rank-1 non-singleton array inputs.
            "V_V": numpy.array([0.5, 0.0, 0.0]),
            "I_A": numpy.array([0.0, 0.0, 3.0]),
            "I_ph_A": numpy.array([7.0, 7.0, 7.0]),
            "I_rs_1_A": numpy.array([6.0e-7, 6.0e-7, 6.0e-7]),
            "n_1": numpy.array([1.25, 1.25, 1.25]),
            "R_s_Ohm": numpy.array([0.1, 0.1, 0.1]),
            "G_p_S": numpy.array([0.005, 0.005, 0.005]),
            "N_s": numpy.array([1, 60, 96]),
            "T_degC": numpy.array([T_degC_stc / 2, T_degC_stc, 2 * T_degC_stc]),
            "I_sum_A_expected": numpy.array(
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
                            * convert_temperature(T_degC_stc / 2, "Celsius", "Kelvin")
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
                            * convert_temperature(2 * T_degC_stc, "Celsius", "Kelvin")
                        )
                    )
                    - 0.005 * 3.0 * 0.1
                    - 3.0,
                ]
            ),
            "T_K_expected": numpy.array(
                [273.15 + T_degC_stc / 2, T_K_stc, 273.15 + 2 * T_degC_stc]
            ),
            "V_1_V_expected": numpy.array([0.5, 0.0, 3.0 * 0.1]),
            "n_1_mod_V_expected": (
                numpy.array([1, 60, 96])
                * 1.25
                * k_B_J_per_K
                * convert_temperature(
                    numpy.array([T_degC_stc / 2, T_degC_stc, 2 * T_degC_stc]),
                    "Celsius",
                    "Kelvin",
                )
            )
            / q_C,
        },
        {  # Can handle mixed inputs with python floats.
            "V_V": numpy.array([0.5, 0.0, 0.0]),
            "I_A": numpy.array([0.0, 0.0, 3.0]),
            "I_ph_A": 7.0,
            "I_rs_1_A": 6.0e-7,
            "n_1": 1.25,
            "R_s_Ohm": 0.1,
            "G_p_S": 0.005,
            "N_s": 1,
            "T_degC": T_degC_stc,
            "I_sum_A_expected": numpy.array(
                [
                    7.0
                    - 6.0e-7
                    * numpy.expm1(q_C * 0.5 / (1 * 1.25 * k_B_J_per_K * T_K_stc))
                    - 0.005 * 0.5,
                    7.0,
                    7.0
                    - 6.0e-7
                    * numpy.expm1(q_C * 3.0 * 0.1 / (1 * 1.25 * k_B_J_per_K * T_K_stc))
                    - 0.005 * 3.0 * 0.1
                    - 3.0,
                ]
            ),
            "T_K_expected": numpy.float64(T_K_stc),
            "V_1_V_expected": numpy.array([0.5, 0.0, 3.0 * 0.1]),
            "n_1_mod_V_expected": numpy.float64(
                (1 * 1.25 * k_B_J_per_K * T_K_stc) / q_C
            ),
        },
        {  # Can handle mixed inputs with rank-2 arrays.
            "V_V": numpy.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.5]]),
            "I_A": numpy.array([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]),
            "I_ph_A": 7.0,
            "I_rs_1_A": numpy.full((1, 3), 6.0e-7),
            "n_1": numpy.array(1.25),
            "R_s_Ohm": numpy.array([0.1]),
            "G_p_S": numpy.full((2, 3), 0.005),
            "N_s": 1,
            "T_degC": T_degC_stc,
            "I_sum_A_expected": numpy.array(
                [
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
                        * numpy.expm1(q_C * 0.5 / (1 * 1.25 * k_B_J_per_K * T_K_stc))
                        - 0.005 * 0.5,
                    ],
                ]
            ),
            "T_K_expected": numpy.float64(T_K_stc),
            "V_1_V_expected": numpy.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])
            + numpy.array([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]) * numpy.array([0.1]),
            "n_1_mod_V_expected": numpy.float64(1 * 1.25 * k_B_J_per_K * T_K_stc) / q_C,
        },
        {  # Can handle mixed inputs and zero shunt conductance with positive series resistance.
            "V_V": 0.5,
            "I_A": 3.0,
            "I_ph_A": 7.0,
            "I_rs_1_A": 6.0e-7,
            "n_1": numpy.array(1.25),
            "R_s_Ohm": 0.1,
            "G_p_S": 0.0,
            "N_s": 1,
            "T_degC": T_degC_stc,
            "I_sum_A_expected": numpy.float64(
                7.0
                - 6.0e-7
                * numpy.expm1(
                    q_C * (0.5 + 3.0 * 0.1) / (1 * 1.25 * k_B_J_per_K * T_K_stc)
                )
                - 3.0
            ),
            "T_K_expected": numpy.float64(T_K_stc),
            "V_1_V_expected": numpy.float64(0.5 + 3 * 0.1),
            "n_1_mod_V_expected": numpy.float64(
                (1 * 1.25 * k_B_J_per_K * T_K_stc) / q_C
            ),
        },
        {  # Can handle mixed inputs and zero shunt conductance with zero series resistance
            "V_V": 0.5,
            "I_A": 3.0,
            "I_ph_A": 7.0,
            "I_rs_1_A": 6.0e-7,
            "n_1": numpy.array([1.25]),
            "R_s_Ohm": 0.0,
            "G_p_S": 0.0,
            "N_s": 1,
            "T_degC": T_degC_stc,
            "I_sum_A_expected": numpy.array(
                [
                    7.0
                    - 6.0e-7
                    * numpy.expm1(q_C * 0.5 / (1 * 1.25 * k_B_J_per_K * T_K_stc))
                    - 3.0
                ]
            ),
            "T_K_expected": numpy.float64(T_K_stc),
            "V_1_V_expected": numpy.float64(0.5),
            "n_1_mod_V_expected": numpy.array(
                [(1 * 1.25 * k_B_J_per_K * T_K_stc) / q_C]
            ),
        },
    ]
)
def current_sum_at_diode_node_fixture(request):
    return request.param


def test_current_sum_at_diode_node(current_sum_at_diode_node_fixture):
    # Note: The computation of this function is so straight forward that
    # we do NOT extensively verify ufunc behavior.

    # Solution set loaded from fixture.
    V_V = current_sum_at_diode_node_fixture["V_V"]
    I_A = current_sum_at_diode_node_fixture["I_A"]
    I_ph_A = current_sum_at_diode_node_fixture["I_ph_A"]
    I_rs_1_A = current_sum_at_diode_node_fixture["I_rs_1_A"]
    n_1 = current_sum_at_diode_node_fixture["n_1"]
    R_s_Ohm = current_sum_at_diode_node_fixture["R_s_Ohm"]
    G_p_S = current_sum_at_diode_node_fixture["G_p_S"]
    N_s = current_sum_at_diode_node_fixture["N_s"]
    T_degC = current_sum_at_diode_node_fixture["T_degC"]
    I_sum_A_expected = current_sum_at_diode_node_fixture["I_sum_A_expected"]
    T_K_expected = current_sum_at_diode_node_fixture["T_K_expected"]
    V_1_V_expected = current_sum_at_diode_node_fixture["V_1_V_expected"]
    n_1_mod_V_expected = current_sum_at_diode_node_fixture["n_1_mod_V_expected"]

    result = equation.current_sum_at_diode_node(
        V_V=V_V,
        I_A=I_A,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )

    assert isinstance(result["I_sum_A"], type(I_sum_A_expected))
    assert result["I_sum_A"].dtype == I_sum_A_expected.dtype
    numpy.testing.assert_array_almost_equal(result["I_sum_A"], I_sum_A_expected)

    assert isinstance(result["T_K"], type(T_K_expected))
    assert result["T_K"].dtype == T_K_expected.dtype
    numpy.testing.assert_array_almost_equal(result["T_K"], T_K_expected)

    assert isinstance(result["V_1_V"], type(V_1_V_expected))
    assert result["V_1_V"].dtype == V_1_V_expected.dtype
    numpy.testing.assert_array_almost_equal(result["V_1_V"], V_1_V_expected)

    assert isinstance(result["n_1_mod_V"], type(n_1_mod_V_expected))
    assert result["n_1_mod_V"].dtype == n_1_mod_V_expected.dtype
    numpy.testing.assert_array_almost_equal(result["n_1_mod_V"], n_1_mod_V_expected)


def test_I_at_V_explicit():

    # Can handle zero series resistance.
    V_V = 0.35
    I_ph_A = 0.125
    I_rs_1_A = 9.24e-7
    n_1 = 1.5
    R_s_Ohm = 0.0
    G_p_S = 0.001
    N_s = 1
    T_degC = T_degC_stc
    I_A_expected = (
        I_ph_A
        - I_rs_1_A * numpy.expm1(q_C * V_V / (N_s * n_1 * k_B_J_per_K * T_K_stc))
        - G_p_S * V_V
    )

    I_A = equation.I_at_V(
        V_V=V_V,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )["I_A"]
    assert isinstance(I_A, type(I_A_expected))
    assert I_A.dtype == I_A_expected.dtype
    numpy.testing.assert_array_almost_equal(I_A, I_A_expected)


def test_I_at_V_implicit():

    # Implicit computation checks out when chaining with inverse function.
    V_V = numpy.array(0.35)
    I_ph_A = 0.125
    I_rs_1_A = 9.24e-7
    n_1 = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.001
    N_s = 1
    T_degC = T_degC_stc
    I_A_type_expected = numpy.float64
    V_V_expected = V_V

    I_A = equation.I_at_V(
        V_V=V_V,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )["I_A"]
    assert isinstance(I_A, I_A_type_expected)
    assert I_A.dtype == V_V.dtype

    V_V_inv_comp = equation.V_at_I(
        I_A=I_A,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )["V_V"]
    numpy.testing.assert_array_almost_equal(V_V_inv_comp, V_V_expected)


def test_V_at_I_explicit():

    # Explicit solution when G_p_S==0.
    I_A = 0.1
    I_ph_A = 0.125
    I_rs_1_A = 9.24e-7
    n_1 = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.0
    N_s = 1
    T_degC = T_degC_stc
    V_V_expected = (
        N_s
        * n_1
        * k_B_J_per_K
        * T_K_stc
        / q_C
        * (numpy.log(I_ph_A - I_A + I_rs_1_A) - numpy.log(I_rs_1_A))
        - I_A * R_s_Ohm
    )

    V_V = equation.V_at_I(
        I_A=I_A,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )["V_V"]
    assert isinstance(V_V, type(V_V_expected))
    assert V_V.dtype == V_V_expected.dtype
    numpy.testing.assert_array_equal(V_V, V_V_expected)


def test_V_at_I_implicit():

    # Implicit computation checks out when chaining with inverse function.
    I_A = numpy.array(0.1)
    I_ph_A = 0.125
    I_rs_1_A = 9.24e-7
    n_1 = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.001
    N_s = 1
    T_degC = T_degC_stc
    V_V_type_expected = numpy.float64
    I_A_expected = I_A

    V_V = equation.V_at_I(
        I_A=I_A,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )["V_V"]
    assert isinstance(V_V, V_V_type_expected)
    assert V_V.dtype == I_A.dtype

    I_A_inv_comp = equation.I_at_V(
        V_V=V_V,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )["I_A"]

    numpy.testing.assert_array_almost_equal(I_A_inv_comp, I_A_expected)


def test_V_at_I_d1_explicit():

    # Explicit solution when G_p_S==0.
    I_A = numpy.array(0.1)
    I_ph_A = 0.125
    I_rs_1_A = 9.24e-7
    n_1 = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.0
    N_s = 1
    T_degC = T_degC_stc
    V_V_expected = numpy.float64(
        N_s
        * n_1
        * k_B_J_per_K
        * T_K_stc
        / q_C
        * (numpy.log(I_ph_A - I_A + I_rs_1_A) - numpy.log(I_rs_1_A))
        - I_A * R_s_Ohm
    )

    result = equation.V_at_I_d1(
        I_A=I_A,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )
    assert isinstance(result["V_V"], type(V_V_expected))
    assert result["V_V"].dtype == V_V_expected.dtype
    numpy.testing.assert_array_equal(result["V_V"], V_V_expected)


def test_P_at_V_explicit():

    # Can handle zero series resistance.
    V_V = numpy.array(0.35)
    I_ph_A = 0.125
    I_rs_1_A = 9.24e-7
    n_1 = 1.5
    R_s_Ohm = 0.0
    G_p_S = 0.001
    N_s = 1
    T_degC = T_degC_stc
    P_W_expected = numpy.float64(
        V_V
        * (
            I_ph_A
            - I_rs_1_A * numpy.expm1(q_C * V_V / (N_s * n_1 * k_B_J_per_K * T_K_stc))
            - G_p_S * V_V
        )
    )

    P_W = equation.P_at_V(
        V_V=V_V,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )["P_W"]
    assert isinstance(P_W, type(P_W_expected))
    assert P_W.dtype == P_W_expected.dtype
    numpy.testing.assert_array_equal(P_W, P_W_expected)


def test_P_mp_no_convergence():

    I_ph_A = 0.125
    I_rs_1_A = 9.24e-7
    n_1 = 1.5
    R_s_Ohm = 0.5625
    G_p_S = 0.001
    N_s = 1
    T_degC = T_degC_stc
    minimize_scalar_bounded_options = {
        "maxiter": 1,
        "xatol": 1e-05,
    }  # Does not converge with only one iteration.

    with pytest.raises(ValueError) as excinfo:
        equation.P_mp(
            N_s=N_s,
            T_degC=T_degC,
            I_ph_A=I_ph_A,
            I_rs_1_A=I_rs_1_A,
            n_1=n_1,
            R_s_Ohm=R_s_Ohm,
            G_p_S=G_p_S,
            minimize_scalar_bounded_options=minimize_scalar_bounded_options,
        )["P_mp_W"]
    assert (
        "mimimize_scalar() with method='bounded' did not converge for options={'maxiter': 1, 'xatol': 1e-05}."
        == str(excinfo.value)
    )


@pytest.fixture(
    params=[
        {  # Happy path for a function that touches many others, scalar case.
            "I_ph_A": 0.125,
            "I_rs_1_A": 9.24e-7,
            "n_1": 1.5,
            "R_s_Ohm": 0.5625,
            "G_p_S": 0.001,
            "N_s": 1,
            "T_degC": T_degC_stc,
            "type_expected": numpy.float64,
            "dtype_expected": numpy.dtype("float64"),
        },
        {  # Happy path for a function that touches many others, rank-0 array case.
            "I_ph_A": 0.125,
            "I_rs_1_A": 1.23e-6,
            "n_1": 2.0,
            "R_s_Ohm": 0.5625,
            "G_p_S": numpy.array(0.005),
            "N_s": 1,
            "T_degC": T_degC_stc,
            "type_expected": numpy.float64,
            "dtype_expected": numpy.dtype("float64"),
        },
        {  # Happy path for a function that touches many others, array case.
            "I_ph_A": 0.125,
            "I_rs_1_A": numpy.array([9.24e-7, 1.23e-6, 7.54e-11]),
            "n_1": 1.75,
            "R_s_Ohm": 0.5625,
            "G_p_S": 0.001,
            "N_s": 1,
            "T_degC": T_degC_stc,
            "type_expected": numpy.ndarray,
            "dtype_expected": numpy.dtype("float64"),
        },
        {  # Happy path for a function that touches many others, zero power case.
            "I_ph_A": 0.0,
            "I_rs_1_A": 9.24e-7,
            "n_1": 1.75,
            "R_s_Ohm": 0.5625,
            "G_p_S": 0.001,
            "N_s": 1,
            "T_degC": T_degC_stc,
            "type_expected": numpy.float64,
            "dtype_expected": numpy.dtype("float64"),
        },
    ]
)
def iv_params_fixture(request):
    return request.param


def test_iv_params(iv_params_fixture):

    # Happy path for a function that touches many others.
    result = equation.iv_params(
        N_s=iv_params_fixture["N_s"],
        T_degC=iv_params_fixture["T_degC"],
        I_ph_A=iv_params_fixture["I_ph_A"],
        I_rs_1_A=iv_params_fixture["I_rs_1_A"],
        n_1=iv_params_fixture["n_1"],
        R_s_Ohm=iv_params_fixture["R_s_Ohm"],
        G_p_S=iv_params_fixture["G_p_S"],
    )

    for key, value in result.items():
        assert isinstance(value, iv_params_fixture["type_expected"])
        assert value.dtype == iv_params_fixture["dtype_expected"]
        if key == "FF":
            # Zero power cases.
            assert numpy.all(numpy.isnan(value[iv_params_fixture["I_ph_A"] == 0.0]))
            # Non-zero power cases.
            assert numpy.all(numpy.isfinite(value[iv_params_fixture["I_ph_A"] != 0.0]))
        else:
            assert numpy.all(numpy.isfinite(value))

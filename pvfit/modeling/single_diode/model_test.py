import numpy
import pytest

from pvfit.common.constants import T_stc_K, T_stc_degC, k_B_J_per_K, q_C, materials
import pvfit.modeling.single_diode.model as model


@pytest.fixture(
    # Not necessarily I-V curve solutions.
    params=[{
        'V_V': numpy.array([0.5, 0., 0.]),
        'I_A': numpy.array([0., 0., 3.]),
        'F': 1.,
        'T_degC': T_stc_degC,
        'N_s': 1,
        'T_0_degC': T_stc_degC,
        'I_sc_0_A': 7.,
        'I_rs_0_A': 6.e-7,
        'n_0': 1.25,
        'R_s_0_Ohm': 0.,  # Zero series resistance gives exact solution.
        'G_p_0_S': 0.005,
        'E_g_0_eV': materials['x-Si']['E_g_stc_eV'],
        'I_sum_A_expected': numpy.array([
            7. - 6.e-7 * numpy.expm1(q_C * 0.5 / (1 * 1.25 * k_B_J_per_K * T_stc_K)) - 0.005 * 0.5,
            7.,
            7. - 3.]),
        'T_K_expected': numpy.float64(T_stc_K),
        'V_diode_V_expected': numpy.array([0.5, 0., 0.]),
        'n_mod_V_expected': numpy.float64((1 * 1.25 * k_B_J_per_K * T_stc_K) / q_C)}])
def current_sum_at_diode_node_fixture(request):
    return request.param


def test_current_sum_at_diode_node(current_sum_at_diode_node_fixture):

    V_V = current_sum_at_diode_node_fixture['V_V']
    I_A = current_sum_at_diode_node_fixture['I_A']
    F = current_sum_at_diode_node_fixture['F']
    T_degC = current_sum_at_diode_node_fixture['T_degC']
    N_s = current_sum_at_diode_node_fixture['N_s']
    T_0_degC = current_sum_at_diode_node_fixture['T_0_degC']
    I_sc_0_A = current_sum_at_diode_node_fixture['I_sc_0_A']
    I_rs_0_A = current_sum_at_diode_node_fixture['I_rs_0_A']
    n_0 = current_sum_at_diode_node_fixture['n_0']
    R_s_0_Ohm = current_sum_at_diode_node_fixture['R_s_0_Ohm']
    G_p_0_S = current_sum_at_diode_node_fixture['G_p_0_S']
    E_g_0_eV = current_sum_at_diode_node_fixture['E_g_0_eV']
    I_sum_A_expected = current_sum_at_diode_node_fixture['I_sum_A_expected']
    T_K_expected = current_sum_at_diode_node_fixture['T_K_expected']
    V_diode_V_expected = current_sum_at_diode_node_fixture['V_diode_V_expected']
    n_mod_V_expected = current_sum_at_diode_node_fixture['n_mod_V_expected']

    result = model.current_sum_at_diode_node(
        V_V=V_V, I_A=I_A, F=F, T_degC=T_degC, N_s=N_s, T_0_degC=T_0_degC, I_sc_0_A=I_sc_0_A, I_rs_0_A=I_rs_0_A, n_0=n_0,
        R_s_0_Ohm=R_s_0_Ohm, G_p_0_S=G_p_0_S, E_g_0_eV=E_g_0_eV)

    assert isinstance(result['I_sum_A'], type(I_sum_A_expected))
    assert result['I_sum_A'].dtype == I_sum_A_expected.dtype
    numpy.testing.assert_array_almost_equal(result['I_sum_A'], I_sum_A_expected)

    assert isinstance(result['T_K'], type(T_K_expected))
    assert result['T_K'].dtype == T_K_expected.dtype
    numpy.testing.assert_array_almost_equal(result['T_K'], T_K_expected)

    assert isinstance(result['V_diode_V'], type(V_diode_V_expected))
    assert result['V_diode_V'].dtype == V_diode_V_expected.dtype
    numpy.testing.assert_array_almost_equal(result['V_diode_V'], V_diode_V_expected)

    assert isinstance(result['n_mod_V'], type(n_mod_V_expected))
    assert result['n_mod_V'].dtype == n_mod_V_expected.dtype
    numpy.testing.assert_array_almost_equal(result['n_mod_V'], n_mod_V_expected)


@pytest.fixture(
    # Not necessarily I-V curve solutions.
    params=[{  # Can handle all python scalar inputs.
        'F': 1.25,
        'T_degC': T_stc_degC,
        'N_s': 1,
        'T_0_degC': T_stc_degC,
        'I_sc_0_A': 7.,
        'I_rs_0_A': 6.e-7,
        'n_0': 1.25,
        'R_s_0_Ohm': 0.,  # Zero series resistance gives exact solution.
        'G_p_0_S': 0.005,
        'E_g_0_eV': materials['x-Si']['E_g_stc_eV'],
        'N_s_expected': numpy.intc(1),
        'T_degC_expected': numpy.float64(T_stc_degC),
        'I_ph_A_expected': numpy.float64(1.25 * 7.),
        'I_rs_A_expected': numpy.float64(6.e-7),
        'n_expected': numpy.float64(1.25),
        'R_s_Ohm_expected': numpy.float64(0.),
        'G_p_S_expected': numpy.float64(0.005)}])
def auxiliary_equations_fixture(request):
    return request.param


def test_auxiliary_equations(auxiliary_equations_fixture):
    F = auxiliary_equations_fixture['F']
    T_degC = auxiliary_equations_fixture['T_degC']
    N_s = auxiliary_equations_fixture['N_s']
    T_0_degC = auxiliary_equations_fixture['T_0_degC']
    I_sc_0_A = auxiliary_equations_fixture['I_sc_0_A']
    I_rs_0_A = auxiliary_equations_fixture['I_rs_0_A']
    n_0 = auxiliary_equations_fixture['n_0']
    R_s_0_Ohm = auxiliary_equations_fixture['R_s_0_Ohm']
    G_p_0_S = auxiliary_equations_fixture['G_p_0_S']
    E_g_0_eV = auxiliary_equations_fixture['E_g_0_eV']

    N_s_expected = auxiliary_equations_fixture['N_s_expected']
    T_degC_expected = auxiliary_equations_fixture['T_degC_expected']
    I_ph_A_expected = auxiliary_equations_fixture['I_ph_A_expected']
    I_rs_A_expected = auxiliary_equations_fixture['I_rs_A_expected']
    n_expected = auxiliary_equations_fixture['n_expected']
    R_s_Ohm_expected = auxiliary_equations_fixture['R_s_Ohm_expected']
    G_p_S_expected = auxiliary_equations_fixture['G_p_S_expected']

    result = model.auxiliary_equations(
        F=F, T_degC=T_degC, N_s=N_s, T_0_degC=T_0_degC, I_sc_0_A=I_sc_0_A, I_rs_0_A=I_rs_0_A, n_0=n_0,
        R_s_0_Ohm=R_s_0_Ohm, G_p_0_S=G_p_0_S, E_g_0_eV=E_g_0_eV)

    assert isinstance(result['N_s'], type(N_s_expected))
    assert result['N_s'].dtype == N_s_expected.dtype
    numpy.testing.assert_array_equal(result['N_s'], N_s_expected)
    assert isinstance(result['T_degC'], type(T_degC_expected))
    assert result['T_degC'].dtype == T_degC_expected.dtype
    numpy.testing.assert_array_equal(result['T_degC'], T_degC_expected)
    assert isinstance(result['I_ph_A'], type(I_ph_A_expected))
    assert result['I_ph_A'].dtype == I_ph_A_expected.dtype
    numpy.testing.assert_array_equal(result['I_ph_A'], I_ph_A_expected)
    assert isinstance(result['I_rs_A'], type(I_rs_A_expected))
    assert result['I_rs_A'].dtype == I_rs_A_expected.dtype
    numpy.testing.assert_array_equal(result['I_rs_A'], I_rs_A_expected)
    assert isinstance(result['n'], type(n_expected))
    assert result['n'].dtype == n_expected.dtype
    numpy.testing.assert_array_equal(result['n'], n_expected)
    assert isinstance(result['R_s_Ohm'], type(R_s_Ohm_expected))
    assert result['R_s_Ohm'].dtype == R_s_Ohm_expected.dtype
    numpy.testing.assert_array_equal(result['R_s_Ohm'], R_s_Ohm_expected)
    assert isinstance(result['G_p_S'], type(G_p_S_expected))
    assert result['G_p_S'].dtype == G_p_S_expected.dtype
    numpy.testing.assert_array_equal(result['G_p_S'], G_p_S_expected)


@pytest.mark.skip("TODO: Write I_at_V_F_T() tests.")
def test_I_at_V_F_T():
    pass


@pytest.mark.skip("TODO: Write V_at_I_F_T() tests.")
def test_V_at_I_F_T():
    pass


@pytest.mark.skip("TODO: Write derived_params() tests.")
def test_derived_params():
    pass

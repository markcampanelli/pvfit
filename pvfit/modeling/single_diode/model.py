import numpy as np
from scipy.constants import convert_temperature

from pvfit.common.constants import (k_B_J_per_K, k_B_eV_per_K, minimize_scalar_maxiter_default,
                                    minimize_scalar_xatol_default, newton_maxiter_default, newton_tol_default, q_C)
from pvfit.common.utils import ensure_numpy_scalars
import pvfit.modeling.single_diode.equation as equation


def current_sum_at_diode_node(
        *, V_V, I_A, F, T_degC, N_s, T_degC_0, I_sc_A_0, I_rs_A_0, n_0, R_s_Ohm_0, G_p_S_0, E_g_eV_0):
    """
    Computes the sum of the currents at the high-voltage diode node in the implicit 6-parameter global single-diode
    equivalent-circuit model (SDM-G).

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Observables at operating conditions (device-level):
            V_V terminal voltage
            I_A terminal current
            F effective irradiance ratio on diode junction
            T_degC effective diode-junction temperature
        Model parameters at reference conditions (device-level):
            N_s integer number of cells in series in each parallel string
            T_degC_0 temperature of diode-junction
            I_sc_A_0 short-circuit current
            I_rs_A_0 diode reverse-saturation current
            n_0 diode ideality factor
            R_s_Ohm_0 series resistance
            G_p_S_0 parallel conductance
            E_g_eV_0 material band gap

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing:
            I_sum_A sum of currents at high-voltage diode node
            T_K temperature of diode junction (in Kelvin)
            V_diode_V voltage at high-voltage diode node
            n_mod_V modified diode ideality factor
    """

    params = auxiliary_equations(F=F, T_degC=T_degC, N_s=N_s, T_degC_0=T_degC_0, I_sc_A_0=I_sc_A_0, I_rs_A_0=I_rs_A_0,
                                 n_0=n_0, R_s_Ohm_0=R_s_Ohm_0, G_p_S_0=G_p_S_0, E_g_eV_0=E_g_eV_0)

    return equation.current_sum_at_diode_node(V_V=V_V, I_A=I_A, **params)


def auxiliary_equations(*, F, T_degC, N_s, T_degC_0, I_sc_A_0, I_rs_A_0, n_0, R_s_Ohm_0, G_p_S_0, E_g_eV_0):
    """
    Computes the auxiliary equations at F and T_degC for the 6-parameter SDM-G.

    Inputs (any broadcast-compatible combination of scalars and numpy arrays):
        Same as current_sum_at_diode_node().

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing model parameters at operating conditions:
            I_ph_A photocurrent
            I_rs_A diode reverse-saturation current
            n diode ideality factor
            R_s_Ohm series resistance
            G_p_S parallel conductance
            N_s integer number of cells in series in each parallel string
            T_degC effective diode-junction temperature
    """

    # Temperatures must be in Kelvin.
    T_K = convert_temperature(T_degC, 'Celsius', 'Kelvin')
    T_K_0 = convert_temperature(T_degC_0, 'Celsius', 'Kelvin')

    # Compute variables at operating conditions.

    # Compute band gap (constant).
    E_g_eV = E_g_eV_0

    # Compute diode ideality factor (constant).
    n = n_0

    # Compute reverse-saturation current at T_degC (this is independent of F, I_sc_A_0, R_s_Ohm_0, and G_p_S_0).
    I_rs_A = I_rs_A_0 * (T_K / T_K_0)**3 * np.exp(E_g_eV / (n * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))

    # Compute series resistance (constant).
    R_s_Ohm = R_s_Ohm_0

    # Compute parallel conductance (constant).
    G_p_S = G_p_S_0

    # Compute parallel conductance (photo-conductive shunt).
    # G_p_S = F * G_p_S_0

    # Compute photo-generated current at F and T_degC (V=0 with I=Isc for this).
    expr1 = I_sc_A_0 * F
    expr2 = expr1 * R_s_Ohm
    I_ph_A = expr1 + I_rs_A * np.expm1(q_C * expr2 / (N_s * n * k_B_J_per_K * T_K)) + G_p_S * expr2

    return ensure_numpy_scalars(dictionary={
        'N_s': N_s, 'T_degC': T_degC, 'I_ph_A': I_ph_A, 'I_rs_A': I_rs_A, 'n': n, 'R_s_Ohm': R_s_Ohm, 'G_p_S': G_p_S})


def I_at_V_F_T(*, V_V, F, T_degC, N_s, T_degC_0, I_sc_A_0, I_rs_A_0, n_0, R_s_Ohm_0, G_p_S_0, E_g_eV_0,
               newton_tol=newton_tol_default, newton_maxiter=newton_maxiter_default):
    """
    Compute terminal current at given terminal voltage, effective irradiance ratio, and effective diode-junction
    temperature.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as current_sum_at_diode_node(), but with removal of I_A and addition of:
            newton_tol (optional) tolerance for Newton solver
            newton_maxiter (optional) maximum number of iterations for Newton solver

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of current_sum_at_diode_node() with the addition of:
            I_A terminal current
    """

    params = auxiliary_equations(F=F, T_degC=T_degC, N_s=N_s, T_degC_0=T_degC_0, I_sc_A_0=I_sc_A_0, I_rs_A_0=I_rs_A_0,
                                 n_0=n_0, R_s_Ohm_0=R_s_Ohm_0, G_p_S_0=G_p_S_0, E_g_eV_0=E_g_eV_0)

    return equation.I_at_V(V_V=V_V, **params, newton_tol=newton_tol, newton_maxiter=newton_maxiter)


def V_at_I_F_T(*, I_A, F, T_degC, N_s, T_degC_0, I_sc_A_0, I_rs_A_0, n_0, R_s_Ohm_0, G_p_S_0, E_g_eV_0,
               newton_tol=newton_tol_default, newton_maxiter=newton_maxiter_default):
    """
    Compute terminal voltage at given terminal current, effective irradiance ratio, and effective diode-junction
    temperature.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as current_sum_at_diode_node(), but with removal of V_V and addition of:
            newton_tol (optional) tolerance for Newton solver
            newton_maxiter (optional) maximum number of iterations for Newton solver

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of current_sum_at_diode_node() with the addition of:
            V_V terminal voltage
    """

    params = auxiliary_equations(F=F, T_degC=T_degC, N_s=N_s, T_degC_0=T_degC_0, I_sc_A_0=I_sc_A_0, I_rs_A_0=I_rs_A_0,
                                 n_0=n_0, R_s_Ohm_0=R_s_Ohm_0, G_p_S_0=G_p_S_0, E_g_eV_0=E_g_eV_0)

    return equation.V_at_I(I_A=I_A, **params, newton_tol=newton_tol, newton_maxiter=newton_maxiter)

# TODO Need to complete the utility functions.


def derived_params(*, F, T_degC, N_s, T_degC_0, I_sc_A_0, I_rs_A_0, n_0, R_s_Ohm_0, G_p_S_0, E_g_eV_0,
                   newton_tol=newton_tol_default, newton_maxiter=newton_maxiter_default,
                   minimize_scalar_xatol=minimize_scalar_xatol_default,
                   minimize_scalar_maxiter=minimize_scalar_maxiter_default):
    """
    Compute derived parameters as a function of irradiance and temperature.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as auxiliary_equations().

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of FF() with the addition of:
            R_oc_Ohm resistance at open circuit
            R_sc_Ohm resistance at short circuit
    """

    params = auxiliary_equations(F=F, T_degC=T_degC, N_s=N_s, T_degC_0=T_degC_0, I_sc_A_0=I_sc_A_0, I_rs_A_0=I_rs_A_0,
                                 n_0=n_0, R_s_Ohm_0=R_s_Ohm_0, G_p_S_0=G_p_S_0, E_g_eV_0=E_g_eV_0)

    result = equation.FF(**params, newton_tol=newton_tol, newton_maxiter=newton_maxiter,
                         minimize_scalar_xatol=minimize_scalar_xatol, minimize_scalar_maxiter=minimize_scalar_maxiter)

    # In SDM-G I_sc_A is a model parameter and I_ph_A is a derived parameter.
    result.pop('I_sc_A')
    result['I_ph_A'] = params['I_ph_A']

    # Compute the additional resistances.
    R_oc_Ohm = equation.R_at_oc(**params,  newton_tol=newton_tol, newton_maxiter=newton_maxiter)['R_oc_Ohm']
    R_sc_Ohm = equation.R_at_sc(**params, newton_tol=newton_tol, newton_maxiter=newton_maxiter)['R_sc_Ohm']
    result.update({'R_oc_Ohm': R_oc_Ohm, 'R_sc_Ohm': R_sc_Ohm})

    return result
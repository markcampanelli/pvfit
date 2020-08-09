import numpy
from scipy.constants import convert_temperature

from pvfit.common.constants import T_stc_degC, k_B_J_per_K, k_B_eV_per_K, q_C
import pvfit.modeling.double_diode.equation as dde


def current_sum_at_diode_node(*, V_V, I_A, F, T_degC, I_sc_0_A, I_rs_1_0_A, n_1_0, I_rs_2_0_A, n_2_0, R_s_0_Ohm,
                              G_p_0_S, E_g_0_eV, N_s, T_degC_0=T_stc_degC):
    """
    Computes the sum of the currents at the high-voltage diode node in the implicit 8-parameter global double-diode
    equivalent-circuit model (DDM-G).

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Observables at operating conditions (device-level):
            V_V terminal voltage
            I_A terminal current
            F effective irradiance ratio on diode junction
            T_degC effective diode-junction temperature
        Parameters at reference conditions (device-level):
            I_sc_0_A short-circuit current
            I_rs_A_1_0 first diode reverse-saturation current
            n_1_0 first diode ideality factor
            I_rs_A_2_0 second diode reverse-saturation current
            n_2_0 second diode ideality factor
            R_s_0_Ohm series resistance
            G_p_0_S parallel (shunt) conductance
            E_g_0_eV material band gap
            N_s integer number of cells in series in each parallel string
            T_degC_0 (optional) effective diode-junction temperature at reference conditions

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing:
            I_sum_A sum of currents at high-voltage diode node
            T_K temperature of diode junction (in Kelvin)
            V_diode_V voltage at high-voltage diode node
            n_mod_1_V first modified diode ideality factor
            n_mod_2_V second modified diode ideality factor
    """

    params = auxiliary_equations(
        F=F, T_degC=T_degC, I_sc_0_A=I_sc_0_A, I_rs_1_0_A=I_rs_1_0_A, n_1_0=n_1_0, I_rs_2_0_A=I_rs_2_0_A, n_2_0=n_2_0,
        R_s_0_Ohm=R_s_0_Ohm, G_p_0_S=G_p_0_S, E_g_0_eV=E_g_0_eV, N_s=N_s, T_degC_0=T_degC_0)

    return dde.current_sum_at_diode_node(V_V=V_V, I_A=I_A, **params)


def auxiliary_equations(*, F, T_degC, I_sc_0_A, I_rs_1_0_A, n_1_0, I_rs_2_0_A, n_2_0, R_s_0_Ohm, G_p_0_S, E_g_0_eV, N_s,
                        T_degC_0=T_stc_degC):
    """
    Computes the auxiliary equations at F and T_degC for the 8-parameter DDM-G.

    Inputs (any broadcast-compatible combination of scalars and numpy arrays):
        Same as current_sum_at_diode_node().

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing:
            I_ph_A photocurrent
            I_rs_1_A first diode reverse-saturation current
            n_1 first diode ideality factor
            I_rs_2_A second diode reverse-saturation current
            n_2 second diode ideality factor
            R_s_Ohm series resistance
            G_p_S parallel conductance
            N_s integer number of cells in series in each parallel string
            T_degC temperature of diode-junction at operating conditions
    """

    # Temperatures must be in Kelvin.
    T_K = convert_temperature(T_degC, 'Celsius', 'Kelvin')
    T_K_0 = convert_temperature(T_degC_0, 'Celsius', 'Kelvin')

    # Optimization.
    V_therm_factor_V_0 = (N_s * k_B_J_per_K * T_K_0) / q_C

    # Compute variables at operating conditions.

    # Compute band gap (constant).
    E_g_eV = E_g_0_eV

    # Compute first diode ideality factor (constant).
    n_1 = n_1_0

    # Compute first reverse-saturation current at T_degC (this is independent of F, I_sc_0_A, R_s_0_Ohm, and G_p_0_S).
    I_rs_1_A = I_rs_1_0_A * (T_K / T_K_0)**3 * numpy.exp(E_g_eV / (n_1 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))

    # Compute first diode ideality factor (constant).
    n_2 = n_2_0

    # Compute first reverse-saturation current at T_degC (this is independent of F, I_sc_0_A, R_s_0_Ohm, and G_p_0_S).
    I_rs_2_A = I_rs_2_0_A * (T_K / T_K_0)**(5/2) * numpy.exp(E_g_eV / (n_2 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))

    # Compute series resistance (constant).
    R_s_Ohm = R_s_0_Ohm

    # Compute parallel conductance (constant).
    G_p_S = G_p_0_S

    # Compute parallel conductance (photo-conductive shunt).
    # G_p_S = F * G_p_0_S

    # Compute photo-generated current at F and T_degC (V=0 with I=Isc for this).
    expr1 = I_sc_0_A * F
    expr2 = expr1 * R_s_Ohm
    I_ph_A = expr1 + I_rs_1_A * numpy.expm1(expr2 / (V_therm_factor_V_0 * n_1)) + \
        I_rs_2_A * numpy.expm1(expr2 / (V_therm_factor_V_0 * n_2)) + G_p_S * expr2

    return {'I_ph_A': I_ph_A, 'I_rs_1_A': I_rs_1_A, 'n_1': n_1, 'I_rs_2_A': I_rs_2_A, 'n_2': n_2, 'R_s_Ohm': R_s_Ohm,
            'G_p_S': G_p_S, 'N_s': N_s, 'T_degC': T_degC}

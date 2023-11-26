from typing import Optional

import numpy
from scipy.constants import convert_temperature

from pvfit.common.constants import T_degC_stc, k_B_J_per_K, k_B_eV_per_K, q_C
import pvfit.modeling.simulation.dc.double_diode.equation as equation


def current_sum_at_diode_node(
    *,
    V_V,
    I_A,
    F,
    T_degC,
    I_sc_A_0,
    I_rs_1_A_0,
    n_1_0,
    I_rs_2_A_0,
    n_2_0,
    R_s_Ohm_0,
    G_p_S_0,
    E_g_eV_0,
    N_s,
    T_degC_0=T_degC_stc,
):
    """
    Computes the sum of the currents at the diode's anode node in the implicit 8-parameter global double-diode
    equivalent-circuit model (DDM-G).

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Observables at operating condition(s) (device-level):
            V_V terminal voltage
            I_A terminal current
            F effective irradiance ratio on diode junction
            T_degC temperature
        Parameters at reference condition (device-level):
            I_sc_A_0 short-circuit current
            I_rs_A_1_0 first diode reverse-saturation current
            n_1_0 first diode ideality factor
            I_rs_A_2_0 second diode reverse-saturation current
            n_2_0 second diode ideality factor
            R_s_Ohm_0 series resistance
            G_p_S_0 parallel (shunt) conductance
            E_g_eV_0 material band gap
            N_s integer number of cells in series in each parallel string
            T_degC_0 (optional) temperature at reference condition

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing:
            I_sum_A sum of currents at diode's anode node
            T_K temperature of diode junction (in Kelvin)
            V_diode_V voltage at diode's anode node
            n_mod_1_V first modified diode ideality factor
            n_mod_2_V second modified diode ideality factor
    """

    params = auxiliary_equations(
        F=F,
        T_degC=T_degC,
        I_sc_A_0=I_sc_A_0,
        I_rs_1_A_0=I_rs_1_A_0,
        n_1_0=n_1_0,
        I_rs_2_A_0=I_rs_2_A_0,
        n_2_0=n_2_0,
        R_s_Ohm_0=R_s_Ohm_0,
        G_p_S_0=G_p_S_0,
        E_g_eV_0=E_g_eV_0,
        N_s=N_s,
        T_degC_0=T_degC_0,
    )

    return equation.current_sum_at_diode_node(V_V=V_V, I_A=I_A, **params)


def auxiliary_equations(
    *,
    F,
    T_degC,
    I_sc_A_0,
    I_rs_1_A_0,
    n_1_0,
    I_rs_2_A_0,
    n_2_0,
    R_s_Ohm_0,
    G_p_S_0,
    E_g_eV_0,
    N_s,
    T_degC_0=T_degC_stc,
):
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
            T_degC temperature
    """

    # Temperatures must be in Kelvin.
    T_K = convert_temperature(T_degC, "Celsius", "Kelvin")
    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")

    # Optimization.
    V_therm_factor_V_0 = (N_s * k_B_J_per_K * T_K_0) / q_C

    # Compute variables at operating condition.

    # Compute band gap (constant).
    E_g_eV = E_g_eV_0

    # Compute first diode ideality factor (constant).
    n_1 = n_1_0

    # Compute first reverse-saturation current at T_degC (this is independent of F, I_sc_A_0, R_s_Ohm_0, and G_p_S_0).
    I_rs_1_A = (
        I_rs_1_A_0
        * (T_K / T_K_0) ** 3
        * numpy.exp(E_g_eV / (n_1 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))
    )

    # Compute first diode ideality factor (constant).
    n_2 = n_2_0

    # Compute first reverse-saturation current at T_degC (this is independent of F, I_sc_A_0, R_s_Ohm_0, and G_p_S_0).
    I_rs_2_A = (
        I_rs_2_A_0
        * (T_K / T_K_0) ** (5 / 2)
        * numpy.exp(E_g_eV / (n_2 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))
    )

    # Compute series resistance (constant).
    R_s_Ohm = R_s_Ohm_0

    # Compute parallel conductance (constant).
    G_p_S = G_p_S_0

    # Compute parallel conductance (photo-conductive shunt).
    # G_p_S = F * G_p_S_0

    # Compute photo-generated current at F and T_degC (V=0 with I=Isc for this).
    expr1 = I_sc_A_0 * F
    expr2 = expr1 * R_s_Ohm
    I_ph_A = (
        expr1
        + I_rs_1_A * numpy.expm1(expr2 / (V_therm_factor_V_0 * n_1))
        + I_rs_2_A * numpy.expm1(expr2 / (V_therm_factor_V_0 * n_2))
        + G_p_S * expr2
    )

    return {
        "I_ph_A": I_ph_A,
        "I_rs_1_A": I_rs_1_A,
        "n_1": n_1,
        "I_rs_2_A": I_rs_2_A,
        "n_2": n_2,
        "R_s_Ohm": R_s_Ohm,
        "G_p_S": G_p_S,
        "N_s": N_s,
        "T_degC": T_degC,
    }


# TODO Need to complete the utility functions.


def iv_params(
    *,
    F,
    T_degC,
    N_s,
    T_degC_0,
    I_sc_A_0,
    I_rs_1_A_0,
    n_1_0,
    I_rs_2_A_0,
    n_2_0,
    R_s_Ohm_0,
    G_p_S_0,
    E_g_eV_0,
    newton_options: Optional[dict] = None,
    minimize_scalar_bounded_options: Optional[dict] = None,
):
    """
    Compute I-V curve parameters at specified effective irradiance ratio
    and device temperature.

    F
        Effective irradiance ratio on device [·].
    T_degC
        Temperature of device [°C].
    N_s
        Number of cells in series in each parallel string [·].
    T_degC_0
        Temperature at reference condtions [°C].
    I_sc_A_0
        Short-circuit current at reference condtions [A].
    I_rs_1_A_0
        Reverse-saturation current of first diode at reference condtions [A].
    n_1_0
        Ideality factor of first diode at reference condtions [·].
    I_rs_2_A_0
        Reverse-saturation current of second diode at reference condtions [A].
    n_2_0
        Ideality factor of second diode at reference condtions [·].
    R_s_Ohm_0
        Series resistance at reference condtions [Ω].
    G_p_S_0
        Parallel conductance at reference condtions [S].
    E_g_eV_0
        Material band gap at reference condtions [eV].
    newton_options
        Options for Newton solver (see scipy.optimize.newton).
    minimize_scalar_bounded_options
        Options for minimizer solver (see scipy.optimize.minimize_scalar).

    Returns
    -------
    result : dict
        FF
            Fill Factor [·].
        I_sc_A
            Short-circuit current [A].
        R_sc_Ohm
            Terminal resistance at short circuit [Ω].
        V_x_V
            Terminal voltage at half of terminal open-circuit voltage [V].
        I_x_A
            Terminal current at V_x_V [A].
        I_mp_A
            Terminal current at maximum terminal power [A].
        P_mp_W
            Maximum terminal power [W].
        V_mp_V
            Terminal voltage at maximum terminal power [V].
        V_xx_V
            Terminal voltage at average of votage at maximum power and
            terminal open-circuit voltage [V].
        I_xx_A
            Terminal current at V_xx_V [A].
        R_oc_Ohm
            Terminal resistance at open circuit [Ω].
        V_oc_V
            Terminal open-circuit voltage [V].

    Notes
    -----
    All parameters are at the device level, where the device consists of
    N_s PV cells in series in each of N_p strings in parallel. Inputs must
    be broadcast compatible. Output values are numpy.float64 or
    numpy.ndarray.
    """

    params = auxiliary_equations(
        F=F,
        T_degC=T_degC,
        N_s=N_s,
        T_degC_0=T_degC_0,
        I_sc_A_0=I_sc_A_0,
        I_rs_1_A_0=I_rs_1_A_0,
        n_1_0=n_1_0,
        I_rs_2_A_0=I_rs_2_A_0,
        n_2_0=n_2_0,
        R_s_Ohm_0=R_s_Ohm_0,
        G_p_S_0=G_p_S_0,
        E_g_eV_0=E_g_eV_0,
    )

    result = equation.iv_params(
        **params,
        minimize_scalar_bounded_options=minimize_scalar_bounded_options,
        newton_options=newton_options,
    )

    return result

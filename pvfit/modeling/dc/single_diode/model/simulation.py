"""PVfit: DC modeling for single-diode model simulation."""

from typing import Optional, Union

import numpy
from scipy.constants import convert_temperature

from pvfit.common.constants import k_B_J_per_K, k_B_eV_per_K, q_C
from pvfit.common.utils import ensure_numpy_scalars
import pvfit.modeling.dc.single_diode.equation.simulation as simulation


def current_sum_at_diode_node(
    *,
    V_V: Union[float, numpy.float64, numpy.ndarray],
    I_A: Union[float, numpy.float64, numpy.ndarray],
    F: Union[float, numpy.float64, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray],
    N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC_0: Union[float, numpy.float64, numpy.ndarray],
    I_sc_A_0: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A_0: Union[float, numpy.float64, numpy.ndarray],
    n_1_0: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm_0: Union[float, numpy.float64, numpy.ndarray],
    G_p_S_0: Union[float, numpy.float64, numpy.ndarray],
    E_g_eV_0: Union[float, numpy.float64, numpy.ndarray],
):
    """
    Computes the sum of currents at the diode's anode node in the
    6-parameter single-diode model (SDM) equivalent-circuit model.

    Parameters
    ----------
    V_V
        Terminal voltage [V].
    I_A
        Terminal current [A].
    F
        Effective irradiance ratio on device [·].
    T_degC
        device temperature [°C].
    N_s
        Number of cells in series in each parallel string [·].
    T_degC_0
        Temperature at reference condtions [°C].
    I_sc_A_0
        Short-circuit current at reference condtions [A].
    I_rs_1_A_0
        Reverse-saturation current of diode at reference condtions [A].
    n_1_0
        Ideality factor of diode at reference condtions [·].
    R_s_Ohm_0
        Series resistance at reference condtions [Ω].
    G_p_S_0
        Parallel conductance at reference condtions [S].
    E_g_eV_0
        Material band gap at reference condtions [eV].

    Returns
    -------
    result : dict
        I_sum_A
            Sum of currents at diode's anode node [A].
        T_K
            Temperature of device [K].
        V_1_V
            Voltage at diode's anode node [V].
        n_1_mod_V
            Modified ideality factor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    # Compute the auxiliary equations.
    params = auxiliary_equations(
        F=F,
        T_degC=T_degC,
        N_s=N_s,
        T_degC_0=T_degC_0,
        I_sc_A_0=I_sc_A_0,
        I_rs_1_A_0=I_rs_1_A_0,
        n_1_0=n_1_0,
        R_s_Ohm_0=R_s_Ohm_0,
        G_p_S_0=G_p_S_0,
        E_g_eV_0=E_g_eV_0,
    )

    return simulation.current_sum_at_diode_node(V_V=V_V, I_A=I_A, **params)


def auxiliary_equations(
    *,
    F: Union[float, numpy.float64, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray],
    N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC_0: Union[float, numpy.float64, numpy.ndarray],
    I_sc_A_0: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A_0: Union[float, numpy.float64, numpy.ndarray],
    n_1_0: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm_0: Union[float, numpy.float64, numpy.ndarray],
    G_p_S_0: Union[float, numpy.float64, numpy.ndarray],
    E_g_eV_0: Union[float, numpy.float64, numpy.ndarray],
):
    """
    Computes the auxiliary equations at effective irradiance ratio and
    device temperature.

    Parameters
    ----------
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
        Reverse-saturation current of diode at reference condtions [A].
    n_1_0
        Ideality factor of diode at reference condtions [·].
    R_s_Ohm_0
        Series resistance at reference condtions [Ω].
    G_p_S_0
        Parallel conductance at reference condtions [S].
    E_g_eV_0
        Material band gap at reference condtions [eV].

    Returns
    -------
    result : dict
        N_s
            Number of cells in series in each parallel string [·].
        T_degC
            Temperature of device [°C].
        I_ph_A
            Photocurrent [A].
        I_rs_1_A
            Reverse-saturation current of diode [A].
        n_1
            Ideality factor of diode [·].
        R_s_Ohm
            Series resistance [Ω].
        G_p_S
            Parallel conductance [S].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    # Temperatures must be in Kelvin.
    T_K = convert_temperature(T_degC, "Celsius", "Kelvin")
    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")

    # Compute variables at operating conditions.

    # Compute band gap (constant).
    E_g_eV = E_g_eV_0

    # Compute diode ideality factor (constant).
    n_1 = n_1_0

    # Compute reverse-saturation current at T_degC (this is independent of
    # F, I_sc_0_A, R_s_0_Ohm, and G_p_0_S).
    I_rs_1_A = (
        I_rs_1_A_0
        * (T_K / T_K_0) ** 3
        * numpy.exp(E_g_eV / (n_1_0 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))
    )

    # Compute series resistance (constant).
    R_s_Ohm = R_s_Ohm_0

    # Compute parallel conductance (constant).
    G_p_S = G_p_S_0

    # Compute parallel conductance (photo-conductive shunt).
    # G_p_S = F * G_p_S_0

    # Compute photo-generated current at F and T_degC (V=0 with I=Isc for this).
    I_sc_A = I_sc_A_0 * F
    V_diode_sc_V = I_sc_A * R_s_Ohm
    I_ph_A = (
        I_sc_A
        + I_rs_1_A * numpy.expm1(q_C * V_diode_sc_V / (N_s * n_1 * k_B_J_per_K * T_K))
        + G_p_S * V_diode_sc_V
    )

    return ensure_numpy_scalars(
        dictionary={
            "N_s": N_s,
            "T_degC": T_degC,
            "I_ph_A": I_ph_A,
            "I_rs_1_A": I_rs_1_A,
            "n_1": n_1,
            "R_s_Ohm": R_s_Ohm,
            "G_p_S": G_p_S,
        }
    )


def I_at_V_F_T(
    *,
    V_V: Union[float, numpy.float64, numpy.ndarray],
    F: Union[float, numpy.float64, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray],
    N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC_0: Union[float, numpy.float64, numpy.ndarray],
    I_sc_A_0: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A_0: Union[float, numpy.float64, numpy.ndarray],
    n_1_0: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm_0: Union[float, numpy.float64, numpy.ndarray],
    G_p_S_0: Union[float, numpy.float64, numpy.ndarray],
    E_g_eV_0: Union[float, numpy.float64, numpy.ndarray],
    newton_options: Optional[dict] = None,
) -> dict:
    """
    Compute terminal current at specified terminal voltage, effective
    irradiance ratio, and device temperature.

    Parameters
    ----------
    V_V
        Terminal voltage [V].
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
        Reverse-saturation current of diode at reference condtions [A].
    n_1_0
        Ideality factor of diode at reference condtions [·].
    R_s_Ohm_0
        Series resistance at reference condtions [Ω].
    G_p_S_0
        Parallel conductance at reference condtions [S].
    E_g_eV_0
        Material band gap at reference condtions [eV].
    newton_options
        Options for Newton solver (see scipy.optimize.newton).

    Returns
    -------
    result : dict
        I_A
            Terminal current [A].
        I_sum_A
            Sum of currents at diode's anode node [A].
        T_K
            Effective diode-junction temperature [K].
        V_1_V
            Voltage at diode's anode node [V].
        n_1_mod_V
            Modified ideality factor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    params = auxiliary_equations(
        F=F,
        T_degC=T_degC,
        N_s=N_s,
        T_degC_0=T_degC_0,
        I_sc_A_0=I_sc_A_0,
        I_rs_1_A_0=I_rs_1_A_0,
        n_1_0=n_1_0,
        R_s_Ohm_0=R_s_Ohm_0,
        G_p_S_0=G_p_S_0,
        E_g_eV_0=E_g_eV_0,
    )

    return simulation.I_at_V(V_V=V_V, **params, newton_options=newton_options)


def V_at_I_F_T(
    *,
    I_A: Union[float, numpy.float64, numpy.ndarray],
    F: Union[float, numpy.float64, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray],
    N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC_0: Union[float, numpy.float64, numpy.ndarray],
    I_sc_A_0: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A_0: Union[float, numpy.float64, numpy.ndarray],
    n_1_0: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm_0: Union[float, numpy.float64, numpy.ndarray],
    G_p_S_0: Union[float, numpy.float64, numpy.ndarray],
    E_g_eV_0: Union[float, numpy.float64, numpy.ndarray],
    newton_options: Optional[dict] = None,
) -> dict:
    """
    Compute terminal voltage at specified terminal current, effective
    irradiance ratio, and device temperature.

    Parameters
    ----------
    I_A
        Terminal current [A].
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
        Reverse-saturation current of diode at reference condtions [A].
    n_1_0
        Ideality factor of diode at reference condtions [·].
    R_s_Ohm_0
        Series resistance at reference condtions [Ω].
    G_p_S_0
        Parallel conductance at reference condtions [S].
    E_g_eV_0
        Material band gap at reference condtions [eV].
    newton_options
        Options for Newton solver (see scipy.optimize.newton).

    Returns
    -------
    result : dict
        V_V
            Terminal voltage [V].
        I_sum_A
            Sum of currents at diode's anode node [A].
        T_K
            Effective diode-junction temperature [K].
        V_1_V
            Voltage at diode's anode node [V].
        n_1_mod_V
            Modified ideality factor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    params = auxiliary_equations(
        F=F,
        T_degC=T_degC,
        N_s=N_s,
        T_degC_0=T_degC_0,
        I_sc_A_0=I_sc_A_0,
        I_rs_1_A_0=I_rs_1_A_0,
        n_1_0=n_1_0,
        R_s_Ohm_0=R_s_Ohm_0,
        G_p_S_0=G_p_S_0,
        E_g_eV_0=E_g_eV_0,
    )

    return simulation.V_at_I(I_A=I_A, **params, newton_options=newton_options)


# TODO Need to complete the utility functions.


def iv_params(
    *,
    F: Union[float, numpy.float64, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray],
    N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC_0: Union[float, numpy.float64, numpy.ndarray],
    I_sc_A_0: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A_0: Union[float, numpy.float64, numpy.ndarray],
    n_1_0: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm_0: Union[float, numpy.float64, numpy.ndarray],
    G_p_S_0: Union[float, numpy.float64, numpy.ndarray],
    E_g_eV_0: Union[float, numpy.float64, numpy.ndarray],
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
        Reverse-saturation current of diode at reference condtions [A].
    n_1_0
        Ideality factor of diode at reference condtions [·].
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
        R_s_Ohm_0=R_s_Ohm_0,
        G_p_S_0=G_p_S_0,
        E_g_eV_0=E_g_eV_0,
    )

    result = simulation.iv_params(
        **params,
        minimize_scalar_bounded_options=minimize_scalar_bounded_options,
        newton_options=newton_options,
    )

    return result

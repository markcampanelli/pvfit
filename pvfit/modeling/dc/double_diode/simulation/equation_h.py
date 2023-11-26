"""
WARNING: This code is experimental and not subject to semantic versioning.

This is to be incorporated into equation.py. Set R_h_Ohm = 0 to "fall back" to standard DDE.
"""

from typing import Optional, Union

import numpy
from scipy.constants import convert_temperature
from scipy.optimize import newton

from pvfit.common.constants import I_sum_A_atol, k_B_J_per_K, q_C
from pvfit.common.utils import ensure_numpy_scalars
import pvfit.modeling.dc.single_diode.simulation.equation


def current_sum_at_diode_node(
    *,
    V_V: Union[float, numpy.float64, numpy.ndarray],
    I_A: Union[float, numpy.float64, numpy.ndarray],
    N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray],
    I_ph_A: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A: Union[float, numpy.float64, numpy.ndarray],
    n_1: Union[float, numpy.float64, numpy.ndarray],
    I_rs_h_A: Union[float, numpy.float64, numpy.ndarray],
    n_h: Union[float, numpy.float64, numpy.ndarray],
    R_h_Ohm: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm: Union[float, numpy.float64, numpy.ndarray],
    G_p_S: Union[float, numpy.float64, numpy.ndarray],
) -> dict:
    """
    Computes the sum of the currents at the high-voltage node of the main
    diode in the 8-parameter double-diode equation (DDE)
    equivalent-circuit model.

    Parameters
    ----------
    V_V
        Terminal voltage [V].
    I_A
        Terminal current [A].
    N_s
        Number of cells in series in each parallel string [·].
    T_degC
        Temperature of device [°C].
    I_ph_A
        Photocurrent [A].
    I_rs_1_A
        Reverse-saturation current of main diode [A].
    n_1
        Ideality factor of main diode [·].
    I_rs_h_A
        Reverse-saturation current of diode in series with resistor [A].
    n_h
        Ideality factor of diode of diode in series with resistor [·].
    R_h_Ohm
        Series resistance of resistor in series with diode [Ω].
    R_s_Ohm
        Series resistance [Ω].
    G_p_S
        Parallel conductance [S].

    Returns
    -------
    result : dict
        I_sum_A
            Sum of currents at high-voltage node of main diode [A].
        T_K
            Temperature of device [K].
        V_1_V
            Voltage at high-voltage node of main diode [V].
        n_1_mod_V
            Modified ideality factor for main diode [V].
        V_h_V
            Voltage at high-voltage node of diode in series with resistor [V].
        n_h_mod_V
            Modified ideality factor for diode in series with resistor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists of
    N_s PV cells in series in each of N_p strings in parallel. Inputs must
    be broadcast compatible. Output values are numpy.float64 or
    numpy.ndarray.
    """

    result = (
        pvfit.modeling.dc.single_diode.simulation.equation.current_sum_at_diode_node(
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
    )

    # Modified ideality factor of diode in series with resistor.
    n_h_mod_V = (N_s * n_h * k_B_J_per_K * result["T_K"]) / q_C

    # Votage drop across diode in series with resistor.
    V_h_V = result["V_1_V"] - R_h_Ohm * result["I_sum_A"]

    # Sum of currents at high-voltage node of main diode.
    # numpy.expm1() returns a numpy.float64 when arguments are all python/numpy scalars.
    I_sum_A = result["I_sum_A"] - I_rs_h_A * numpy.expm1(V_h_V / n_h_mod_V)

    return ensure_numpy_scalars(
        dictionary={
            "I_sum_A": I_sum_A,
            "T_K": result["T_K"],
            "V_1_V": result["V_1_V"],
            "n_1_mod_V": result["n_1_mod_V"],
            "V_h_V": V_h_V,
            "n_h_mod_V": n_h_mod_V,
        }
    )


def I_at_V(
    *,
    V_V: Union[float, numpy.float64, numpy.ndarray],
    N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray],
    I_ph_A: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A: Union[float, numpy.float64, numpy.ndarray],
    n_1: Union[float, numpy.float64, numpy.ndarray],
    I_rs_h_A: Union[float, numpy.float64, numpy.ndarray],
    n_h: Union[float, numpy.float64, numpy.ndarray],
    R_h_Ohm: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm: Union[float, numpy.float64, numpy.ndarray],
    G_p_S: Union[float, numpy.float64, numpy.ndarray],
    newton_options: Optional[dict] = None,
) -> dict:
    """
    Compute terminal current from terminal voltage using Newton's method.

    Parameters
    ----------
    V_V
        Terminal voltage [V].
    N_s
        Number of cells in series in each parallel string [·].
    T_degC
        Temperature of device [°C].
    I_ph_A
        Photocurrent [A].
    I_rs_1_A
        Reverse-saturation current of main diode [A].
    n_1
        Ideality factor of main diode [·].
    I_rs_h_A
        Reverse-saturation current of diode in series with resistor [A].
    n_h
        Ideality factor of diode of diode in series with resistor [·].
    R_h_Ohm
        Series resistance of resistor in series with diode [Ω].
    R_s_Ohm
        Series resistance [Ω].
    G_p_S
        Parallel conductance [S].
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
            Temperature of device [K].
        V_1_V
            Voltage at high-voltage side of main diode node [V].
        n_1_mod_V
            Modified ideality factor for main diode [V].
        V_h_V
            Voltage at high-voltage side of diode in series with resistor [V].
        n_h_mod_V
            Modified ideality factor for diode in series with resistor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists of
    N_s PV cells in series in each of N_p strings in parallel. Inputs must
    be broadcast compatible. Output values are numpy.float64 or
    numpy.ndarray.

    Compute strategy:

    1) Compute initial condition for I_A with explicit equation using
    R_s_Ohm==0.
    2) Compute using scipy.optimize.newton.
    """
    # Optimizations.
    common_factor = (
        N_s * k_B_J_per_K * convert_temperature(T_degC, "Celsius", "Kelvin")
    ) / q_C
    n_1_mod_V = n_1 * common_factor
    n_h_mod_V = n_h * common_factor

    # Initial condition: Compute with zero R_s_Ohm and infinite R_h_Ohm and include all terms to preserve shapes.
    V_1_V_ic = V_V + 0.0 * R_s_Ohm
    I_A_ic = (
        I_ph_A
        - I_rs_1_A * numpy.expm1(V_1_V_ic / n_1_mod_V)
        - (I_rs_h_A * 0.0 * R_h_Ohm * n_h)
        - G_p_S * V_1_V_ic
    )

    # Use closures in function definitions for Newton's method.
    def func(I_A):
        V_1_V = V_V + I_A * R_s_Ohm
        I_sum_A_partial = (
            I_ph_A - I_rs_1_A * numpy.expm1(V_1_V / n_1_mod_V) - G_p_S * V_1_V - I_A
        )
        return I_sum_A_partial - I_rs_h_A * numpy.expm1(
            (V_1_V - R_h_Ohm * I_sum_A_partial) / n_h_mod_V
        )

    def fprime(I_A):
        V_1_V = V_V + I_A * R_s_Ohm
        common_term = (
            -I_rs_1_A
            * R_s_Ohm
            / n_1_mod_V
            * numpy.exp((V_V + I_A * R_s_Ohm) / n_1_mod_V)
            - G_p_S * R_s_Ohm
            - 1.0
        )
        return common_term - (
            I_rs_h_A / n_h_mod_V * (R_s_Ohm - R_h_Ohm * common_term)
        ) * numpy.exp(
            (
                V_1_V
                - R_h_Ohm
                * (
                    I_ph_A
                    - I_rs_1_A * numpy.expm1(V_1_V / n_1_mod_V)
                    - G_p_S * V_1_V
                    - I_A
                )
            )
            / n_h_mod_V
        )

    # FUTURE Consider using this in Halley's method.
    # def fprime2(I_A):
    #     return TODO

    # Solve for I_A using Newton's method.
    I_A = newton(func, I_A_ic, fprime=fprime, **newton_options)

    # Verify convergence, because newton() documentation says that this should be checked.
    result = current_sum_at_diode_node(
        V_V=V_V,
        I_A=I_A,
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_1_A=I_rs_1_A,
        n_1=n_1,
        I_rs_h_A=I_rs_h_A,
        n_h=n_h,
        R_h_Ohm=R_h_Ohm,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )
    numpy.testing.assert_allclose(result["I_sum_A"], 0.0, atol=I_sum_A_atol)

    # Add verified currents to result.
    result.update(ensure_numpy_scalars(dictionary={"I_A": I_A}))

    return result

from typing import Union

import numpy
from scipy.constants import convert_temperature
from scipy.optimize import minimize_scalar, newton

from pvfit.common.constants import (
    I_sum_A_atol, k_B_J_per_K, minimize_scalar_bounded_options_default, newton_options_default, q_C)
from pvfit.common.utils import ensure_numpy_scalars


def current_sum_at_diode_node(
    *, V_V: Union[float, numpy.float64, numpy.ndarray], I_A: Union[float, numpy.float64, numpy.ndarray],
    N_s: Union[int, numpy.intc, numpy.ndarray], T_degC: Union[float, numpy.float64, numpy.ndarray],
    I_ph_A: Union[float, numpy.float64, numpy.ndarray], I_rs_1_A: Union[float, numpy.float64, numpy.ndarray],
    n_1: Union[float, numpy.float64, numpy.ndarray], R_s_Ohm: Union[float, numpy.float64, numpy.ndarray],
        G_p_S: Union[float, numpy.float64, numpy.ndarray]) -> dict:
    """
    Computes the sum of the currents at the diode's anode in the
    5-parameter single-diode equation (SDE) equivalent-circuit model.

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
        Reverse-saturation current of diode [A].
    n_1
        Ideality factor of diode [·].
    R_s_Ohm
        Series resistance [Ω].
    G_p_S
        Parallel conductance [S].

    Returns
    -------
    result : dict
        I_sum_A
            Sum of currents at high-voltage diode node [A].
        T_K
            Effective diode-junction temperature [K].
        V_1_V
            Voltage at high-voltage diode node [V].
        n_1_mod_V
            Modified ideality factor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    # Temperature in Kelvin.
    T_K = convert_temperature(T_degC, 'Celsius', 'Kelvin')

    # Voltage at diode node.
    V_1_V = V_V + I_A * R_s_Ohm

    # Modified idealityV_1_V factor.
    n_1_mod_V = (N_s * n_1 * k_B_J_per_K * T_K) / q_C

    # Sum of currents at diode node. numpy.expm1() returns a numpy.float64 when arguments are all python/numpy scalars.
    I_sum_A = I_ph_A - I_rs_1_A * numpy.expm1(V_1_V / n_1_mod_V) - G_p_S * V_1_V - I_A

    return ensure_numpy_scalars(dictionary={'I_sum_A': I_sum_A, 'T_K': T_K, 'V_1_V': V_1_V, 'n_1_mod_V': n_1_mod_V})


def I_at_V(
    *, V_V: Union[float, numpy.float64, numpy.ndarray], N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray], I_ph_A: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A: Union[float, numpy.float64, numpy.ndarray], n_1: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm: Union[float, numpy.float64, numpy.ndarray], G_p_S: Union[float, numpy.float64, numpy.ndarray],
        newton_options: dict = newton_options_default) -> dict:
    """
    Compute terminal current at specified terminal voltage.

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
        Reverse-saturation current of diode [A].
    n_1
        Ideality factor of diode [·].
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

    Compute strategy:

    1) Compute initial condition for I_A with explicit equation using
    R_s_Ohm==0.
    2) Compute using scipy.optimize.newton.
    """
    # Optimization.
    n_1_mod_V = (N_s * n_1 * k_B_J_per_K * convert_temperature(T_degC, 'Celsius', 'Kelvin')) / q_C

    # Preserve shape of excluded R_s_Ohm in inital condition.
    V_1_V_ic = V_V + 0. * R_s_Ohm

    # Compute initially with zero R_s_Ohm.
    I_A_ic = I_ph_A - I_rs_1_A * numpy.expm1(V_1_V_ic / n_1_mod_V) - G_p_S * V_1_V_ic

    # Use closures in function definitions for Newton's method.
    def func(I_A):
        V_1_V = V_V + I_A * R_s_Ohm
        return I_ph_A - I_rs_1_A * numpy.expm1(V_1_V / n_1_mod_V) - G_p_S * V_1_V - I_A

    def fprime(I_A):
        return -I_rs_1_A * R_s_Ohm / n_1_mod_V * numpy.exp((V_V + I_A * R_s_Ohm) / n_1_mod_V) - G_p_S * R_s_Ohm - 1.

    # FUTURE Consider using this in Halley's method.
    # def fprime2(I_A):
    #     return -I_rs_1_A * (R_s_Ohm / n_1_mod_V)**2 * numpy.exp((V_V + I_A * R_s_Ohm) / n_1_mod_V)

    # Solve for I_A using Newton's method.
    I_A = newton(func, I_A_ic, fprime=fprime, **newton_options)

    # Verify convergence, because newton() documentation says that this should be checked.
    result = current_sum_at_diode_node(V_V=V_V, I_A=I_A, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A,
                                       n_1=n_1, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)
    numpy.testing.assert_allclose(result['I_sum_A'], 0., atol=I_sum_A_atol)

    # Add verified currents to result.
    result.update(ensure_numpy_scalars(dictionary={'I_A': I_A}))

    return result


def I_at_V_d1(
    *, V_V: Union[float, numpy.float64, numpy.ndarray], N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray], I_ph_A: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A: Union[float, numpy.float64, numpy.ndarray], n_1: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm: Union[float, numpy.float64, numpy.ndarray], G_p_S: Union[float, numpy.float64, numpy.ndarray],
        newton_options: dict = newton_options_default) -> dict:
    """
    Compute 1st derivative of terminal current with respect to terminal
    voltage at specified terminal voltage.

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
        Reverse-saturation current of diode [A].
    n_1
        Ideality factor of diode [·].
    R_s_Ohm
        Series resistance [Ω].
    G_p_S
        Parallel conductance [S].

    Returns
    -------
    result : dict
        I_d1_V_S
            1st derivative of terminal current w.r.t terminal voltage [S].
        I_A
            Terminal current [A].
        I_sum_A
            Sum of currents at high-voltage diode node [A].
        T_K
            Effective diode-junction temperature [K].
        V_1_V
            Voltage at high-voltage diode node [V].
        n_1_mod_V
            Modified ideality factor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.

    This derivative is needed, e.g., for R_oc_Ohm and R_sc_Ohm
    calculations.
    """
    # Compute terminal current.
    result = I_at_V(V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                    G_p_S=G_p_S, newton_options=newton_options)

    # Compute first derivative of current with respect to voltage at specified voltage.
    expr1 = I_rs_1_A / result['n_1_mod_V'] * numpy.exp(result['V_1_V'] / result['n_1_mod_V']) + G_p_S
    I_d1_V_S = -expr1 / (1. + R_s_Ohm * expr1)
    result.update(ensure_numpy_scalars(dictionary={'I_d1_V_S': I_d1_V_S}))

    return result


def V_at_I(
    *, I_A: Union[float, numpy.float64, numpy.ndarray], N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray], I_ph_A: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A: Union[float, numpy.float64, numpy.ndarray], n_1: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm: Union[float, numpy.float64, numpy.ndarray], G_p_S: Union[float, numpy.float64, numpy.ndarray],
        newton_options: dict = newton_options_default) -> dict:
    """
    Compute terminal voltage at specified terminal current.

    Parameters
    ----------
    I_A
        Terminal current [A].
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
    newton_options
        Options for Newton solver (see scipy.optimize.newton).

    Returns
    -------
    result : dict
        V_V
            Terminal voltage [V].
        I_sum_A
            Sum of currents at high-voltage diode node [A].
        T_K
            Effective diode-junction temperature [K].
        V_1_V
            Voltage at high-voltage diode node [V].
        n_1_mod_V
            Modified ideality factor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.

    Compute strategy:

    1) Compute initial condition for V_V with explicit equation
    using G_p_S==0.
    2) Compute using scipy.optimize.newton.
    """
    # Optimization.
    n_1_mod_V = (N_s * n_1 * k_B_J_per_K * convert_temperature(T_degC, 'Celsius', 'Kelvin')) / q_C

    # Compute initially with zero G_p_S, preserving shape of excluded G_p_S in inital condition.
    V_V_ic = n_1_mod_V * (numpy.log(I_ph_A + I_rs_1_A - 0. * G_p_S - I_A) - numpy.log(I_rs_1_A)) - I_A * R_s_Ohm

    # Use closures in these function definitions for Newton's method.
    def func(V_V):
        V_1_V = V_V + I_A * R_s_Ohm
        return I_ph_A - I_rs_1_A * numpy.expm1(V_1_V / n_1_mod_V) - G_p_S * V_1_V - I_A

    def fprime(V_V):
        return -I_rs_1_A / n_1_mod_V * numpy.exp((V_V + I_A * R_s_Ohm) / n_1_mod_V) - G_p_S

    # FUTURE Consider using this in Halley's method.
    # def fprime2(V_V):
    #     return -I_rs_1_A / n_1_mod_V**2. * numpy.exp((V_V + I_A * R_s_Ohm) / n_1_mod_V)

    # Solve for V_V using Newton's method.
    V_V = newton(func, V_V_ic, fprime=fprime, **newton_options)

    # Verify convergence. newton() documentation says that this should be checked.
    result = current_sum_at_diode_node(V_V=V_V, I_A=I_A, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A,
                                       n_1=n_1, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)
    numpy.testing.assert_allclose(result['I_sum_A'], 0., atol=I_sum_A_atol)

    # Add verified voltages to result.
    result.update(ensure_numpy_scalars(dictionary={'V_V': V_V}))

    return result


def V_at_I_d1(
    *, I_A: Union[float, numpy.float64, numpy.ndarray], N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray], I_ph_A: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A: Union[float, numpy.float64, numpy.ndarray], n_1: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm: Union[float, numpy.float64, numpy.ndarray], G_p_S: Union[float, numpy.float64, numpy.ndarray],
        newton_options: dict = newton_options_default) -> dict:
    """
    Compute 1st derivative of terminal voltage with respect to terminal
    current at specified terminal current.

    Parameters
    ----------
    I_A
        Terminal current [A].
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

    Returns
    -------
    result : dict
        V_d1_I_Ohm
            1st derivative of terminal voltage w.r.t terminal current [Ω].
        V_V
            Terminal voltage [V].
        I_sum_A
            Sum of currents at high-voltage diode node [A].
        T_K
            Effective diode-junction temperature [K].
        V_1_V
            Voltage at high-voltage diode node [V].
        n_1_mod_V
            Modified ideality factor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.

    This derivative is needed, e.g., for solving the differential
    equation for capacitor charging.
    """
    # Compute terminal voltage.
    result = V_at_I(I_A=I_A, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                    G_p_S=G_p_S, newton_options=newton_options)

    # Compute first derivative of voltage with respect to current at specified current.
    expr1 = I_rs_1_A / result['n_1_mod_V'] * numpy.exp(result['V_1_V'] / result['n_1_mod_V']) + G_p_S
    V_d1_I_Ohm = -1. / expr1 - R_s_Ohm
    result.update(ensure_numpy_scalars(dictionary={'V_d1_I_Ohm': V_d1_I_Ohm}))

    return result


def P_at_V(
    *, V_V: Union[float, numpy.float64, numpy.ndarray], N_s: Union[int, numpy.intc, numpy.ndarray],
    T_degC: Union[float, numpy.float64, numpy.ndarray], I_ph_A: Union[float, numpy.float64, numpy.ndarray],
    I_rs_1_A: Union[float, numpy.float64, numpy.ndarray], n_1: Union[float, numpy.float64, numpy.ndarray],
    R_s_Ohm: Union[float, numpy.float64, numpy.ndarray], G_p_S: Union[float, numpy.float64, numpy.ndarray],
        newton_options: dict = newton_options_default) -> dict:
    """
    Compute terminal power at specified terminal voltage.

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
        Reverse-saturation current of diode [A].
    n_1
        Ideality factor of diode [·].
    R_s_Ohm
        Series resistance [Ω].
    G_p_S
        Parallel conductance [S].
    newton_options
        Options for Newton solver (see scipy.optimize.newton).

    Returns
    -------
    result : dict
        P_W
            Terminal power [W].
        I_A
            Terminal current [A].
        I_sum_A
            Sum of currents at high-voltage diode node [A].
        T_K
            Effective diode-junction temperature [K].
        V_1_V
            Voltage at high-voltage diode node [V].
        n_1_mod_V
            Modified ideality factor [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    # Compute power.
    result = I_at_V(V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                    G_p_S=G_p_S, newton_options=newton_options)
    P_W = V_V * result['I_A']
    result.update(ensure_numpy_scalars(dictionary={'P_W': P_W}))

    return result


def P_mp(
    *, N_s: Union[int, numpy.intc, numpy.ndarray], T_degC: Union[float, numpy.float64, numpy.ndarray],
    I_ph_A: Union[float, numpy.float64, numpy.ndarray], I_rs_1_A: Union[float, numpy.float64, numpy.ndarray],
    n_1: Union[float, numpy.float64, numpy.ndarray], R_s_Ohm: Union[float, numpy.float64, numpy.ndarray],
    G_p_S: Union[float, numpy.float64, numpy.ndarray], newton_options: dict = newton_options_default,
        minimize_scalar_bounded_options: dict = minimize_scalar_bounded_options_default) -> dict:
    """
    Compute maximum terminal power.

    Parameters
    ----------
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
    newton_options
        Options for Newton solver (see scipy.optimize.newton).
    minimize_scalar_bounded_options
        Options for minimization solver (see scipy.optimize.minimize_scalar).

    Returns
    -------
    result : dict
        I_mp_A
            Terminal current at maximum terminal power [A].
        P_mp_W
            Maximum terminal power [W].
        V_mp_V
            Terminal voltage at maximum terminal power [V].
        V_oc_V
            Terminal open-circuit voltage [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.

    Compute strategy:

    1) Compute solution bracketing interval as [0, Voc].
    2) Compute maximum power in solution bracketing interval using scipy.optimize.minimize_scalar.
    """
    # Compute Voc for assumed Vmp bracket [0, Voc].
    V_oc_V = V_at_I(I_A=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,\
                    G_p_S=G_p_S, newton_options=newton_options)['V_V']

    # This allows us to make a ufunc out of minimize_scalar(). Note closures over solver arguments/options.
    def opposite_P_at_V(V_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, R_s_Ohm, G_p_S):
        return -P_at_V(V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                       G_p_S=G_p_S, newton_options=newton_options)['P_W']

    array_min_func = numpy.frompyfunc(
        lambda V_oc_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, R_s_Ohm, G_p_S: minimize_scalar(
            opposite_P_at_V, bounds=(0., V_oc_V), args=(N_s, T_degC, I_ph_A, I_rs_1_A, n_1, R_s_Ohm, G_p_S),
            method='bounded', options=minimize_scalar_bounded_options), 8, 1)

    # Solve for the array of OptimizeResult objects.
    res_array = array_min_func(V_oc_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, R_s_Ohm, G_p_S)

    # Verify convergence. Note that numpy.frompyfunc() always returns a PyObject array, which must be cast.
    if not numpy.all(numpy.array(numpy.frompyfunc(lambda res: res.success, 1, 1)(res_array), dtype=bool)):
        raise ValueError(
            f"mimimize_scalar() with method='bounded' did not converge for options={minimize_scalar_bounded_options}.")

    # Collect results. Casting with numpy.float64() creates numpy.ndarray if needed.
    V_mp_V = numpy.float64(numpy.frompyfunc(lambda res: res.x, 1, 1)(res_array))
    P_mp_W = numpy.float64(numpy.frompyfunc(lambda res: -res.fun, 1, 1)(res_array))
    with numpy.errstate(divide='ignore', invalid='ignore'):
        I_mp_A = numpy.float64(numpy.where(V_mp_V != 0., P_mp_W / V_mp_V, 0.))

    return ensure_numpy_scalars(dictionary={'P_mp_W': P_mp_W, 'I_mp_A': I_mp_A, 'V_mp_V': V_mp_V, 'V_oc_V': V_oc_V})


def FF(
    *, N_s: Union[int, numpy.intc, numpy.ndarray], T_degC: Union[float, numpy.float64, numpy.ndarray],
    I_ph_A: Union[float, numpy.float64, numpy.ndarray], I_rs_1_A: Union[float, numpy.float64, numpy.ndarray],
    n_1: Union[float, numpy.float64, numpy.ndarray], R_s_Ohm: Union[float, numpy.float64, numpy.ndarray],
    G_p_S: Union[float, numpy.float64, numpy.ndarray], newton_options: dict = newton_options_default,
        minimize_scalar_bounded_options: dict = minimize_scalar_bounded_options_default) -> dict:
    """
    Compute fill factor (unitless fraction).

    Parameters
    ----------
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
    newton_options
        Options for Newton solver (see scipy.optimize.newton).
    minimize_scalar_bounded_options
        Options for minimization solver (see
        scipy.optimize.minimize_scalar).

    Returns
    -------
    result : dict
        FF
            Fill Factor [·].
        I_sc_A
            Terminal short-circuit current [A].
        I_mp_A
            Terminal current at maximum terminal power [A].
        P_mp_W
            Maximum terminal power [W].
        V_mp_V
            Terminal voltage at maximum terminal power [V].
        V_oc_V
            Terminal open-circuit voltage [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    # Compute Pmp.
    result = P_mp(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                  minimize_scalar_bounded_options=minimize_scalar_bounded_options, newton_options=newton_options)
    # Compute Isc.
    I_sc_A = I_at_V(V_V=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                    G_p_S=G_p_S, newton_options=newton_options)['I_A']
    result.update({'I_sc_A': I_sc_A})
    # Compute FF.
    denominator = I_sc_A * result['V_oc_V']
    with numpy.errstate(divide='ignore', invalid='ignore'):
        # numpy.where() does not respect types, always giving numpy.ndarray, so cast with numpy.float64()
        FF = numpy.float64(numpy.where(denominator != 0, result['P_mp_W'] / denominator, numpy.nan))
    result.update(ensure_numpy_scalars(dictionary={'FF': FF}))

    return result


def R_at_oc(
    *, N_s: Union[int, numpy.intc, numpy.ndarray], T_degC: Union[float, numpy.float64, numpy.ndarray],
    I_ph_A: Union[float, numpy.float64, numpy.ndarray], I_rs_1_A: Union[float, numpy.float64, numpy.ndarray],
    n_1: Union[float, numpy.float64, numpy.ndarray], R_s_Ohm: Union[float, numpy.float64, numpy.ndarray],
        G_p_S: Union[float, numpy.float64, numpy.ndarray], newton_options: dict = newton_options_default) -> dict:
    """
    Compute terminal resistance at open circuit in Ohms.

    Parameters
    ----------
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
    newton_options
        Options for Newton solver (see scipy.optimize.newton).

    Returns
    -------
    result : dict
        R_oc_Ohm
            Terminal resistance at open circuit [Ω].
        V_oc_V
            Terminal open-circuit voltage [V].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    # Compute Voc.
    V_oc_V = V_at_I(I_A=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                    G_p_S=G_p_S, newton_options=newton_options)['V_V']

    # Compute slope at Voc.
    result = I_at_V_d1(V_V=V_oc_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                       G_p_S=G_p_S, newton_options=newton_options)

    # Compute resistance at Voc.
    R_oc_Ohm = -1 / result['I_d1_V_S']

    return ensure_numpy_scalars(dictionary={'R_oc_Ohm': R_oc_Ohm, 'V_oc_V': V_oc_V})


def R_at_sc(
    *, N_s: Union[int, numpy.intc, numpy.ndarray], T_degC: Union[float, numpy.float64, numpy.ndarray],
    I_ph_A: Union[float, numpy.float64, numpy.ndarray], I_rs_1_A: Union[float, numpy.float64, numpy.ndarray],
    n_1: Union[float, numpy.float64, numpy.ndarray], R_s_Ohm: Union[float, numpy.float64, numpy.ndarray],
        G_p_S: Union[float, numpy.float64, numpy.ndarray], newton_options: dict = newton_options_default) -> dict:
    """
    Compute terminal resistance at short circuit in Ohms.

    Parameters
    ----------
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
    newton_options
        Options for Newton solver (see scipy.optimize.newton).

    Returns
    -------
    result : dict
        R_sc_Ohm
            Terminal resistance at short circuit [Ω].
        I_sc_A
            Terminal short-circuit current [A].

    Notes
    -----
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    # Compute derivative at Isc.
    result = I_at_V_d1(V_V=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                       G_p_S=G_p_S, newton_options=newton_options)
    I_sc_A = result['I_A']

    # Compute resistance at Isc.
    R_sc_Ohm = -1 / result['I_d1_V_S']

    return ensure_numpy_scalars(dictionary={'R_sc_Ohm': R_sc_Ohm, 'I_sc_A': I_sc_A})


def derived_params(
    *, N_s: Union[int, numpy.intc, numpy.ndarray], T_degC: Union[float, numpy.float64, numpy.ndarray],
    I_ph_A: Union[float, numpy.float64, numpy.ndarray], I_rs_1_A: Union[float, numpy.float64, numpy.ndarray],
    n_1: Union[float, numpy.float64, numpy.ndarray], R_s_Ohm: Union[float, numpy.float64, numpy.ndarray],
    G_p_S: Union[float, numpy.float64, numpy.ndarray], newton_options: dict = newton_options_default,
        minimize_scalar_bounded_options: dict = minimize_scalar_bounded_options_default) -> dict:
    """
    Compute derived parameters.

    Parameters
    ----------
    F
        Effective irradiance ratio on device [·].
    T_degC
        Temperature of device [°C].
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
    newton_options
        Options for Newton solver (see scipy.optimize.newton).
    minimize_scalar_bounded_options
        Options for minimization solver (see scipy.optimize.minimize_scalar).

    Returns
    -------
    result : dict
        FF
            Fill Factor [·].
        I_sc_A
            Terminal short-circuit current [A].
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
    All parameters are at the device level, where the device consists
    of N_s PV cells in series in each of N_p strings in parallel.
    Inputs must be broadcast compatible. Output values are
    numpy.float64 or numpy.ndarray.
    """
    result = FF(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                newton_options=newton_options, minimize_scalar_bounded_options=minimize_scalar_bounded_options)
    R_sc_Ohm = R_at_sc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                       newton_options=newton_options)['R_sc_Ohm']
    V_x_V = result['V_oc_V'] / 2
    I_x_A = I_at_V(V_V=V_x_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm,
                   G_p_S=G_p_S, newton_options=newton_options)['I_A']
    V_xx_V = (result['V_mp_V'] + result['V_oc_V']) / 2
    I_xx_A = I_at_V(V_V=V_xx_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1,
                    R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)['I_A']
    R_oc_Ohm = R_at_oc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                       newton_options=newton_options)['R_oc_Ohm']
    result.update({'R_oc_Ohm': R_oc_Ohm, 'V_x_V': V_x_V, 'I_x_A': I_x_A, 'V_xx_V': V_xx_V, 'I_xx_A': I_xx_A,
                   'R_sc_Ohm': R_sc_Ohm})

    return result

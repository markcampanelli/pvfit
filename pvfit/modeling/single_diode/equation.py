import numpy as np
from scipy.constants import convert_temperature
from scipy.optimize import minimize_scalar, newton

from pvfit.common.constants import (k_B_J_per_K, minimize_scalar_maxiter_default, minimize_scalar_xatol_default,
                                    newton_maxiter_default, newton_tol_default, q_C)
from pvfit.common.utils import ensure_numpy_scalars


def current_sum_at_diode_node(*, V_V, I_A, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S):
    """
    Computes the sum of the currents at the high-voltage diode node in the implicit 5-parameter local single-diode
    equivalent-circuit model (SDM-L).

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Observables at operating conditions (device-level):
            V_V terminal voltage
            I_A terminal current
        Model parameters at operating conditions (device-level):
            N_s integer number of cells in series in each parallel string
            T_degC effective diode-junction temperature
            I_ph_A photocurrent
            I_rs_A diode reverse-saturation current
            n diode ideality factor
            R_s_Ohm series resistance
            G_p_S parallel conductance

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing:
            I_sum_A sum of currents at high-voltage diode node
            T_K effective diode-junction temperature
            V_diode_V voltage at high-voltage diode node
            n_mod_V modified ideality factor
    """

    # Temperature in Kelvin.
    T_K = convert_temperature(T_degC, 'Celsius', 'Kelvin')

    # Voltage at diode node.
    V_diode_V = V_V + I_A * R_s_Ohm

    # Modified ideality factor.
    n_mod_V = (N_s * n * k_B_J_per_K * T_K) / q_C

    # Sum of currents at diode node. np.expm1() returns a np.float64 when arguments are all python/numpy scalars.
    I_sum_A = I_ph_A - I_rs_A * np.expm1(V_diode_V / n_mod_V) - G_p_S * V_diode_V - I_A

    result = {'I_sum_A': I_sum_A, 'T_K': T_K, 'V_diode_V': V_diode_V, 'n_mod_V': n_mod_V}

    return ensure_numpy_scalars(dictionary=result)


def I_at_V(*, V_V, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
           newton_maxiter=newton_maxiter_default):
    """
    Compute terminal current from terminal voltage using Newton's method.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as current_sum_at_diode_node(), but with removal of I_A and addition of:
            newton_tol (optional) tolerance for Newton solver
            newton_maxiter (optional) maximum number of iterations for Newton solver

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of current_sum_at_diode_node() with the addition of:
            I_A terminal current

    Compute strategy:
        1) Compute initial condition for I_A with explicit equation using R_s_Ohm==0.
        2) Compute using Newton's method.
    """

    # Optimization.
    n_mod_V = (N_s * n * k_B_J_per_K * convert_temperature(T_degC, 'Celsius', 'Kelvin')) / q_C

    # Preserve shape of excluded R_s_Ohm in inital condition.
    V_diode_V_ic = V_V + 0. * R_s_Ohm

    # Compute initially with zero R_s_Ohm.
    I_A_ic = I_ph_A - I_rs_A * np.expm1(V_diode_V_ic / n_mod_V) - G_p_S * V_diode_V_ic

    # Use closures in function definitions for Newton's method.
    def func(I_A):
        V_diode_V = V_V + I_A * R_s_Ohm
        return I_ph_A - I_rs_A * np.expm1(V_diode_V / n_mod_V) - G_p_S * V_diode_V - I_A

    def fprime(I_A):
        return -I_rs_A * R_s_Ohm / n_mod_V * np.exp((V_V + I_A * R_s_Ohm) / n_mod_V) - G_p_S * R_s_Ohm - 1.

    # FUTURE Consider using this in Halley's method.
    # def fprime2(I_A):
    #     return -I_rs_A * (R_s_Ohm / n_mod_V)**2 * np.exp((V_V + I_A * R_s_Ohm) / n_mod_V)

    # Solve for I_A using Newton's method.
    I_A = newton(func, I_A_ic, fprime=fprime, tol=newton_tol, maxiter=newton_maxiter)

    # Verify convergence, because newton() documentation says that this should be checked.
    result = current_sum_at_diode_node(
        V_V=V_V, I_A=I_A, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)
    np.testing.assert_array_almost_equal(result['I_sum_A'], 0)
    result.update(ensure_numpy_scalars(dictionary={'I_A': I_A}))

    return result


def I_at_V_d1(*, V_V, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
              newton_maxiter=newton_maxiter_default):
    """
    Compute 1st derivative of terminal current with respect to terminal voltage at specified terminal voltage.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as I_at_V()).

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of I_at_V() with the addition of:
            I_d1_V_S 1st derivative of terminal current w.r.t terminal voltage

    Notes:
        This derivative is needed, e.g., for R_oc_Ohm and R_sc_Ohm calculations.
    """

    # Compute terminal current.
    result = I_at_V(V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                    newton_tol=newton_tol, newton_maxiter=newton_maxiter)

    # Compute first derivative of current with respect to voltage at specified voltage.
    expr1 = I_rs_A / result['n_mod_V'] * np.exp(result['V_diode_V'] / result['n_mod_V']) + G_p_S
    I_d1_V_S = -expr1 / (1. + R_s_Ohm * expr1)
    result.update(ensure_numpy_scalars(dictionary={'I_d1_V_S': I_d1_V_S}))

    return result


def V_at_I(*, I_A, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
           newton_maxiter=newton_maxiter_default):
    """
    Compute terminal voltage from terminal current using Newton's method.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as current_sum_at_diode_node(), but with removal of V_V and addition of:
            newton_tol (optional) tolerance for Newton solver
            newton_maxiter (optional) maximum number of iterations for Newton solver

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of current_sum_at_diode_node() with the addition of:
            V_V terminal voltage

    Compute strategy:
        1) Compute initial condition for V_V with explicit equation using G_p_S==0.
        2) Compute using Newton's method.
    """

    # Optimization.
    n_mod_V = (N_s * n * k_B_J_per_K * convert_temperature(T_degC, 'Celsius', 'Kelvin')) / q_C

    # Compute initially with zero G_p_S, preserving shape of excluded G_p_S in inital condition.
    V_V_ic = n_mod_V * (np.log(I_ph_A + I_rs_A - 0. * G_p_S - I_A) - np.log(I_rs_A)) - I_A * R_s_Ohm

    # Use closures in these function definitions for Newton's method.
    def func(V_V):
        V_diode_V = V_V + I_A * R_s_Ohm
        return I_ph_A - I_rs_A * np.expm1(V_diode_V / n_mod_V) - G_p_S * V_diode_V - I_A

    def fprime(V_V):
        return -I_rs_A / n_mod_V * np.exp((V_V + I_A * R_s_Ohm) / n_mod_V) - G_p_S

    # FUTURE Consider using this in Halley's method.
    # def fprime2(V_V):
    #     return -I_rs_A / n_mod_V**2. * np.exp((V_V + I_A * R_s_Ohm) / n_mod_V)

    # Solve for V_V using Newton's method.
    V_V = newton(func, V_V_ic, fprime=fprime, tol=newton_tol, maxiter=newton_maxiter)

    # Verify convergence. newton() documentation says that this should be checked.
    result = current_sum_at_diode_node(
        V_V=V_V, I_A=I_A, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)
    np.testing.assert_array_almost_equal(result['I_sum_A'], 0.)
    result.update(ensure_numpy_scalars(dictionary={'V_V': V_V}))

    return result


def V_at_I_d1(*, I_A, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
              newton_maxiter=newton_maxiter_default):
    """
    Compute 1st derivative of terminal voltage with respect to terminal current at specified terminal current.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as V_at_I()).

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of V_at_I() with the addition of:
            V_d1_I_Ohm 1st derivative of terminal voltage w.r.t terminal current

    Notes:
        This derivative is needed, e.g., for solving the differential equation for capacitor charging.
    """

    # Compute terminal voltage.
    result = V_at_I(I_A=I_A, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                    newton_tol=newton_tol, newton_maxiter=newton_maxiter)

    # Compute first derivative of voltage with respect to current at specified current.
    expr1 = I_rs_A / result['n_mod_V'] * np.exp(result['V_diode_V'] / result['n_mod_V']) + G_p_S
    V_d1_I_Ohm = -1. / expr1 - R_s_Ohm
    result.update(ensure_numpy_scalars(dictionary={'V_d1_I_Ohm': V_d1_I_Ohm}))

    return result


def P_at_V(*, V_V, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
           newton_maxiter=newton_maxiter_default):
    """
    Compute terminal power from terminal voltage.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as I_at_V().

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of I_at_V() with the addition of:
            P_W terminal power
    """

    # Compute power.
    result = I_at_V(V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                    newton_tol=newton_tol, newton_maxiter=newton_maxiter)
    P_W = V_V * result['I_A']
    result.update(ensure_numpy_scalars(dictionary={'P_W': P_W}))

    return result


def P_mp(*, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
         newton_maxiter=newton_maxiter_default, minimize_scalar_xatol=minimize_scalar_xatol_default,
         minimize_scalar_maxiter=minimize_scalar_maxiter_default):
    """
    Compute maximum terminal power.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as P_at_V(), but with removal of V_V.

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing:
            P_mp_W maximum power
            V_mp_V voltage at maximum power
            I_mp_A current at maximum power
            V_oc_V voltage at open circuit

    Compute strategy:
        1) Compute solution bracketing interval as [0, Voc].
        2) Compute maximum power on solution bracketing interval using scipy.optimize.minimize_scalar().
    """

    # Compute Voc for assumed Vmp bracket [0, Voc].
    V_oc_V = V_at_I(I_A=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                    newton_tol=newton_tol, newton_maxiter=newton_maxiter)['V_V']

    # This allows us to make a ufunc out of minimize_scalar(). Note closures over solver arguments/options.
    def opposite_P_at_V(V_V, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S):
        return -P_at_V(V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                       newton_tol=newton_tol, newton_maxiter=newton_maxiter)['P_W']
    options = {'xatol': minimize_scalar_xatol, 'maxiter': minimize_scalar_maxiter}
    array_min_func = np.frompyfunc(lambda V_oc_V, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S: minimize_scalar(
        opposite_P_at_V, bounds=(0., V_oc_V), args=(N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S), method='bounded',
        options=options), 8, 1)

    # Solve for the array of OptimizeResult objects.
    res_array = array_min_func(V_oc_V, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S)

    # Verify convergence. Note that np.frompyfunc() always returns a PyObject array, which must be cast.
    if not np.all(np.array(np.frompyfunc(lambda res: res.success, 1, 1)(res_array), dtype=bool)):
        raise ValueError(f"mimimize_scalar() with method='bounded' did not converge for options={options}.")

    # Collect results. Casting with np.float64() creates numpy.ndarray if needed.
    V_mp_V = np.float64(np.frompyfunc(lambda res: res.x, 1, 1)(res_array))
    P_mp_W = np.float64(np.frompyfunc(lambda res: -res.fun, 1, 1)(res_array))
    with np.errstate(divide='ignore', invalid='ignore'):
        # numpy.where() does not respect types, always giving numpy.ndarray, so cast with np.float64().
        I_mp_A = np.float64(np.where(V_mp_V != 0., P_mp_W / V_mp_V, 0.))

    return ensure_numpy_scalars(dictionary={'P_mp_W': P_mp_W, 'I_mp_A': I_mp_A, 'V_mp_V': V_mp_V, 'V_oc_V': V_oc_V})


def FF(*, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
       newton_maxiter=newton_maxiter_default, minimize_scalar_xatol=minimize_scalar_xatol_default,
       minimize_scalar_maxiter=minimize_scalar_maxiter_default):
    """
    Compute fill factor (unitless fraction).

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
       Same as P_mp().

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of P_mp() with the addition of:
            FF fill factor
            I_sc_A short-circuit current
    """

    # Compute Pmp.
    result = P_mp(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                  newton_tol=newton_tol, newton_maxiter=newton_maxiter, minimize_scalar_xatol=minimize_scalar_xatol,
                  minimize_scalar_maxiter=minimize_scalar_maxiter)
    # Compute Isc.
    I_sc_A = I_at_V(V_V=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                    newton_tol=newton_tol, newton_maxiter=newton_maxiter)['I_A']
    result.update({'I_sc_A': I_sc_A})
    # Compute FF.
    denominator = I_sc_A * result['V_oc_V']
    with np.errstate(divide='ignore', invalid='ignore'):
        # numpy.where() does not respect types, always giving numpy.ndarray, so cast with np.float64()
        FF = np.float64(np.where(denominator != 0, result['P_mp_W'] / denominator, np.nan))
    result.update(ensure_numpy_scalars(dictionary={'FF': FF}))

    return result


def R_at_oc(*, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
            newton_maxiter=newton_maxiter_default):
    """
    Compute resistance at open circuit in Ohms.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
       Same as P_mp().

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing:
            R_oc_Ohm resistance at open circuit
            V_oc_V open-circuit voltage
    """

    # Compute Voc.
    V_oc_V = V_at_I(I_A=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                    newton_tol=newton_tol, newton_maxiter=newton_maxiter)['V_V']

    # Compute slope at Voc.
    result = I_at_V_d1(V_V=V_oc_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm,
                       G_p_S=G_p_S, newton_tol=newton_tol, newton_maxiter=newton_maxiter)

    # Compute resistance at Voc.
    R_oc_Ohm = -1 / result['I_d1_V_S']

    return ensure_numpy_scalars(dictionary={'R_oc_Ohm': R_oc_Ohm, 'V_oc_V': V_oc_V})


def R_at_sc(*, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
            newton_maxiter=newton_maxiter_default):
    """
    Compute resistance at short circuit in Ohms.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as P_mp().

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing:
            R_sc_Ohm resistance at short circuit
            I_sc_A short-circuit current
    """

    # Compute derivative at Isc.
    result = I_at_V_d1(V_V=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                       newton_tol=newton_tol, newton_maxiter=newton_maxiter)
    I_sc_A = result['I_A']

    # Compute resistance at Isc.
    R_sc_Ohm = -1 / result['I_d1_V_S']

    return ensure_numpy_scalars(dictionary={'R_sc_Ohm': R_sc_Ohm, 'I_sc_A': I_sc_A})


def derived_params(*, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S, newton_tol=newton_tol_default,
                   newton_maxiter=newton_maxiter_default, minimize_scalar_xatol=minimize_scalar_xatol_default,
                   minimize_scalar_maxiter=minimize_scalar_maxiter_default):
    """
    Compute derived parameters.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as P_mp().

    Outputs (device-level, at each combination of broadcast inputs, return type is np.float64 when all scalar inputs):
        dict containing the outputs of FF() with the addition of:
            R_oc_Ohm resistance at open circuit
            R_sc_Ohm resistance at short circuit
    """

    result = FF(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                newton_tol=newton_tol, newton_maxiter=newton_maxiter, minimize_scalar_xatol=minimize_scalar_xatol,
                minimize_scalar_maxiter=minimize_scalar_maxiter)
    R_oc_Ohm = R_at_oc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                       newton_tol=newton_tol, newton_maxiter=newton_maxiter)['R_oc_Ohm']
    R_sc_Ohm = R_at_sc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_A=I_rs_A, n=n, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S,
                       newton_tol=newton_tol, newton_maxiter=newton_maxiter)['R_sc_Ohm']
    result.update({'R_oc_Ohm': R_oc_Ohm, 'R_sc_Ohm': R_sc_Ohm})

    return result

import numpy
from scipy.constants import convert_temperature
from scipy.optimize import minimize_scalar, newton

from pvfit.common.constants import k_B_J_per_K, minimize_scalar_bounded_options_default, newton_options_default, q_C


def current_sum_at_diode_node(*, V_V, I_A, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S):
    """
    Computes the sum of the currents at the diode's anode node in the 7-parameter double-diode
    equation (DDE) equivalent-circuit model at a single temperature and irradiance.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Observables at operating conditions (device-level):
            V_V terminal voltage
            I_A terminal current
        Model parameters at operating conditions (device-level):
            N_s integer number of cells in series in each parallel string
            T_degC effective diode-junction temperature
            I_ph_A photocurrent
            I_rs_1_A first diode reverse-saturation current
            n_1 first diode ideality factor
            I_rs_2_A second diode reverse-saturation current
            n_2 second diode ideality factor
            R_s_Ohm series resistance
            G_p_S parallel conductance

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing:
            I_sum_A sum of currents at high-voltage diodes node
            V_diode_V voltage at high-voltage diodes node
            n_mod_1_V first modified diode ideality factor
            n_mod_2_V second modified diode ideality factor
        """

    # Optimizations.
    T_K = convert_temperature(T_degC, 'Celsius', 'Kelvin')
    V_therm_factor_V = (N_s * k_B_J_per_K * T_K) / q_C
    # Modified diode ideality factors
    n_mod_1_V = n_1 * V_therm_factor_V
    n_mod_2_V = n_2 * V_therm_factor_V
    # Voltage at diode node.
    V_diode_V = V_V + I_A * R_s_Ohm

    # Sum of currents at diode node. numpy.expm1() returns a numpy.float64 when arguments are all python/numpy scalars.
    I_sum_A = I_ph_A - I_rs_1_A * numpy.expm1(V_diode_V / n_mod_1_V) - I_rs_2_A * numpy.expm1(V_diode_V / n_mod_2_V) - \
        G_p_S * V_diode_V - I_A

    # Make sure to return numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    if isinstance(I_sum_A, numpy.float64) and any(map(lambda x: isinstance(x, numpy.ndarray), [
            V_V, I_A, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S, N_s, T_degC])):
        I_sum_A = numpy.array(I_sum_A)
        T_K = numpy.array(T_K)
        V_diode_V = numpy.array(V_diode_V)
        n_mod_1_V = numpy.array(n_mod_1_V)
        n_mod_2_V = numpy.array(n_mod_2_V)
    elif isinstance(I_sum_A, numpy.float64):
        T_K = numpy.float64(T_K)
        V_diode_V = numpy.float64(V_diode_V)
        n_mod_1_V = numpy.float64(n_mod_1_V)
        n_mod_2_V = numpy.float64(n_mod_2_V)
    else:  # numpy.ndarray
        T_K = numpy.ones_like(I_sum_A) * T_K
        V_diode_V = numpy.ones_like(I_sum_A) * V_diode_V
        n_mod_1_V = numpy.ones_like(I_sum_A) * n_mod_1_V
        n_mod_2_V = numpy.ones_like(I_sum_A) * n_mod_2_V

    return {'I_sum_A': I_sum_A, 'T_K': T_K, 'V_diode_V': V_diode_V, 'n_mod_1_V': n_mod_1_V, 'n_mod_2_V': n_mod_2_V}


def I_at_V(*, V_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S,
           newton_options=newton_options_default):
    """
    Compute terminal current from terminal voltage using Newton's method.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as current_sum_at_diode_node(), but with removal of I_A and addition of:
            newton_options (optional) options for Newton solver

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing the outputs of current_sum_at_diode_node() with the addition of:
            I_A terminal current

    Compute strategy:
        1) Compute initial condition for I_A with explicit equation using R_s_Ohm==0.
        2) Compute using Newton's method.
    """

    # Optimization.
    V_therm_factor_V = (N_s * k_B_J_per_K * convert_temperature(T_degC, 'Celsius', 'Kelvin')) / q_C
    n_mod_1_V = n_1 * V_therm_factor_V
    n_mod_2_V = n_2 * V_therm_factor_V

    # Preserve shape of excluded R_s_Ohm in inital condition.
    V_diode_V_ic = V_V + 0. * R_s_Ohm

    # Compute initially with zero R_s_Ohm.
    I_A_ic = I_ph_A - I_rs_1_A * numpy.expm1(V_diode_V_ic / n_mod_1_V) - \
        I_rs_2_A * numpy.expm1(V_diode_V_ic / n_mod_2_V) - G_p_S * V_diode_V_ic

    # Use closures in function definitions for Newton's method.
    def func(I_A):
        V_diode_V = V_V + I_A * R_s_Ohm
        return I_ph_A - I_rs_1_A * numpy.expm1(V_diode_V / n_mod_1_V) - \
            I_rs_2_A * numpy.expm1(V_diode_V / n_mod_2_V) - G_p_S * V_diode_V - I_A

    def fprime(I_A):
        V_diode_V = V_V + I_A * R_s_Ohm
        return -I_rs_1_A * R_s_Ohm / n_mod_1_V * numpy.exp(V_diode_V / n_mod_1_V) - \
            I_rs_2_A * R_s_Ohm / n_mod_2_V * numpy.exp(V_diode_V / n_mod_2_V) - G_p_S * R_s_Ohm - 1

    # FUTURE Consider using this in Halley's method.
    # def fprime2(I_A):
    #     V_diode_V = V_V + I_A * R_s_Ohm
    #     return -I_rs_1_A * (R_s_Ohm / n_mod_1_V)**2 * numpy.exp(V_diode_V / n_mod_1_V) - \
    #         I_rs_2_A * (R_s_Ohm / n_mod_2_V)**2 * numpy.exp(V_diode_V / n_mod_2_V)

    # Solve for I_A using Newton's method.
    I_A = newton(func, I_A_ic, fprime=fprime, **newton_options)

    # Verify convergence. newton() documentation says that this should be checked.
    result = current_sum_at_diode_node(V_V=V_V, I_A=I_A, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A,
                                       n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)
    numpy.testing.assert_array_almost_equal(result['I_sum_A'], 0)

    # Make sure to return an numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    if not isinstance(I_A, numpy.ndarray) and any(map(lambda x: isinstance(x, numpy.ndarray), [
            V_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_1, R_s_Ohm, G_p_S])):
        I_A = numpy.array(I_A)

    result.update({'I_A': I_A})

    return result


def I_at_V_d1(*, V_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S,
              newton_options=newton_options_default):
    """
    Compute 1st derivative of terminal current with respect to terminal voltage at specified terminal voltage.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as I_at_V()).

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing the outputs of I_at_V() with the addition of:
            I_d1_V_S 1st derivative of terminal current w.r.t terminal voltage

    Notes:
        This derivative is needed for R_oc_Ohm and R_sc_Ohm calculations.
    """

    # Compute current.
    result = I_at_V(V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A,
                    n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)

    # Calculate first derivative of current with respect to voltage
    expr1 = I_rs_1_A / result['n_mod_1_V'] * numpy.exp(result['V_diode_V'] / result['n_mod_1_V']) + \
        I_rs_2_A / result['n_mod_2_V'] * numpy.exp(result['V_diode_V'] / result['n_mod_2_V']) + G_p_S
    I_d1_V_S = -expr1 / (1. + R_s_Ohm * expr1)

    # Make sure to return an numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    if not isinstance(I_d1_V_S, numpy.ndarray) and isinstance(result['I_A'], numpy.ndarray):
        I_d1_V_S = numpy.array(I_d1_V_S)

    result.update({'I_d1_V_S': I_d1_V_S})

    return result


def V_oc(*, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S,
         newton_options=newton_options_default):
    """
    Compute open-circuit voltage (terminal voltage where terminal current is zero).

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
      Same as current_sum_at_diode_node(), with the exception of V_V and I_A

    Outputs (device-level, at each combination of inputs):
        dict containing:
            V_oc_V open-circuit voltage

    # Compute strategy:
        1) Use V=0-centered second degree polynomial approximation of the (V=Voc, I=0)-substituted DDM equation to get
           an initial approximation Voc that must be greater than the true Voc.
        2) If that Voc approximation causes large negative current in the I-V curve, then bisect Voc and iterate with
           the new centering point.
        3) Solve using Newton's method that must reliably converge to Voc from a starting Voc that is greater than the
           true value, because of the curvature of the function to be zeroed.
    """

    # First estimate of Voc uses quadratic approximation to Voc equation centered at V_V=0.
    # Initially broadcasts I_ph_A value, using current_sum_at_diode_node() to properly preserve types.
    result = current_sum_at_diode_node(V_V=0, I_A=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1,
                                       I_rs_2_A=I_rs_2_A, n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)

    # Record the return type, then ensure that an array computation can be done.
    is_ndarray = isinstance(result['I_sum_A'], numpy.ndarray)
    pos_idx = 0 < result['I_sum_A']  # Will cast a rank-0 array to a numpy scalar, due to numpy vaguarity!
    # Ensure that all inputs can be indexed as numpy.ndarray.
    I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S = numpy.broadcast_arrays(
        I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S)
    # Starting Voc.
    V_oc_V_est = numpy.zeros_like(result['I_sum_A'])

    # Work with voltage-independent modified ideality factors to simplify equations.
    n_mod_1_V = result['n_mod_1_V']
    n_mod_2_V = result['n_mod_2_V']

    # Define functions needed for solver.
    def func(V_V, I_ph_A, I_rs_1_A, n_mod_1_V, I_rs_2_A, n_mod_2_V, R_s_Ohm, G_p_S):
        """The zero of this function is Voc."""

        return I_ph_A - I_rs_1_A * numpy.expm1(V_V / n_mod_1_V) - I_rs_2_A * numpy.expm1(V_V / n_mod_2_V) - G_p_S * V_V

    def fprime(V_V, I_ph_A, I_rs_1_A, n_mod_1_V, I_rs_2_A, n_mod_2_V, R_s_Ohm, G_p_S):
        """Derivative of func w.r.t. V_V."""

        return -I_rs_1_A / n_mod_1_V * numpy.exp(V_V / n_mod_1_V) - \
            I_rs_2_A / n_mod_2_V * numpy.exp(V_V / n_mod_2_V) - G_p_S

    def qudratic_estimate(*, V_V, I_ph_A, I_rs_1_A, n_mod_1_V, I_rs_2_A, n_mod_2_V, R_s_Ohm, G_p_S):
        """Estimate Voc using a quadratic approximation centered at V_V."""

        # Compute quadratic coefficients from series expansion.
        a = -(I_rs_1_A / n_mod_1_V**2 * numpy.exp(V_V / n_mod_1_V) +
              I_rs_2_A / n_mod_2_V**2. * numpy.exp(V_V / n_mod_2_V)) / 2
        b = fprime(V_V, I_ph_A, I_rs_1_A, n_mod_1_V, I_rs_2_A, n_mod_2_V, R_s_Ohm, G_p_S)
        c = func(V_V, I_ph_A, I_rs_1_A, n_mod_1_V, I_rs_2_A, n_mod_2_V, R_s_Ohm, G_p_S)

        # Take largest root of upside-down parabola that should have a vertex in first quadrant.
        return V_V + (-b - numpy.sqrt(b**2. - 4. * a * c)) / (2. * a)

    while numpy.any(pos_idx):
        # Record current estimates in case we need to iterate.
        V_oc_V_est_last = numpy.copy(V_oc_V_est)

        # Because of curvature ordering of actual vs. approximation, this estimate is guaranteed to be greater than Voc.
        V_oc_V_est[pos_idx] = qudratic_estimate(
            V_V=V_oc_V_est[pos_idx], I_ph_A=I_ph_A[pos_idx], I_rs_1_A=I_rs_1_A[pos_idx], n_mod_1_V=n_mod_1_V[pos_idx],
            I_rs_2_A=I_rs_2_A[pos_idx], n_mod_2_V=n_mod_2_V[pos_idx], R_s_Ohm=R_s_Ohm[pos_idx], G_p_S=G_p_S[pos_idx])

        # Find any voltage estimates that blew the current up negative because they were too large.
        large_neg_idx = func(V_V=V_oc_V_est, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_mod_1_V=n_mod_1_V, I_rs_2_A=I_rs_2_A,
                             n_mod_2_V=n_mod_2_V, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S) < -I_ph_A

        while numpy.any(large_neg_idx):
            # Bisect to last estimate until function evaluation is not too large negative.
            V_oc_V_est[large_neg_idx] = (V_oc_V_est[large_neg_idx] + V_oc_V_est_last[large_neg_idx]) / 2.
            large_neg_idx = func(V_V=V_oc_V_est, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_mod_1_V=n_mod_1_V,
                                 I_rs_2_A=I_rs_2_A, n_mod_2_V=n_mod_2_V, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S) < -I_ph_A

        # Some of these bisected values may evaluate positive again, so repeat quadratic approximation for them.
        pos_idx = 0. < func(V_V=V_oc_V_est, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_mod_1_V=n_mod_1_V, I_rs_2_A=I_rs_2_A,
                            n_mod_2_V=n_mod_2_V, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)

    # All Voc estimates should be >= Voc and have finite evaluation in func, so now it's safe to use Newton's method.
    V_oc_V = newton(func, V_oc_V_est, fprime=fprime, args=(
        I_ph_A, I_rs_1_A, n_mod_1_V, I_rs_2_A, n_mod_2_V, R_s_Ohm, G_p_S), **newton_options)

    # Verify convergence. newton() documentation says that this should be checked.
    numpy.testing.assert_array_almost_equal(current_sum_at_diode_node(
        V_V=V_oc_V, I_A=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A,
        n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)['I_sum_A'], 0)

    if is_ndarray:
        return {'V_oc_V': V_oc_V}
    else:
        return {'V_oc_V': numpy.asscalar(V_oc_V)}


def V_at_I(*, I_A, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S,
           newton_options=newton_options_default):
    """
    Compute terminal voltage from terminal current using Newton's method.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as current_sum_at_diode_node(), but with removal of v_V and addition of:
            newton_options (optional) options for Newton solver

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing the outputs of current_sum_at_diode_node() with the addition of:
            v_V terminal voltage

    Compute strategy:
        1) Compute initial condition for V_V with explicit piecewise linear approximation to I-V curve using the tangent
           lines passing through the I-V curve at Isc and Voc.
        2) Compute using Newton's method.
    """

    # Optimization.
    V_therm_factor_V = (N_s * k_B_J_per_K * convert_temperature(T_degC, 'Celsius', 'Kelvin')) / q_C
    n_mod_1_V = n_1 * V_therm_factor_V
    n_mod_2_V = n_2 * V_therm_factor_V

    # Compute Isc and resistance (equiv. slope) at Isc, which are generally arrays.
    result = R_sc(V_V=0, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
                  R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options_default)
    I_sc_A = result['I_sc_A']
    R_sc_Ohm = result['R_sc_Ohm']

    # Compute Voc and resistance (equiv. slope) at Voc, which are generally arrays.
    result = R_oc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
                  R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options_default)
    V_oc_V = result['V_oc_V']
    R_oc_Ohm = result['R_oc_Ohm']

    # Compute I_A knot point(s) in the piecewise linear approximation(s) to the I-V curve(s) used as the IC.
    I_knot_A = (V_oc_V - I_sc_A * R_sc_Ohm) / (R_oc_Ohm - R_sc_Ohm)
    V_V_ic = numpy.where(I_A <= I_knot_A, V_oc_V - I_A * R_oc_Ohm, R_sc_Ohm * (I_sc_A - I_A))

    # Make sure to return numpy.float64 if all inputs were that type, because numpy.where() always produces
    # numpy.ndarray.
    if not isinstance(I_A, numpy.ndarray):
        V_V_ic = numpy.float64(V_V_ic)

    # Use closures in function definitions for Newton's method.
    def func(V_V):
        V_diode_V = V_V + I_A * R_s_Ohm
        return I_ph_A - I_rs_1_A * numpy.expm1(V_diode_V / n_mod_1_V) - \
            I_rs_2_A * numpy.expm1(V_diode_V / n_mod_2_V) - G_p_S * V_diode_V - I_A

    def fprime(V_V):
        V_diode_V = V_V + I_A * R_s_Ohm
        return -I_rs_1_A / n_mod_1_V * numpy.exp(V_diode_V / n_mod_1_V) - \
            I_rs_2_A / n_mod_2_V * numpy.exp(V_diode_V / n_mod_2_V) - G_p_S

    # FUTURE Consider using this in Halley's method.
    # def fprime2(V_V):
    #     v_diode_V = V_V + I_A * R_s_Ohm
    #     return -I_rs_1_A / n_mod_1_V**2. * numpy.exp(v_diode_V / n_mod_1_V) - \
    #         I_rs_2_A / n_mod_2_V**2. * numpy.exp(v_diode_V / n_mod_2_V)

    # Solve for V_V using Newton's method.
    V_V = newton(func, V_V_ic, fprime=fprime, **newton_options)

    # Verify convergence. newton() documentation says that this should be checked.
    result = current_sum_at_diode_node(V_V=V_V, I_A=I_A, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A,
                                       n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S)
    numpy.testing.assert_array_almost_equal(result['I_sum_A'], 0)

    # Make sure to return numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    if not isinstance(V_V, numpy.ndarray) and any(map(lambda x: isinstance(x, numpy.ndarray), [
            I_A, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_1, R_s_Ohm, G_p_S])):
        V_V = numpy.array(V_V)

    result.update({'V_V': V_V})

    return result


def P_at_V(*, V_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S,
           newton_options=newton_options_default):
    """
    Compute terminal power from terminal voltage.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as I_at_V().

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing the outputs of I_at_V() with the addition of:
            P_W terminal power
    """

    result = I_at_V(V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A,
                    n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options_default)
    P_W = V_V * result['I_A']

    # Make sure to return numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    if not isinstance(P_W, numpy.ndarray) and isinstance(result['I_A'], numpy.ndarray):
        P_W = numpy.array(P_W)

    result.update({'P_W': P_W})

    return result


def P_mp(*, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S,
         minimize_scalar_bounded_options=minimize_scalar_bounded_options_default,
         newton_options=newton_options_default):
    """
    Compute maximum terminal power.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as power_from_voltage(), but with removal of V_V.

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
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
    V_oc_V = V_oc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
                  R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)['V_oc_V']

    # This allows us to make a ufunc out of minimize_scalar(). Note closures over solver arguments/options.
    def opposite_P_at_V(V_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S):
        return -P_at_V(
            V_V=V_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
            R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)['P_W']
    array_min_func = numpy.frompyfunc(
        lambda V_oc_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S: minimize_scalar(
            opposite_P_at_V, bounds=(0., V_oc_V), args=(
                N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S), method='bounded',
            options=minimize_scalar_bounded_options), 10, 1)

    # Solve for the array of OptimizeResult objects.
    res_array = array_min_func(V_oc_V, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S)

    # Verify convergence. Note that numpy.frompyfunc() always returns a PyObject array, which must be cast.
    if not numpy.all(numpy.array(numpy.frompyfunc(lambda res: res.success, 1, 1)(res_array), dtype=bool)):
        raise ValueError(
            f"mimimize_scalar() with method='bounded' did not converge for options={minimize_scalar_bounded_options}.")

    # Make sure to return an numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    V_mp_V = numpy.float64(numpy.frompyfunc(lambda res: res.x, 1, 1)(res_array))
    if not isinstance(V_mp_V, numpy.ndarray) and isinstance(V_oc_V, numpy.ndarray):
        V_mp_V = numpy.array(V_mp_V)

    # Make sure to return an numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    P_mp_W = numpy.float64(numpy.frompyfunc(lambda res: -res.fun, 1, 1)(res_array))
    if not isinstance(P_mp_W, numpy.ndarray) and isinstance(V_oc_V, numpy.ndarray):
        P_mp_W = numpy.array(P_mp_W)

    # Make sure to return an numpy.float64 if all inputs were that type, because numpy.where() always produces
    # numpy.ndarray.
    with numpy.errstate(divide='ignore', invalid='ignore'):
        I_mp_A = numpy.where(V_mp_V != 0, P_mp_W / V_mp_V, 0.)
    if not isinstance(V_oc_V, numpy.ndarray):
        I_mp_A = numpy.float64(I_mp_A)

    return {'P_mp_W': P_mp_W, 'I_mp_A': I_mp_A, 'V_mp_V': V_mp_V, 'V_oc_V': V_oc_V}


def FF(*, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S,
       minimize_scalar_bounded_options=minimize_scalar_bounded_options_default,
       newton_options=newton_options_default, ):
    """
    Compute fill factor (unitless fraction).

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
       Same as P_mp().

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing the outputs of P_mp() with the addition of:
            FF fill factor
            I_sc_A short-circuit current
    """

    # Compute Isc.
    I_sc_A = I_at_V(V_V=0., N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A,
                    n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)['I_A']
    # Compute Pmp.
    result = P_mp(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
                  R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, minimize_scalar_bounded_options=minimize_scalar_bounded_options_default,
                  newton_options=newton_options)
    # Compute FF.
    denominator = I_sc_A * result['V_oc_V']
    with numpy.errstate(divide='ignore', invalid='ignore'):
        FF = numpy.where(denominator != 0, result['P_mp_W'] / denominator, 0.)

    # Make sure to return an numpy.float64 if all inputs were that type, because numpy.where() always produces
    # numpy.ndarray.
    if not isinstance(I_sc_A, numpy.ndarray):
        FF = numpy.float64(FF)

    result.update({'FF': FF, 'I_sc_A': I_sc_A})

    return result


def R_oc(*, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S, newton_options=newton_options_default):
    """
    Compute resistance at open circuit in Ohms.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
       Same as P_mp().

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing:
            R_oc_Ohm resistance at open circuit
            V_oc_V open-circuit voltage
    """

    # Compute Voc.
    V_oc_V = V_oc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
                  R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)['V_oc_V']

    # Compute slope at Voc.
    result = I_at_V_d1(V_V=V_oc_V, N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A,
                       n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)

    # Compute resistance at Voc.
    R_oc_Ohm = -1 / result['I_d1_V_S']

    # Make sure to return an numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    if not isinstance(R_oc_Ohm, numpy.ndarray) and isinstance(V_oc_V, numpy.ndarray):
        R_oc_Ohm = numpy.array(R_oc_Ohm)

    return {'R_oc_Ohm': R_oc_Ohm, 'V_oc_V': V_oc_V}


def R_sc(*, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S, newton_options=newton_options_default):
    """
    Compute resistance at short circuit in Ohms.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as P_mp().

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing:
            R_sc_Ohm resistance at short circuit
            I_sc_A short-circuit current
    """

    # Compute derivative at Isc.
    result = I_at_V_d1(V_V=0., N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A,
                       n_2=n_2, R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)
    I_sc_A = result['I_A']

    # Compute resistance at Isc.
    R_sc_Ohm = -1 / result['I_d1_V_S']

    # Make sure to return an numpy.ndarray if any input was that type (undoes casting of rank-0 numpy.ndarray to
    # numpy.float64).
    if not isinstance(R_sc_Ohm, numpy.ndarray) and isinstance(I_sc_A, numpy.ndarray):
        R_sc_Ohm = numpy.array(R_sc_Ohm)

    return {'R_sc_Ohm': R_sc_Ohm, 'I_sc_A': I_sc_A}


def iv_params(*, N_s, T_degC, I_ph_A, I_rs_1_A, n_1, I_rs_2_A, n_2, R_s_Ohm, G_p_S,
              minimize_scalar_bounded_options=minimize_scalar_bounded_options_default,
              newton_options=newton_options_default):
    """
    Compute I-V curve parameters.

    Inputs (any broadcast-compatible combination of python/numpy scalars and numpy arrays):
        Same as P_mp().

    Outputs (device-level, at each combination of broadcast inputs, return type is numpy.float64 for all scalar inputs):
        dict containing the outputs of FF() with the addition of:
            R_oc_Ohm resistance at open circuit
            R_sc_Ohm resistance at short circuit
    """

    result = FF(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
                R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, minimize_scalar_bounded_options=minimize_scalar_bounded_options,
                newton_options=newton_options)
    R_oc_Ohm = R_oc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
                    R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)['R_oc_Ohm']
    R_sc_Ohm = R_sc(N_s=N_s, T_degC=T_degC, I_ph_A=I_ph_A, I_rs_1_A=I_rs_1_A, n_1=n_1, I_rs_2_A=I_rs_2_A, n_2=n_2,
                    R_s_Ohm=R_s_Ohm, G_p_S=G_p_S, newton_options=newton_options)['R_sc_Ohm']
    result.update({'R_oc_Ohm': R_oc_Ohm, 'R_sc_Ohm': R_sc_Ohm})

    return result

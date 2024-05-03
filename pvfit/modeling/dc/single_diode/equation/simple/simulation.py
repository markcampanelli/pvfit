"""
PVfit: Single-diode equation (SDE) simulation.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional

import numpy
from scipy.optimize import brentq, newton

from pvfit.measurement.iv.types import IVCurveParametersArray, IVData
from pvfit.modeling.dc.common import get_scaled_thermal_voltage
from pvfit.modeling.dc.single_diode.equation.simple.types import ModelParameters
from pvfit.types import FloatArray, FloatBroadcastable, NewtonOptions


def I_sum_diode_anode_at_I_V(
    *,
    iv_data: IVData,
    model_parameters: ModelParameters,
) -> dict:
    """
    Computes the sum of the currents at the diode's anode in the 5-parameter
    equivalent-circuit single-diode equation (SDE).

    Parameters
    ----------
    iv_data
        I-V data
    model_parameters
        Model parameters for SDE

    Returns
    -------
    dictionary with elements
        I_sum_diode_anode_A
            Sum of currents at diode's anode node [A]
    """
    # Voltage at diode anode.
    V_diode_V = iv_data.V_V + iv_data.I_A * model_parameters["R_s_Ohm"]

    # Modified ideality factor.
    n_mod_V = model_parameters["n"] * get_scaled_thermal_voltage(
        N_s=model_parameters["N_s"], T_degC=model_parameters["T_degC"]
    )

    # Sum of currents at diode node.
    return {
        "I_sum_diode_anode_A": numpy.array(
            model_parameters["I_ph_A"]
            - model_parameters["I_rs_A"] * numpy.expm1(V_diode_V / n_mod_V)
            - model_parameters["G_p_S"] * V_diode_V
            - iv_data.I_A
        )
    }


def I_at_V(
    *,
    V_V: FloatBroadcastable,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute terminal current at specified terminal voltage.

    Parameters
    ----------
    V_V
        Terminal voltage [V]
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        I_A
            Terminal current [A]

    Compute strategy:
    1) Compute initial condition for I_A with explicit equation using R_s_Ohm==0.
    2) Compute using Halley's method via scipy.optimize.newton.
    """
    if newton_options is None:
        newton_options = NewtonOptions()

    # Ensure shapes are always full throughout computations.
    V_V, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S = numpy.broadcast_arrays(
        V_V,
        model_parameters["N_s"],
        model_parameters["T_degC"],
        model_parameters["I_ph_A"],
        model_parameters["I_rs_A"],
        model_parameters["n"],
        model_parameters["R_s_Ohm"],
        model_parameters["G_p_S"],
    )

    # Modified ideality factor.
    n_mod_V = n * get_scaled_thermal_voltage(N_s=N_s, T_degC=T_degC)

    # Use closures in function definitions for newton() call.
    def I_sum_diode_anode_at_I(I_A: FloatArray) -> FloatArray:
        return I_sum_diode_anode_at_I_V(
            iv_data=IVData(I_A=I_A, V_V=V_V),
            model_parameters=ModelParameters(
                N_s=N_s,
                T_degC=T_degC,
                I_ph_A=I_ph_A,
                I_rs_A=I_rs_A,
                n=n,
                R_s_Ohm=R_s_Ohm,
                G_p_S=G_p_S,
            ),
        )["I_sum_diode_anode_A"]

    def dI_sum_diode_anode_dI_at_I(I_A: FloatArray) -> FloatArray:
        return numpy.array(
            -I_rs_A * R_s_Ohm / n_mod_V * numpy.exp((V_V + I_A * R_s_Ohm) / n_mod_V)
            - G_p_S * R_s_Ohm
            - 1.0
        )

    def d2I_sum_diode_anode_dI2_at_I(I_A: FloatArray) -> FloatArray:
        return numpy.array(
            -I_rs_A
            * (R_s_Ohm / n_mod_V) ** 2
            * numpy.exp((V_V + I_A * R_s_Ohm) / n_mod_V)
        )

    # Compute initial condition (IC) for newton solver using zero R_s_Ohm.
    I_A_ic = numpy.array(I_ph_A - I_rs_A * numpy.expm1(V_V / n_mod_V) - G_p_S * V_V)

    # Solve for I_A using Halley's method.
    newton_result = newton(
        I_sum_diode_anode_at_I,
        I_A_ic,
        fprime=dI_sum_diode_anode_dI_at_I,
        fprime2=d2I_sum_diode_anode_dI2_at_I,
        full_output=True,
        **newton_options,
    )

    # newton_result varies dependening on shape of computaion.
    if I_A_ic.shape:
        # Non-scalar case.
        I_A = newton_result[0]
        converged = newton_result[1]
    else:
        # Scalar case.
        I_A = numpy.array(newton_result[1].root)
        converged = numpy.array(newton_result[1].converged)

    # Verify convergence, because newton() documentation says this should be checked.
    numpy.testing.assert_equal(converged, True)

    return {"I_A": I_A}


def dI_dV_at_V(
    *,
    V_V: FloatBroadcastable,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute 1st derivative of terminal current with respect to terminal voltage at
    specified terminal voltage.

    Parameters
    ----------
    V_V
        Terminal voltage [V]
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        dI_dV_S
            1st derivative of terminal current w.r.t terminal voltage [S]
        I_A
            Terminal current [A]

    This derivative is needed, e.g., for R_oc_Ohm, R_sc_Ohm, and P_mp_W calculations.
    """
    # Compute terminal current.
    result = I_at_V(
        V_V=V_V,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )

    # Modified ideality factor.
    n_mod_V = model_parameters["n"] * get_scaled_thermal_voltage(
        N_s=model_parameters["N_s"], T_degC=model_parameters["T_degC"]
    )

    # Compute first derivative of current with respect to voltage at specified voltage.
    expr1 = (
        model_parameters["I_rs_A"]
        / n_mod_V
        * numpy.exp((V_V + result["I_A"] * model_parameters["R_s_Ohm"]) / n_mod_V)
        + model_parameters["G_p_S"]
    )

    result["dI_dV_S"] = numpy.array(
        -expr1 / (1.0 + model_parameters["R_s_Ohm"] * expr1)
    )

    return result


def d2I_dV2_at_V(
    *,
    V_V: FloatBroadcastable,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute 2nd derivative of terminal current with respect to terminal voltage at
    specified terminal voltage.

    Parameters
    ----------
    V_V
        Terminal voltage [V]
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        d2I_dV2_S_per_V
            1st derivative of terminal current w.r.t terminal voltage [S/V]
        dI_dV_S
            1st derivative of terminal current w.r.t terminal voltage [S]
        I_A
            Terminal current [A]

    This derivative is needed, e.g., for P_mp_W calculations.
    """
    # Compute terminal current.
    result = I_at_V(
        V_V=V_V,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )

    # Modified ideality factor.
    n_mod_V = model_parameters["n"] * get_scaled_thermal_voltage(
        N_s=model_parameters["N_s"], T_degC=model_parameters["T_degC"]
    )

    expr0 = numpy.exp((V_V + result["I_A"] * model_parameters["R_s_Ohm"]) / n_mod_V)
    expr1 = model_parameters["I_rs_A"] / n_mod_V
    expr2 = expr1 * expr0
    expr3 = expr2 + model_parameters["G_p_S"]
    expr4 = 1.0 + model_parameters["R_s_Ohm"] * expr3
    result["dI_dV_S"] = numpy.array(-expr3 / expr4)
    result["d2I_dV2_S_per_V"] = numpy.array(
        (-expr2 / n_mod_V * (1 + result["dI_dV_S"] * model_parameters["R_s_Ohm"]))
        / expr4
    )

    return result


def V_at_I(
    *,
    I_A: FloatBroadcastable,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute terminal voltage at specified terminal current.

    Parameters
    ----------
    I_A
        Terminal current [A]
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        V_V
            Terminal voltage [V]

    Compute strategy:
    1) Compute initial condition for V_V with explicit equation using G_p_S==0.
    2) Compute using Halley's method via scipy.optimize.newton.
    """
    if newton_options is None:
        newton_options = NewtonOptions()

    # Ensure shapes are always full throughout computations.
    I_A, N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S = numpy.broadcast_arrays(
        I_A,
        model_parameters["N_s"],
        model_parameters["T_degC"],
        model_parameters["I_ph_A"],
        model_parameters["I_rs_A"],
        model_parameters["n"],
        model_parameters["R_s_Ohm"],
        model_parameters["G_p_S"],
    )

    # Modified ideality factor.
    n_mod_V = n * get_scaled_thermal_voltage(N_s=N_s, T_degC=T_degC)

    # Use closures in function definitions for newton() call.
    def I_sum_diode_anode_at_V(V_V: FloatArray) -> FloatArray:
        return I_sum_diode_anode_at_I_V(
            iv_data=IVData(I_A=I_A, V_V=V_V),
            model_parameters=ModelParameters(
                N_s=N_s,
                T_degC=T_degC,
                I_ph_A=I_ph_A,
                I_rs_A=I_rs_A,
                n=n,
                R_s_Ohm=R_s_Ohm,
                G_p_S=G_p_S,
            ),
        )["I_sum_diode_anode_A"]

    def dI_sum_diode_anode_dV_at_V(V_V: FloatArray) -> FloatArray:
        return -I_rs_A / n_mod_V * numpy.exp((V_V + I_A * R_s_Ohm) / n_mod_V) - G_p_S

    def d2I_sum_diode_anode_dV2_at_V(V_V: FloatArray) -> FloatArray:
        return -I_rs_A / n_mod_V**2 * numpy.exp((V_V + I_A * R_s_Ohm) / n_mod_V)

    # Compute initial condition (IC) for newton solver using zero G_p_S.
    V_V_ic = numpy.array(
        n_mod_V * (numpy.log(I_ph_A + I_rs_A - I_A) - numpy.log(I_rs_A)) - I_A * R_s_Ohm
    )

    # Solve for V_V using Halley's method.
    newton_result = newton(
        I_sum_diode_anode_at_V,
        V_V_ic,
        fprime=dI_sum_diode_anode_dV_at_V,
        fprime2=d2I_sum_diode_anode_dV2_at_V,
        full_output=True,
        **newton_options,
    )

    # newton_result varies dependening on shape of computaion.
    if V_V_ic.shape:
        # Non-scalar case.
        V_V = newton_result[0]
        # Second return value is convergence array.
        converged = newton_result[1]
        not_converged = numpy.logical_not(converged)

        if numpy.any(not_converged):
            # Fall back to Newton's method for indices without convergence.
            newton_result = newton(
                I_sum_diode_anode_at_V,
                V_V_ic[not_converged],
                fprime=dI_sum_diode_anode_dV_at_V,
                full_output=True,
                **newton_options,
            )
            V_V[not_converged] = newton_result[0]

            if numpy.array(newton_result[0]).shape:
                # Non-scalar case. Second return value is convergence array.
                converged[not_converged] = newton_result[1]
            else:
                # Scalar case. Second return value is RootResults object.
                converged[not_converged] = newton_result[1].converged
    else:
        # Scalar case.
        # Second return value is RootResults object with scalar converged attribute.
        if not newton_result[1].converged:
            # Fall back to Newton's method (which can only be scalar case).
            newton_result = newton(
                I_sum_diode_anode_at_V,
                V_V_ic,
                fprime=dI_sum_diode_anode_dV_at_V,
                full_output=True,
                **newton_options,
            )

        V_V = numpy.array(newton_result[0])
        converged = numpy.array(newton_result[1].converged)

    # Verify overall convergence, which newton() documentation says should be checked.
    numpy.testing.assert_equal(converged, True)

    return {"V_V": V_V}


def dV_dI_at_I(
    *,
    I_A: FloatBroadcastable,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute 1st derivative of terminal voltage with respect to terminal current at
    specified terminal current.

    Parameters
    ----------
    I_A
        Terminal current [A]
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        dV_dI_Ohm
            1st derivative of terminal voltage w.r.t terminal current [Ohm]
        V_V
            Terminal current [V]

    This derivative is needed, e.g., for solving the differential equation for capacitor
    charging.
    """
    # Compute terminal voltage.
    result = V_at_I(
        I_A=I_A,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )

    # Modified ideality factor.
    n_mod_V = model_parameters["n"] * get_scaled_thermal_voltage(
        N_s=model_parameters["N_s"], T_degC=model_parameters["T_degC"]
    )

    # Compute first derivative of voltage with respect to current at specified current.
    expr1 = (
        model_parameters["I_rs_A"]
        / n_mod_V
        * numpy.exp((result["V_V"] + I_A * model_parameters["R_s_Ohm"]) / n_mod_V)
        + model_parameters["G_p_S"]
    )
    result["dV_dI_Ohm"] = numpy.array(-1.0 / expr1 - model_parameters["R_s_Ohm"])

    return result


def P_at_V(
    *,
    V_V: FloatBroadcastable,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute terminal power at specified terminal voltage.

    Parameters
    ----------
    V_V
        Terminal voltage [V]
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        P_W
            Terminal power [W]
        I_A
            Terminal current [A]
    """
    # Compute current at voltage.
    result = I_at_V(
        V_V=V_V,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )

    # Compute power at voltage.
    result["P_W"] = numpy.array(V_V * result["I_A"])

    return result


def P_mp(
    *,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute maximum terminal power.

    Parameters
    ----------
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        P_mp_W
            Maximum terminal power [W]
        I_mp_A
            Terminal current at maximum terminal power [A]
        V_mp_V
            Terminal voltage at maximum terminal power [V]
        V_oc_V
            Terminal open-circuit voltage [V]

    Compute strategy:
      1) Compute initial condition for Vmp in bracketing interval [0, Voc].
      2) Compute maximum power using Newton's method via scipy.optimize.newton.

    """
    if newton_options is None:
        newton_options = NewtonOptions()

    # Use closures in function definitions for newton() call.
    def dP_dV(V_V):
        result = dI_dV_at_V(
            V_V=V_V,
            model_parameters=model_parameters,
            newton_options=newton_options,
        )

        return numpy.array(result["dI_dV_S"] * V_V + result["I_A"])

    def d2P_dV2(V_V):
        result = d2I_dV2_at_V(
            V_V=V_V,
            model_parameters=model_parameters,
            newton_options=newton_options,
        )

        return numpy.array(result["d2I_dV2_S_per_V"] * V_V + 2 * result["dI_dV_S"])

    # Compute Voc for assumed Vmp-bracketing interval [0, Voc].
    V_oc_V = V_at_I(
        I_A=0.0,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )["V_V"]

    V_mp_V_ic = numpy.array(3 / 4 * V_oc_V)

    # Solve for V_mp_V using Newton's method.
    newton_result = newton(
        dP_dV,
        V_mp_V_ic,
        fprime=d2P_dV2,
        full_output=True,
        **newton_options,
    )

    # newton_result varies depending on shape of computation.
    if V_mp_V_ic.shape:
        # Non-scalar case.
        V_mp_V = newton_result[0]
        converged = newton_result[1]
    else:
        # Scalar case.
        V_mp_V = numpy.array(newton_result[1].root)
        converged = numpy.array(newton_result[1].converged)

    # Verify convergence, because newton() documentation says this should be checked.
    numpy.testing.assert_equal(converged, True)

    for idx, V_mp_V_ in numpy.ndenumerate(V_mp_V):
        if not numpy.isfinite(V_mp_V_):

            def dP_dV(V_V):
                result = dI_dV_at_V(
                    V_V=V_V,
                    model_parameters=ModelParameters(
                        N_s=model_parameters["N_s"][idx],
                        T_degC=model_parameters["T_degC"][idx],
                        I_ph_A=model_parameters["I_ph_A"][idx],
                        I_rs_A=model_parameters["I_rs_A"][idx],
                        n=model_parameters["n"][idx],
                        R_s_Ohm=model_parameters["R_s_Ohm"][idx],
                        G_p_S=model_parameters["G_p_S"][idx],
                    ),
                    newton_options=newton_options,
                )

                return numpy.array(result["dI_dV_S"] * V_V + result["I_A"])

            V_mp_V[idx] = brentq(dP_dV, 0.0, V_oc_V[idx])

    I_mp_A = I_at_V(
        V_V=V_mp_V, model_parameters=model_parameters, newton_options=newton_options
    )["I_A"]
    P_mp_W = numpy.array(I_mp_A * V_mp_V)

    return {"P_mp_W": P_mp_W, "I_mp_A": I_mp_A, "V_mp_V": V_mp_V, "V_oc_V": V_oc_V}


def FF(
    *,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute fill factor (unitless fraction).

    Parameters
    ----------
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        FF
            Fill Factor [·]
        I_sc_A
            Terminal short-circuit current [A]
        I_mp_A
            Terminal current at maximum terminal power [A]
        P_mp_W
            Maximum terminal power [W]
        V_mp_V
            Terminal voltage at maximum terminal power [V]
        V_oc_V
            Terminal open-circuit voltage [V]
    """
    result = {
        "I_sc_A": I_at_V(
            V_V=0.0,
            model_parameters=model_parameters,
            newton_options=newton_options,
        )["I_A"],
        **P_mp(
            model_parameters=model_parameters,
            newton_options=newton_options,
        ),
    }

    # Compute FF.
    denominator = result["I_sc_A"] * result["V_oc_V"]
    with numpy.errstate(divide="ignore", invalid="ignore"):
        result["FF"] = numpy.where(
            denominator != 0, result["P_mp_W"] / denominator, numpy.nan
        )

    return result


def R_at_sc(
    *,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute terminal resistance at short circuit in Ohms.

    Parameters
    ----------
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        R_sc_Ohm
            Terminal resistance at short circuit [Ohm]
        I_sc_A
            Terminal short-circuit current [A]
    """
    # Compute derivative at Isc.
    result = dI_dV_at_V(
        V_V=0.0,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )

    # Compute resistance at Isc.
    return {"R_sc_Ohm": numpy.array(-1.0 / result["dI_dV_S"]), "I_sc_A": result["I_A"]}


def R_at_oc(
    *,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> dict:
    """
    Compute terminal resistance at open circuit in Ohms.

    Parameters
    ----------
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    dictionary with elements
        R_oc_Ohm
            Terminal resistance at open circuit [Ohm]
        V_oc_V
            Terminal open-circuit voltage [V]
    """
    # Compute Voc.
    V_oc_V = V_at_I(
        I_A=0.0,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )["V_V"]

    dI_dV_S = dI_dV_at_V(
        V_V=V_oc_V,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )["dI_dV_S"]

    # Compute slope at Voc.
    return {"R_oc_Ohm": numpy.array(-1.0 / dI_dV_S), "V_oc_V": V_oc_V}


def iv_curve_parameters(
    *,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> IVCurveParametersArray:
    """
    Compute I-V curve parameters.

    Parameters
    ----------
    model_parameters
        Model parameters for SDE
    newton_options
        Options for Newton solver (see scipy.optimize.newton)

    Returns
    -------
    iv_curve_parameters
        Parameters for I-V curve(s)
    """
    ff_result = FF(model_parameters=model_parameters, newton_options=newton_options)

    R_sc_Ohm = R_at_sc(
        model_parameters=model_parameters,
        newton_options=newton_options,
    )["R_sc_Ohm"]

    V_x_V = numpy.array(ff_result["V_oc_V"] / 2)
    I_x_A = I_at_V(
        V_V=V_x_V,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )["I_A"]

    V_xx_V = numpy.array((ff_result["V_mp_V"] + ff_result["V_oc_V"]) / 2)
    I_xx_A = I_at_V(
        V_V=V_xx_V,
        model_parameters=model_parameters,
        newton_options=newton_options,
    )["I_A"]

    R_oc_Ohm = R_at_oc(
        model_parameters=model_parameters,
        newton_options=newton_options,
    )["R_oc_Ohm"]

    return IVCurveParametersArray(
        I_sc_A=ff_result["I_sc_A"],
        R_sc_Ohm=R_sc_Ohm,
        V_x_V=V_x_V,
        I_x_A=I_x_A,
        I_mp_A=ff_result["I_mp_A"],
        P_mp_W=ff_result["P_mp_W"],
        V_mp_V=ff_result["V_mp_V"],
        V_xx_V=V_xx_V,
        I_xx_A=I_xx_A,
        R_oc_Ohm=R_oc_Ohm,
        V_oc_V=ff_result["V_oc_V"],
        FF=ff_result["FF"],
    )

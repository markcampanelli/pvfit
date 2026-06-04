"""
PVfit: Initial conditions (IC) for single-diode equation (SDE) inference.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from dataclasses import asdict
from typing import Optional
import warnings

import numpy

from pvfit.measurement.iv.computation import estimate_iv_curve_parameters
from pvfit.measurement.iv.types import IVCurve
from pvfit.modeling.dc.common import N_IC_MAX, N_IC_MIN, get_scaled_thermal_voltage
from pvfit.modeling.dc.single_diode.equation.simple.types import (
    ModelParametersFittable,
    ModelParametersUnfittable,
)


def determine_model_parameters_fittable_ic(
    *,
    model_parameters_unfittable: Optional[ModelParametersUnfittable] = None,
    iv_curve: Optional[IVCurve] = None,
    I_ph_A: Optional[float] = None,
    I_rs_A: Optional[float] = None,
    n: Optional[float] = None,
    R_s_Ohm: Optional[float] = None,
    G_p_S: Optional[float] = None,
) -> ModelParametersFittable:
    """
    Initial conditions for fittable model parameters for use in scalar model-parameter
    fitting. Omitted values are calculated from unfittable model parameters and I-V
    curve data.

    FIXME Inputs and outputs.
    """

    if (
        I_ph_A is None
        or I_rs_A is None
        or n is None
        or R_s_Ohm is None
        or G_p_S is None
    ) and (model_parameters_unfittable is None or iv_curve is None):
        raise ValueError(
            "must specify both model_parameters_unfittable and iv_curve if any initial "
            "condition is None and thus to be calculated from data"
        )

    iv_curve_parameters = estimate_iv_curve_parameters(iv_curve=iv_curve)

    if I_ph_A is None:
        I_ph_A = iv_curve_parameters.I_sc_A
    elif not numpy.isfinite(I_ph_A):
        raise ValueError(f"photocurrent IC is not well defined: {I_ph_A}")
    elif I_ph_A < 0:
        raise ValueError(f"photocurrent IC is less than zero: {I_ph_A}")

    if R_s_Ohm is None:
        R_s_Ohm = iv_curve_parameters.R_oc_Ohm
    elif not numpy.isfinite(R_s_Ohm):
        raise ValueError(f"series resistance IC is not well defined: {R_s_Ohm}")
    elif R_s_Ohm < 0:
        raise ValueError(f"series resistance IC is less than zero: {R_s_Ohm}")

    if G_p_S is None:
        # Assumes not dividing by zero in default value.
        G_p_S = 1 / iv_curve_parameters.R_sc_Ohm
    elif not numpy.isfinite(G_p_S):
        raise ValueError(f"parallel conductance IC is not well defined: {G_p_S}")
    elif G_p_S < 0:
        raise ValueError(f"parallel conductance IC is less than zero: {G_p_S}")

    # I_rs_A and n initial conditions determined together.
    if I_rs_A is None:
        I_rs_A = float("nan")
    elif not numpy.isfinite(I_rs_A):
        raise ValueError(f"reverse saturation current IC is not well defined: {I_rs_A}")
    elif I_rs_A <= 0:
        raise ValueError(
            f"reverse saturation current IC is less than or equal to zero: {I_rs_A}"
        )

    if n is None:
        n = float("nan")
    elif not numpy.isfinite(n):
        raise ValueError(f"ideality factor IC is not well defined: {n}")
    elif n <= 0:
        raise ValueError(f"ideality factor IC is less than or equal to zero: {n}")

    if not (numpy.isnan(I_rs_A) or numpy.isnan(n)):
        return ModelParametersFittable(
            I_ph_A=float(I_ph_A),
            I_rs_A=float(I_rs_A),
            n=float(n),
            R_s_Ohm=float(R_s_Ohm),
            G_p_S=float(G_p_S),
        )

    V_diode_mp_V = iv_curve_parameters.V_mp_V + iv_curve_parameters.I_mp_A * R_s_Ohm
    scaled_thermal_voltage_V = get_scaled_thermal_voltage(
        **asdict(model_parameters_unfittable)
    )

    if numpy.isnan(I_rs_A) and numpy.isnan(n):
        # Approximate exp(x - 1) by exp(x) at Pmp and Voc, and solve for n and
        # I_rs_A.
        n = min(
            N_IC_MAX,
            max(
                N_IC_MIN,
                (
                    (V_diode_mp_V - iv_curve_parameters.V_oc_V)
                    / (
                        scaled_thermal_voltage_V
                        * numpy.log(
                            (I_ph_A + iv_curve_parameters.I_mp_A + G_p_S * V_diode_mp_V)
                            / (I_ph_A + G_p_S * iv_curve_parameters.V_oc_V)
                        )
                    )
                ).item(),
            ),
        )

        I_rs_A = (I_ph_A + G_p_S * iv_curve_parameters.V_oc_V) * numpy.exp(
            -iv_curve_parameters.V_oc_V / (scaled_thermal_voltage_V * n)
        ).item()

        if I_rs_A <= 0 or not numpy.isfinite(I_rs_A) or n <= 0 or not numpy.isfinite(n):
            # Fall back to taking zero R_s_Ohm and G_p_S for simplified IC computation.
            warnings.warn(
                "falling back to alternative estimation of initial conditions for "
                f"I_rs_A and n: {I_rs_A} and {n}"
            )

            n = min(
                N_IC_MAX,
                max(
                    N_IC_MIN,
                    (
                        (iv_curve_parameters.V_mp_V - iv_curve_parameters.V_oc_V)
                        / scaled_thermal_voltage_V
                        / numpy.log(1 - iv_curve_parameters.I_mp_A / I_ph_A)
                    ).item(),
                ),
            )

            I_rs_A = (
                I_ph_A
                / numpy.exp(iv_curve_parameters.V_oc_V / (scaled_thermal_voltage_V * n))
            ).item()
    elif numpy.isnan(I_rs_A) and not numpy.isnan(n):
        I_rs_A = (
            (I_ph_A - G_p_S * V_diode_mp_V - iv_curve_parameters.I_mp_A)
            / numpy.expm1(V_diode_mp_V / (scaled_thermal_voltage_V * n))
        ).item()
    elif not numpy.isnan(I_rs_A) and numpy.isnan(n):
        n = min(
            N_IC_MAX,
            max(
                N_IC_MIN,
                (
                    V_diode_mp_V
                    / scaled_thermal_voltage_V
                    / numpy.log1p(
                        (I_ph_A - G_p_S * V_diode_mp_V - iv_curve_parameters.I_mp_A)
                        / I_rs_A
                    )
                ).item(),
            ),
        )

    return ModelParametersFittable(
        I_ph_A=float(I_ph_A),
        I_rs_A=float(I_rs_A),
        n=float(n),
        R_s_Ohm=float(R_s_Ohm),
        G_p_S=float(G_p_S),
    )

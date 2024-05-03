"""
PVfit: Initial conditions (IC) for single-diode equation (SDE) inference.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional
import warnings

import numpy

from pvfit.measurement.iv.types import IVCurveParametersScalar
from pvfit.modeling.dc.common import N_IC_MAX, N_IC_MIN, get_scaled_thermal_voltage
from pvfit.modeling.dc.single_diode.equation.simple.types import (
    ModelParametersFittable,
    ModelParametersFittableProvided,
    ModelParametersUnfittable,
    validate_model_parameters_fittable,
    validate_model_parameters_unfittable,
)


def estimate_model_parameters_fittable_ic(
    *,
    iv_curve_parameters: IVCurveParametersScalar,
    model_parameters_unfittable: ModelParametersUnfittable,
    model_parameters_fittable_ic_provided: Optional[
        ModelParametersFittableProvided
    ] = None,
) -> ModelParametersFittable:
    """
    Estimate initial conditions (IC) for fittable model parameters.

    Parameters
    ----------
    iv_curve_parameters
        I-V curve parameters, e.g., Isc, Pmp, Voc, etc.
    model_parameters_unfittable
        Model parameters that are are not fittable
    model_parameters_fittable_ic_provided (optional)
        Provided initial conditions (IC) for model parameters that are fittable

    Returns
    -------
    model_parameters_fittable_ic
        Complete initial conditions (IC) for model parameters that are fittable (complete)
    """
    validate_model_parameters_unfittable(
        model_parameters_unfittable=model_parameters_unfittable
    )

    scaled_thermal_voltage_V = get_scaled_thermal_voltage(**model_parameters_unfittable)

    if model_parameters_fittable_ic_provided is None:
        model_parameters_fittable_ic_provided = ModelParametersFittableProvided()

    I_ph_A_ic = model_parameters_fittable_ic_provided.get(
        "I_ph_A", iv_curve_parameters["I_sc_A"]
    )

    R_s_Ohm_ic = model_parameters_fittable_ic_provided.get(
        "R_s_Ohm", iv_curve_parameters["R_oc_Ohm"]
    )

    # Assumes not dividing by zero in default value.
    G_p_S_ic = model_parameters_fittable_ic_provided.get(
        "G_p_S", 1 / iv_curve_parameters["R_sc_Ohm"]
    )

    # I_rs_A and n initial conditions determined together.
    I_rs_A_ic = model_parameters_fittable_ic_provided.get("I_rs_A", float("nan"))
    n_ic = model_parameters_fittable_ic_provided.get("n", float("nan"))

    V_diode_mp_V = (
        iv_curve_parameters["V_mp_V"] + iv_curve_parameters["I_mp_A"] * R_s_Ohm_ic
    )

    if numpy.isnan(I_rs_A_ic) and numpy.isnan(n_ic):
        # Approximate exp(x - 1) by exp(x) at Pmp and Voc, and solve for n and
        # I_rs_A.
        n_ic = min(
            N_IC_MAX,
            max(
                N_IC_MIN,
                (
                    (V_diode_mp_V - iv_curve_parameters["V_oc_V"])
                    / (
                        scaled_thermal_voltage_V
                        * numpy.log(
                            (
                                I_ph_A_ic
                                + iv_curve_parameters["I_mp_A"]
                                + G_p_S_ic * V_diode_mp_V
                            )
                            / (I_ph_A_ic + G_p_S_ic * iv_curve_parameters["V_oc_V"])
                        )
                    )
                ).item(),
            ),
        )

        I_rs_A_ic = (I_ph_A_ic + G_p_S_ic * iv_curve_parameters["V_oc_V"]) * numpy.exp(
            -iv_curve_parameters["V_oc_V"] / (scaled_thermal_voltage_V * n_ic)
        ).item()

        if (
            I_rs_A_ic <= 0
            or not numpy.isfinite(I_rs_A_ic)
            or n_ic <= 0
            or not numpy.isfinite(n_ic)
        ):
            # Fall back to taking zero R_s_Ohm and G_p_S for simplified IC computation.
            warnings.warn(
                "falling back to alternative estimation of initial conditions for "
                f"I_rs_A and n: {I_rs_A_ic}, {n_ic}"
            )

            n_ic = min(
                N_IC_MAX,
                max(
                    N_IC_MIN,
                    (
                        (iv_curve_parameters["V_mp_V"] - iv_curve_parameters["V_oc_V"])
                        / scaled_thermal_voltage_V
                        / numpy.log(1 - iv_curve_parameters["I_mp_A"] / I_ph_A_ic)
                    ).item(),
                ),
            )

            I_rs_A_ic = (
                I_ph_A_ic
                / numpy.exp(
                    iv_curve_parameters["V_oc_V"] / (scaled_thermal_voltage_V * n_ic)
                )
            ).item()
    elif numpy.isnan(I_rs_A_ic) and not numpy.isnan(n_ic):
        I_rs_A_ic = (
            (I_ph_A_ic - G_p_S_ic * V_diode_mp_V - iv_curve_parameters["I_mp_A"])
            / numpy.expm1(V_diode_mp_V / (scaled_thermal_voltage_V * n_ic))
        ).item()
    elif not numpy.isnan(I_rs_A_ic) and numpy.isnan(n_ic):
        n_ic = min(
            N_IC_MAX,
            max(
                N_IC_MIN,
                (
                    V_diode_mp_V
                    / scaled_thermal_voltage_V
                    / numpy.log1p(
                        (
                            I_ph_A_ic
                            - G_p_S_ic * V_diode_mp_V
                            - iv_curve_parameters["I_mp_A"]
                        )
                        / I_rs_A_ic
                    )
                ).item(),
            ),
        )

    model_parameters_fittable_ic = ModelParametersFittable(
        I_ph_A=I_ph_A_ic,
        I_rs_A=I_rs_A_ic,
        n=n_ic,
        R_s_Ohm=R_s_Ohm_ic,
        G_p_S=G_p_S_ic,
    )

    # Raise if something didn't work. For example, bad user-provided value or something
    # computed as NaN.
    validate_model_parameters_fittable(
        model_parameters_fittable=model_parameters_fittable_ic
    )

    return model_parameters_fittable_ic

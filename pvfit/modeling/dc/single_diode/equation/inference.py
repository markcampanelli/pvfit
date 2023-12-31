"""
PVfit: Single-diode equation (SDE) inference.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional, Tuple
import warnings

import numpy
import scipy.odr

from pvfit.common import (
    ODR_NOT_FULL_RANK_ERROR_CODE,
    ODR_NUMERICAL_ERROR_CODE,
    ODR_SUCCESS_CODES,
)
from pvfit.measurement.iv.computation import estimate_iv_curve_parameters
from pvfit.measurement.iv.types import IVCurve, IVCurveParameters
from pvfit.modeling.dc.common import N_1_IC_MAX, N_1_IC_MIN, get_scaled_thermal_voltage
from pvfit.modeling.dc.single_diode.equation.types import (
    ModelParameters,
    ModelParametersFittable,
    ModelParametersFittableProvided,
    ModelParametersFittableFixedProvided,
    ModelParametersUnfittable,
    get_model_parameters_fittable_fixed_default,
    validate_model_parameters_fittable,
    validate_model_parameters_unfittable,
)
from pvfit.types import OdrOptions


def fit(
    *,
    iv_curve: IVCurve,
    model_parameters_unfittable: ModelParametersUnfittable,
    model_parameters_fittable_ic: Optional[ModelParametersFittableProvided] = None,
    model_parameters_fittable_fixed: Optional[
        ModelParametersFittableFixedProvided
    ] = None,
    normalize_iv_curve: bool = True,
    odr_options: Optional[OdrOptions] = None,
) -> Tuple[ModelParameters, scipy.odr.ODR]:
    """
    Use orthogonal distance regression (ODR) to fit the 5-parameter single-diode
    equation (SDE) equivalent-circuit model given current-voltage (I-V) curve data taken
    at a single irradiance and temperature.

    Parameters
    ----------
    iv_curve
        I-V curve data
    model_parameters_unfittable
        Model parameters that are are not fittable
    model_parameters_fittable_ic (optional)
        Inititial conditions (IC) for model parameters that are fittable (possibly
            incomplete, missing values are determined automatically)
    model_parameters_fittable_fixed (optional)
        Indicators for model parameters that are to remain fixed at IC value (possibly
            incomplete, missing values are not fixed)
    normalize_iv_curve (optional)
        Indicator for normalizing currents by Isc and voltaged by Voc
    odr_options (optional)
        Options for the ODR solver

    Returns
    -------
    model_parameters
        Model parameters from fit
    odr
        ODR solver result (for a transformed problem)
    """
    validate_model_parameters_unfittable(
        model_parameters_unfittable=model_parameters_unfittable,
    )

    iv_curve_parameters = estimate_iv_curve_parameters(iv_curve=iv_curve)

    model_parameters_fittable_ic_ = estimate_model_parameters_fittable_ic(
        iv_curve_parameters=iv_curve_parameters,
        model_parameters_unfittable=model_parameters_unfittable,
        model_parameters_fittable_ic=model_parameters_fittable_ic,
    )

    # Check for provided fit parameters to be fixed, and assign default if None.
    model_parameters_fittable_fixed_ = get_model_parameters_fittable_fixed_default()
    if model_parameters_fittable_fixed is not None:
        model_parameters_fittable_fixed_.update(model_parameters_fittable_fixed)

    # Check for provided odr parameters, and assign default if None.
    odr_options_ = OdrOptions(maxit=1000)
    if odr_options is not None:
        odr_options_.update(odr_options)

    if normalize_iv_curve:
        V_V_scale = iv_curve_parameters["V_oc_V"]
        I_A_scale = iv_curve_parameters["I_sc_A"]
    else:
        V_V_scale = 1.0
        I_A_scale = 1.0

    data = scipy.odr.Data(
        numpy.vstack((iv_curve.V_V / V_V_scale, iv_curve.I_A / I_A_scale)), 1
    )

    scaled_thermal_voltage = (
        get_scaled_thermal_voltage(**model_parameters_unfittable) / V_V_scale
    )

    def current_sum_at_diode_node(beta, x):
        """The scaled SDE model to fit."""
        I_ph = beta[0]
        I_rs_1 = numpy.exp(beta[1])
        n_1 = beta[2]
        R_s = beta[3]
        G_p = beta[4]

        V = x[0]
        I = x[1]

        V_diode = V + I * R_s

        return (
            I_ph
            - I_rs_1 * numpy.expm1(V_diode / (n_1 * scaled_thermal_voltage))
            - G_p * V_diode
            - I
        )

    model = scipy.odr.Model(current_sum_at_diode_node, implicit=True)

    beta0 = numpy.array(
        [
            model_parameters_fittable_ic_["I_ph_A"] / I_A_scale,
            numpy.log(model_parameters_fittable_ic_["I_rs_1_A"] / I_A_scale),
            model_parameters_fittable_ic_["n_1"],
            model_parameters_fittable_ic_["R_s_Ohm"] * I_A_scale / V_V_scale,
            model_parameters_fittable_ic_["G_p_S"] * V_V_scale / I_A_scale,
        ]
    )

    ifixb = [
        int(model_parameters_fittable_fixed_[key] is False)
        for key in ("I_ph_A", "I_rs_1_A", "n_1", "R_s_Ohm", "G_p_S")
    ]

    recompute = True
    while recompute:
        # Do not allow negative R_s_Ohm or G_p_S by recomputing fit, if necessary.
        # Uncertain if this is significantly different from an ODR solver that permits
        # parameter bounds.
        recompute = False

        # By construction, this loop must stop after at most two recomputes, because
        # once a negative fit parameter is fixed to zero, it must stay fixed at zero.
        odr = scipy.odr.ODR(data, model, beta0=beta0, ifixb=ifixb, **odr_options_)
        output = odr.run()

        odr_code = str(output.info)
        if odr_code not in ODR_SUCCESS_CODES:
            # ODR occassionally returns a numerical error after apparent convergence.
            if (
                len(odr_code) == 5
                and odr_code[0] == ODR_NUMERICAL_ERROR_CODE
                and odr_code[-1] in ODR_SUCCESS_CODES
            ):
                warnings.warn(
                    "ODR solver reported a numerical error despite apparent "
                    f"convergence, {odr_code}: {output.stopreason}"
                )
            elif (
                len(odr_code) == 2
                and odr_code[-2] == ODR_NOT_FULL_RANK_ERROR_CODE
                and odr_code[-1] in ODR_SUCCESS_CODES
            ):
                warnings.warn(
                    f"ODR solver reported questionable results, {odr_code}: "
                    f"{output.stopreason}"
                )
            else:
                raise RuntimeError(
                    f"ODR solver failed to converge to solution, {odr_code}: "
                    f"{output.stopreason}"
                )

        if output.beta[3] < 0:
            # R_s_Ohm was negative.
            ifixb[3] = 0
            beta0[3] = 0.0
            recompute = True

        if output.beta[4] < 0:
            # G_p_S was negative.
            ifixb[4] = 0
            beta0[4] = 0.0
            recompute = True

    model_parameters_fittable_fit = ModelParametersFittable(
        I_ph_A=output.beta[0] * I_A_scale,
        I_rs_1_A=numpy.exp(output.beta[1]) * I_A_scale,
        n_1=output.beta[2],
        R_s_Ohm=output.beta[3] * V_V_scale / I_A_scale,
        G_p_S=output.beta[4] * I_A_scale / V_V_scale,
    )

    validate_model_parameters_fittable(
        model_parameters_fittable=model_parameters_fittable_fit,
    )

    return (
        ModelParameters(**model_parameters_unfittable, **model_parameters_fittable_fit),
        odr,
    )


def estimate_model_parameters_fittable_ic(
    *,
    iv_curve_parameters: IVCurveParameters,
    model_parameters_unfittable: ModelParametersUnfittable,
    model_parameters_fittable_ic: Optional[ModelParametersFittableProvided] = None,
) -> ModelParametersFittable:
    """
    Estimate initial conditions (IC) for fittable model parameters.

    Parameters
    ----------
    iv_curve_parameters
        I-V curve parameters, e.g., Isc, Pmp, Voc, etc.
    model_parameters_unfittable
        Model parameters that are are not fittable
    model_parameters_fittable_ic (optional)
        Initial conditions (IC) for model parameters that are fittable (possibly
            incomplete)

    Returns
    -------
    model_parameters_fittable_ic
        Initial conditions (IC) for model parameters that are fittable (complete)
    """
    validate_model_parameters_unfittable(
        model_parameters_unfittable=model_parameters_unfittable
    )

    scaled_thermal_voltage_V = get_scaled_thermal_voltage(**model_parameters_unfittable)

    if model_parameters_fittable_ic is None:
        model_parameters_fittable_ic = ModelParametersFittableProvided()

    I_ph_A_ic = model_parameters_fittable_ic.get(
        "I_ph_A", iv_curve_parameters["I_sc_A"]
    )

    R_s_Ohm_ic = model_parameters_fittable_ic.get(
        "R_s_Ohm", iv_curve_parameters["R_oc_Ohm"]
    )

    # Assumes not dividing by zero in default value.
    G_p_S_ic = model_parameters_fittable_ic.get(
        "G_p_S", 1 / iv_curve_parameters["R_sc_Ohm"]
    )

    # I_rs_1_A and n_1 initial conditions determined together.
    I_rs_1_A_ic = model_parameters_fittable_ic.get("I_rs_1_A", float("nan"))
    n_1_ic = model_parameters_fittable_ic.get("n_1", float("nan"))

    V_diode_mp_V = (
        iv_curve_parameters["V_mp_V"] + iv_curve_parameters["I_mp_A"] * R_s_Ohm_ic
    )

    if numpy.isnan(I_rs_1_A_ic) and numpy.isnan(n_1_ic):
        # Approximate exp(x - 1) by exp(x) at Pmp and Voc, and solve for n_1 and
        # I_rs_1_A.
        n_1_ic = min(
            N_1_IC_MAX,
            max(
                N_1_IC_MIN,
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

        I_rs_1_A_ic = (
            I_ph_A_ic + G_p_S_ic * iv_curve_parameters["V_oc_V"]
        ) * numpy.exp(
            -iv_curve_parameters["V_oc_V"] / (scaled_thermal_voltage_V * n_1_ic)
        ).item()

        if (
            I_rs_1_A_ic <= 0
            or not numpy.isfinite(I_rs_1_A_ic)
            or n_1_ic <= 0
            or not numpy.isfinite(n_1_ic)
        ):
            # Fall back to taking zero R_s_Ohm and G_p_S for simplified IC computation.
            warnings.warn(
                "falling back to alternative estimation of initial conditions for "
                f"I_rs_1_A and n_1: {I_rs_1_A_ic}, {n_1_ic}"
            )

            n_1_ic = min(
                N_1_IC_MAX,
                max(
                    N_1_IC_MIN,
                    (
                        (iv_curve_parameters["V_mp_V"] - iv_curve_parameters["V_oc_V"])
                        / scaled_thermal_voltage_V
                        / numpy.log(1 - iv_curve_parameters["I_mp_A"] / I_ph_A_ic)
                    ).item(),
                ),
            )

            I_rs_1_A_ic = (
                I_ph_A_ic
                / numpy.exp(
                    iv_curve_parameters["V_oc_V"] / (scaled_thermal_voltage_V * n_1_ic)
                )
            ).item()
    elif numpy.isnan(I_rs_1_A_ic) and not numpy.isnan(n_1_ic):
        I_rs_1_A_ic = (
            (I_ph_A_ic - G_p_S_ic * V_diode_mp_V - iv_curve_parameters["I_mp_A"])
            / numpy.expm1(V_diode_mp_V / (scaled_thermal_voltage_V * n_1_ic))
        ).item()
    elif not numpy.isnan(I_rs_1_A_ic) and numpy.isnan(n_1_ic):
        n_1_ic = min(
            N_1_IC_MAX,
            max(
                N_1_IC_MIN,
                (
                    V_diode_mp_V
                    / scaled_thermal_voltage_V
                    / numpy.log1p(
                        (
                            I_ph_A_ic
                            - G_p_S_ic * V_diode_mp_V
                            - iv_curve_parameters["I_mp_A"]
                        )
                        / I_rs_1_A_ic
                    )
                ).item(),
            ),
        )

    model_parameters_fittable_ic_ = ModelParametersFittable(
        I_ph_A=I_ph_A_ic,
        I_rs_1_A=I_rs_1_A_ic,
        n_1=n_1_ic,
        R_s_Ohm=R_s_Ohm_ic,
        G_p_S=G_p_S_ic,
    )

    # Raise if something didn't work. For example, bad user-provided value or something
    # computed as NaN.
    validate_model_parameters_fittable(
        model_parameters_fittable=model_parameters_fittable_ic_
    )

    return model_parameters_fittable_ic_

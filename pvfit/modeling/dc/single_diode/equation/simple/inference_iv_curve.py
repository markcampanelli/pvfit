"""
PVfit: Single-diode equation (SDE) inference.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional
import warnings

import numpy
import scipy.odr

from pvfit.common import (
    ODR_NOT_FULL_RANK_ERROR_CODE,
    ODR_NUMERICAL_ERROR_CODE,
    ODR_SUCCESS_CODES,
)
from pvfit.measurement.iv.computation import estimate_iv_curve_parameters
from pvfit.measurement.iv.types import IVCurve
from pvfit.modeling.dc.common import get_scaled_thermal_voltage
from pvfit.modeling.dc.single_diode.equation.simple.inference_ic import (
    estimate_model_parameters_fittable_ic,
)
import pvfit.modeling.dc.single_diode.equation.simple.types as types
from pvfit.types import OdrOptions


def fit(
    *,
    iv_curve: IVCurve,
    model_parameters_unfittable: types.ModelParametersUnfittable,
    model_parameters_fittable_ic_provided: Optional[
        types.ModelParametersFittableProvided
    ] = None,
    model_parameters_fittable_fixed_provided: Optional[
        types.ModelParametersFittableFixedProvided
    ] = None,
    normalize_iv_curve: bool = True,
    odr_options: Optional[OdrOptions] = None,
) -> types.FitResultODR:
    """
    Use orthogonal distance regression (ODR) to fit the implicit 5-parameter
    equivalent-circuit single-diode equation (SDE) given current-voltage (I-V) curve
    data taken at a single effective-irradiance ratio and cell temperatures.

    Parameters
    ----------
    iv_curve
        I-V curve data
    model_parameters_unfittable
        Model parameters that are are not fittable
    model_parameters_fittable_ic_provided (optional)
        Inititial conditions (IC) for model parameters that are fittable (possibly
            incomplete, missing values are determined automatically)
    model_parameters_fittable_fixed_provided (optional)
        Indicators for model parameters that are to remain fixed at IC value (possibly
            incomplete, missing values are not fixed)
    normalize_iv_curve (optional)
        Indicator for normalizing currents by Isc and voltages by Voc
    odr_options (optional)
        Options for the ODR solver

    Returns
    -------
    fit_result
        Collected results of the fit
            model_parameters
                Model parameters from fit
            model_parameters_fittable_ic
                Model parameters from fit's initial-condition (IC) calculation
            odr
                ODR object, with solver result (for a transformed problem)
    """
    types.validate_model_parameters_unfittable(
        model_parameters_unfittable=model_parameters_unfittable,
    )

    iv_curve_parameters = estimate_iv_curve_parameters(iv_curve=iv_curve)

    model_parameters_fittable_ic = estimate_model_parameters_fittable_ic(
        iv_curve_parameters=iv_curve_parameters,
        model_parameters_unfittable=model_parameters_unfittable,
        model_parameters_fittable_ic_provided=model_parameters_fittable_ic_provided,
    )

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

    def I_sum_diode_anode(beta, x):
        """The scaled SDE model to fit."""
        I_ph = beta[0]
        I_rs = numpy.exp(beta[1])
        n = beta[2]
        R_s = beta[3]
        G_p = beta[4]

        V = x[0]
        I = x[1]

        V_diode = V + I * R_s

        return (
            I_ph
            - I_rs * numpy.expm1(V_diode / (n * scaled_thermal_voltage))
            - G_p * V_diode
            - I
        )

    model = scipy.odr.Model(I_sum_diode_anode, implicit=True)

    beta0 = numpy.array(
        [
            model_parameters_fittable_ic["I_ph_A"] / I_A_scale,
            numpy.log(model_parameters_fittable_ic["I_rs_A"] / I_A_scale),
            model_parameters_fittable_ic["n"],
            model_parameters_fittable_ic["R_s_Ohm"] * I_A_scale / V_V_scale,
            model_parameters_fittable_ic["G_p_S"] * V_V_scale / I_A_scale,
        ]
    )

    # Check for provided fit parameters to be fixed, and assign default if None.
    model_parameters_fittable_fixed = (
        types.get_model_parameters_fittable_fixed_default()
    )
    if model_parameters_fittable_fixed_provided is not None:
        model_parameters_fittable_fixed.update(model_parameters_fittable_fixed_provided)

    ifixb = [
        int(model_parameters_fittable_fixed[key] is False)
        for key in ("I_ph_A", "I_rs_A", "n", "R_s_Ohm", "G_p_S")
    ]

    # Check for provided odr parameters, and assign default if None.
    odr_options_ = OdrOptions(maxit=1000)
    if odr_options is not None:
        odr_options_.update(odr_options)

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
                    "odr solver reported a numerical error despite apparent "
                    f"convergence, {odr_code}: {output.stopreason}"
                )
            elif (
                len(odr_code) == 2
                and odr_code[-2] == ODR_NOT_FULL_RANK_ERROR_CODE
                and odr_code[-1] in ODR_SUCCESS_CODES
            ):
                warnings.warn(
                    f"odr solver reported questionable results, {odr_code}: "
                    f"{output.stopreason}"
                )
            else:
                raise RuntimeError(
                    f"odr solver failed to converge to a solution, {odr_code}: "
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

    model_parameters_fittable = types.ModelParametersFittable(
        I_ph_A=output.beta[0] * I_A_scale,
        I_rs_A=numpy.exp(output.beta[1]) * I_A_scale,
        n=output.beta[2],
        R_s_Ohm=output.beta[3] * V_V_scale / I_A_scale,
        G_p_S=output.beta[4] * I_A_scale / V_V_scale,
    )

    types.validate_model_parameters_fittable(
        model_parameters_fittable=model_parameters_fittable,
    )

    return types.FitResultODR(
        model_parameters_ic=types.ModelParameters(
            **model_parameters_unfittable, **model_parameters_fittable_ic
        ),
        model_parameters=types.ModelParameters(
            **model_parameters_unfittable, **model_parameters_fittable
        ),
        odr_output=output,
    )

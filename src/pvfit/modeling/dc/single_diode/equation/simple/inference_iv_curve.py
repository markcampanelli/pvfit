"""
PVfit: Single-diode equation (SDE) inference.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from dataclasses import asdict
from typing import Optional

import numpy
import odrpack

from pvfit.measurement.iv.computation import estimate_iv_curve_parameters
from pvfit.measurement.iv.types import IVCurve
from pvfit.modeling.dc.common import get_scaled_thermal_voltage
from pvfit.modeling.dc.single_diode.equation.simple.initial_conditions import (
    determine_model_parameters_fittable_ic,
)
from pvfit.modeling.dc.single_diode.equation.simple.types import (
    FitResultODR,
    ModelParameters,
    ModelParametersFittable,
    ModelParametersFittableFixed,
    ModelParametersUnfittable,
)
from pvfit.types import OdrOptions


def fit(
    *,
    iv_curve: IVCurve,
    model_parameters_unfittable: ModelParametersUnfittable,
    model_parameters_fittable_ic: Optional[ModelParametersFittable] = None,
    model_parameters_fittable_fixed: Optional[ModelParametersFittableFixed] = None,
    normalize_iv_curve: bool = True,
    odr_options: Optional[OdrOptions] = None,
) -> FitResultODR:
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
    model_parameters_fittable_ic (optional)
        Inititial conditions (IC) for model parameters that are fittable
    model_parameters_fittable_fixed (optional)
        Indicators for model parameters that are to remain fixed at IC value
    normalize_iv_curve (optional)
        Indicator for normalizing currents by Isc and voltages by Voc
    odr_options (optional)
        Options for the ODR solver

    Returns
    -------
    dictionary with the following
        model_parameters_ic
            Model parameters from fit's initial-condition (IC) calculation
        model_parameters
            Model parameters from fit
        solver_result
            odrpack.OdrResult object, with solver result (for a transformed problem)
    """

    if model_parameters_fittable_ic is None:
        model_parameters_fittable_ic = determine_model_parameters_fittable_ic(
            model_parameters_unfittable=model_parameters_unfittable,
            iv_curve=iv_curve,
        )

    if model_parameters_fittable_fixed is None:
        model_parameters_fittable_fixed = ModelParametersFittableFixed()

    if normalize_iv_curve:
        iv_curve_parameters = estimate_iv_curve_parameters(iv_curve=iv_curve)
        V_V_scale = iv_curve_parameters.V_oc_V
        I_A_scale = iv_curve_parameters.I_sc_A
    else:
        V_V_scale = 1.0
        I_A_scale = 1.0

    scaled_thermal_voltage = (
        get_scaled_thermal_voltage(**asdict(model_parameters_unfittable)) / V_V_scale
    )

    def f(x: numpy.ndarray, beta: numpy.ndarray) -> numpy.ndarray:
        """
        The scaled SDE model to fit. The sum of currents at the diode's anode node.

        Note closure over one/more fixed variables.
        """
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

    xdata = numpy.vstack((iv_curve.V_V / V_V_scale, iv_curve.I_A / I_A_scale))
    ydata = numpy.zeros(xdata.shape[-1])
    beta0 = numpy.array(
        [
            model_parameters_fittable_ic.I_ph_A / I_A_scale,
            numpy.log(model_parameters_fittable_ic.I_rs_A / I_A_scale),
            model_parameters_fittable_ic.n,
            model_parameters_fittable_ic.R_s_Ohm * I_A_scale / V_V_scale,
            model_parameters_fittable_ic.G_p_S * V_V_scale / I_A_scale,
        ]
    )
    bounds = (numpy.zeros_like(beta0), numpy.full_like(beta0, numpy.inf))
    bounds[0][1] = -numpy.inf  # Log of reverse saturation current.
    fix_beta = [
        asdict(model_parameters_fittable_fixed)[key]
        for key in asdict(model_parameters_fittable_ic).keys()
    ]

    if odr_options is None:
        odr_options = OdrOptions()

    print(
        xdata,
        ydata,
        beta0,
        bounds,
        fix_beta,
    )

    solver_result = odrpack.odr_fit(
        f,
        xdata,
        ydata,
        beta0,
        bounds=bounds,
        task="implicit-ODR",
        fix_beta=fix_beta,
        **asdict(odr_options),
    )

    if not solver_result.success:
        raise RuntimeError(
            f"odr solver returned an error, {solver_result.info}: "
            f"{solver_result.stopreason}"
        )

    return FitResultODR(
        model_parameters_ic=ModelParameters(
            **asdict(model_parameters_unfittable),
            **asdict(model_parameters_fittable_ic),
        ),
        model_parameters=ModelParameters(
            **asdict(model_parameters_unfittable),
            I_ph_A=float(solver_result.beta[0] * I_A_scale),
            I_rs_A=float(numpy.exp(solver_result.beta[1]) * I_A_scale),
            n=float(solver_result.beta[2]),
            R_s_Ohm=float(solver_result.beta[3] * V_V_scale / I_A_scale),
            G_p_S=float(solver_result.beta[4] * I_A_scale / V_V_scale),
        ),
        solver_result=solver_result,
    )

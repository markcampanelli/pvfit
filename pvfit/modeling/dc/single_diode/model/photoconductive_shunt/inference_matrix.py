"""
PVfit: Calibrate single-diode model (SDM) with photoconductive shunt from IEC 61853-1
matrix data (or similar) using orthogonal distance regression (ODR).

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional, Tuple, TypedDict
import warnings

import numpy
from scipy.constants import convert_temperature
import scipy.odr


from pvfit.common import (
    ODR_NOT_FULL_RANK_ERROR_CODE,
    ODR_NUMERICAL_ERROR_CODE,
    ODR_SUCCESS_CODES,
)
from pvfit.common import k_B_J_per_K, k_B_eV_per_K, q_C
from pvfit.measurement.iv.types import (
    FTData,
    IVCurveParametersArray,
    IVPerformanceMatrix,
)
import pvfit.modeling.dc.single_diode.equation.simple.simulation as sde_sim
import pvfit.modeling.dc.single_diode.model.photoconductive_shunt.auxiliary_equations as sdm_ps_ae
from pvfit.modeling.dc.single_diode.model.simple.inference_ic import (
    estimate_model_parameters_fittable_ic,
)
from pvfit.modeling.dc.single_diode.model.simple import types
from pvfit.types import OdrOptions


class ModelParametersFittable(TypedDict):
    """Fittable model parameters."""

    I_rs_A_0: bool
    n_0: bool
    R_s_Ohm_0: bool
    G_p_S_0: bool
    E_g_eV_0: bool


class ModelParametersFittableProvided(TypedDict, total=False):
    """
    Optionally provided fittable model parameters, e.g., for initial conditions (IC).
    """

    I_rs_A_0: float
    n_0: float
    R_s_Ohm_0: float
    G_p_S_0: float
    E_g_eV_0: float


class ModelParametersFittableFixed(TypedDict):
    """Fittable model parameters to be fixed for parameter fits."""

    I_rs_A_0: bool
    n_0: bool
    R_s_Ohm_0: bool
    G_p_S_0: bool
    E_g_eV_0: bool


def get_model_parameters_fittable_fixed_default() -> ModelParametersFittableFixed:
    """Get default ModelParametersFittableFixed (no parameter fixing)."""

    return ModelParametersFittableFixed(
        I_rs_A_0=False,
        n_0=False,
        R_s_Ohm_0=False,
        G_p_S_0=False,
        E_g_eV_0=False,
    )


class ModelParametersFittableFixedProvided(TypedDict, total=False):
    """Optionally provided fittable model parameters to be fixed for parameter fits."""

    I_rs_A_0: bool
    n_0: bool
    R_s_Ohm_0: bool
    G_p_S_0: bool
    E_g_eV_0: bool


def fun(beta, x, N_s, T_K_0, I_sc_A_0):
    """
    Implicit system of SDM-derived equations over which model parameters are optimized.
    """
    I_rs_A_0 = numpy.exp(beta[0])
    n_0 = beta[1]
    R_s_Ohm_0 = beta[2]
    G_p_S_0 = beta[3]
    E_g_eV_0 = beta[4]

    I_sc_A = x[0, :]
    I_mp_A = x[1, :]
    V_mp_V = x[2, :]
    V_oc_V = x[3, :]
    T_K = x[4, :]

    F = I_sc_A_0 / I_sc_A

    scaled_thermal_voltage_V = (N_s * n_0 * k_B_J_per_K * T_K) / q_C

    # Parallel conductance with photoconductive shunt.
    G_p_S = F * G_p_S_0

    # Reverse-saturation current.
    I_rs_A = (
        I_rs_A_0
        * (T_K / T_K_0) ** 3
        * numpy.exp(E_g_eV_0 / (n_0 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))
    )

    # Photocurrent from short-circuit point.
    V_diode_sc_V = I_sc_A * R_s_Ohm_0
    I_ph_A = (
        I_rs_A * numpy.expm1(V_diode_sc_V / scaled_thermal_voltage_V)
        + G_p_S * V_diode_sc_V
        + I_sc_A
    )

    # Maximum-power point.
    V_diode_mp_V = V_mp_V + I_mp_A * R_s_Ohm_0
    y0 = (
        I_ph_A
        - I_rs_A * numpy.expm1(V_diode_mp_V / scaled_thermal_voltage_V)
        - G_p_S * V_diode_mp_V
        - I_mp_A
    )

    # Maximum attained at maximum-power point.
    y1 = (
        (I_mp_A * R_s_Ohm_0 - V_mp_V)
        * (
            I_rs_A
            / scaled_thermal_voltage_V
            * numpy.exp(V_diode_mp_V / scaled_thermal_voltage_V)
            + G_p_S
        )
        + I_mp_A
    ) / (
        I_rs_A
        / scaled_thermal_voltage_V
        * numpy.exp(V_diode_mp_V / scaled_thermal_voltage_V)
        + G_p_S * R_s_Ohm_0
        + 1
    )

    # Open-circuit point.
    V_diode_oc_V = V_oc_V
    y2 = (
        I_ph_A
        - I_rs_A * numpy.expm1(V_diode_oc_V / scaled_thermal_voltage_V)
        - G_p_S * V_diode_oc_V
    )

    return numpy.vstack((y0, y1, y2))


def fit(
    *,
    iv_performance_matrix: IVPerformanceMatrix,
    model_parameters_fittable_ic_provided: Optional[
        ModelParametersFittableProvided
    ] = None,
    model_parameters_fittable_fixed_provided: Optional[
        ModelParametersFittableFixedProvided
    ] = None,
    normalize_iv_curves: bool = True,
    odr_options: Optional[OdrOptions] = None,
) -> types.FitResultODR:
    """
    Use orthogonal distance regression (ODR) to fit the implicit 6-parameter
    equivalent-circuit single-diode model (SDM) given current-voltage (I-V) curve
    data taken over a range of effective-irradiance ratio and cell temperature (F-T)
    operating conditions.

    Parameters
    ----------
    iv_performance_matrix
        I-V performance matrix data
    model_parameters_fittable_ic_provided (optional)
        Inititial conditions (IC) for model parameters that are fittable (possibly
            incomplete, missing values are determined automatically)
    model_parameters_fittable_fixed_provided (optional)
        Indicators for model parameters that are to remain fixed at IC value (possibly
            incomplete, missing values are not fixed)
    normalize_iv_curves (optional)
        Indicator for normalizing currents by Isc and voltages by Voc
    odr_options (optional)
        Options for the ODR solver

    Returns
    -------
    dictionary with the following
        model_parameters
            Model parameters from fit
        model_parameters_fittable_ic
            Model parameters from fit's initial-condition (IC) calculation
        odr
            ODR object, with solver result (for a transformed problem)
    """
    model_parameters_unfittable = types.ModelParametersUnfittable(
        N_s=iv_performance_matrix.N_s,
        T_degC_0=iv_performance_matrix.T_degC_0,
    )
    types.validate_model_parameters_unfittable(
        model_parameters_unfittable=model_parameters_unfittable,
    )
    N_s = model_parameters_unfittable["N_s"]
    T_degC_0 = model_parameters_unfittable["T_degC_0"]
    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")

    if model_parameters_fittable_ic_provided is None:
        model_parameters_fittable_ic_provided = ModelParametersFittableProvided()

    model_parameters_fittable_ic = estimate_model_parameters_fittable_ic(
        ivft_data=iv_performance_matrix.ivft_data,
        model_parameters_unfittable=model_parameters_unfittable,
        model_parameters_fittable_ic_provided=types.ModelParametersFittableProvided(
            I_sc_A_0=iv_performance_matrix.I_sc_A_0,
            **model_parameters_fittable_ic_provided,
        ),
        material=iv_performance_matrix.material,
    )

    # FIXME Implement data scaling?
    if normalize_iv_curves:
        V_V_scale = iv_performance_matrix.V_oc_V_0
        I_A_scale = iv_performance_matrix.I_sc_A_0
        T_K_scale = iv_performance_matrix.T_K_0
    else:
        V_V_scale = 1.0
        I_A_scale = 1.0
        T_K_scale = 1.0

    data = scipy.odr.Data(
        numpy.vstack(
            (
                iv_performance_matrix.I_sc_A,
                iv_performance_matrix.I_mp_A,
                iv_performance_matrix.V_mp_V,
                iv_performance_matrix.V_oc_V,
                iv_performance_matrix.T_K,
            )
        ),
        3,
    )

    model = scipy.odr.Model(
        fun, implicit=True, extra_args=(N_s, T_K_0, iv_performance_matrix.I_sc_A_0)
    )

    beta0 = numpy.array(
        [
            numpy.log(model_parameters_fittable_ic["I_rs_A_0"]),
            model_parameters_fittable_ic["n_0"],
            model_parameters_fittable_ic["R_s_Ohm_0"],
            model_parameters_fittable_ic["G_p_S_0"],
            model_parameters_fittable_ic["E_g_eV_0"],
        ]
    )

    # Check for provided fit parameters to be fixed, and assign default if None.
    model_parameters_fittable_fixed = get_model_parameters_fittable_fixed_default()
    if model_parameters_fittable_fixed_provided is not None:
        model_parameters_fittable_fixed.update(model_parameters_fittable_fixed_provided)

    ifixb = [
        int(model_parameters_fittable_fixed[key] is False)
        for key in ("I_rs_A_0", "n_0", "R_s_Ohm_0", "G_p_S_0", "E_g_eV_0")
    ]

    # Check for provided odr parameters, and assign default if None.
    odr_options_ = OdrOptions(maxit=1000)
    if odr_options is not None:
        odr_options_.update(odr_options)

    recompute = True
    while recompute:
        # Do not allow negative R_s_Ohm_0 or G_p_S_0 by recomputing fit, if necessary.
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
            # R_s_Ohm_0 was negative.
            ifixb[3] = 0
            beta0[3] = 0.0
            recompute = True

        if output.beta[4] < 0:
            # G_p_S_0 was negative.
            ifixb[4] = 0
            beta0[4] = 0.0
            recompute = True

    # Transform back fit values.
    model_parameters_fittable = types.ModelParametersFittable(
        I_sc_A_0=iv_performance_matrix.I_sc_A_0,
        I_rs_A_0=numpy.exp(output.beta[0]),
        n_0=output.beta[1],
        R_s_Ohm_0=output.beta[2],
        G_p_S_0=output.beta[3],
        E_g_eV_0=output.beta[4],
    )

    # Raise if something didn't work. For example, bad user-provided value or something
    # computed as NaN.
    types.validate_model_parameters_fittable(
        model_parameters_fittable=model_parameters_fittable
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


def compute_fit_quality(
    iv_performance_matrix: IVPerformanceMatrix,
    model_parameters: types.ModelParameters,
) -> Tuple[dict, IVCurveParametersArray]:
    """Compute FIXME"""

    iv_curve_parameters = sde_sim.iv_curve_parameters(
        model_parameters=sdm_ps_ae.compute_sde_model_parameters(
            ft_data=FTData(
                F=iv_performance_matrix.F, T_degC=iv_performance_matrix.T_degC
            ),
            model_parameters=model_parameters,
        )
    )

    I_sc_pc_error = 100 * (
        iv_curve_parameters["I_sc_A"] / iv_performance_matrix.I_sc_A - 1
    )
    I_mp_pc_error = 100 * (
        iv_curve_parameters["I_mp_A"] / iv_performance_matrix.I_mp_A - 1
    )
    P_mp_pc_error = 100 * (
        iv_curve_parameters["P_mp_W"] / iv_performance_matrix.P_mp_W - 1
    )
    V_mp_pc_error = 100 * (
        iv_curve_parameters["V_mp_V"] / iv_performance_matrix.V_mp_V - 1
    )
    V_oc_pc_error = 100 * (
        iv_curve_parameters["V_oc_V"] / iv_performance_matrix.V_oc_V - 1
    )

    return {
        "mape": {
            "I_sc_A": numpy.mean(numpy.abs(I_sc_pc_error)),
            "I_mp_A": numpy.mean(numpy.abs(I_mp_pc_error)),
            "P_mp_W": numpy.mean(numpy.abs(P_mp_pc_error)),
            "V_mp_V": numpy.mean(numpy.abs(V_mp_pc_error)),
            "V_oc_V": numpy.mean(numpy.abs(V_oc_pc_error)),
        },
        "mbpe": {
            "I_sc_A": numpy.mean(I_sc_pc_error),
            "I_mp_A": numpy.mean(I_mp_pc_error),
            "P_mp_W": numpy.mean(P_mp_pc_error),
            "V_mp_V": numpy.mean(V_mp_pc_error),
            "V_oc_V": numpy.mean(V_oc_pc_error),
        },
    }, iv_curve_parameters

"""
PVfit: Calibrate single-diode model (SDM) with photoconductive shunt from IEC 61853-1
matrix data (or similar) using orthogonal distance regression (ODR).

Copyright 2023 Intelligent Measurement Systems LLC
"""

from dataclasses import asdict
from typing import Optional, Tuple

import numpy
import odrpack
from scipy.constants import convert_temperature

from pvfit.common import k_B_J_per_K, k_B_eV_per_K, q_C
from pvfit.measurement.iv.types import (
    FTData,
    IVCurveParametersArray,
    IVPerformanceMatrix,
)
import pvfit.modeling.dc.single_diode.equation.simple.simulation as sde_sim
import pvfit.modeling.dc.single_diode.model.photoconductive_shunt.auxiliary_equations as sdm_ps_ae
from pvfit.modeling.dc.single_diode.model.simple.initial_conditions import (
    determine_model_parameters_fittable_ic,
)
from pvfit.modeling.dc.single_diode.model.simple.types import (
    ModelParametersFitResultODR,
    ModelParameters,
    ModelParametersFittable,
    ModelParametersFittableFixed,
    ModelParametersUnfittable,
)
from pvfit.types import NewtonOptions, OdrOptions


def fit(
    *,
    iv_performance_matrix: IVPerformanceMatrix,
    model_parameters_fittable_ic: Optional[ModelParametersFittable] = None,
    model_parameters_fittable_fixed: Optional[ModelParametersFittableFixed] = None,
    normalize_iv_performance_matrix: bool = True,
    odr_options: Optional[OdrOptions] = None,
) -> ModelParametersFitResultODR:
    """
    Use orthogonal distance regression (ODR) to fit the implicit 6-parameter
    equivalent-circuit single-diode model (SDM) given current-voltage (I-V) curve
    data taken over a range of effective-irradiance ratio and cell temperature (F-T)
    operating conditions. Parallel shunt conductance is photoconductive.

    Parameters
    ----------
    iv_performance_matrix
        I-V performance matrix data
    model_parameters_fittable_ic (optional)
        Inititial conditions (IC) for model parameters that are fittable
    model_parameters_fittable_fixed (optional)
        Indicators for model parameters that are to remain fixed at IC value
    normalize_iv_performance_matrix (optional)
        Indicator for normalizing currents by I_sc_A_0, voltages by V_oc_A_0, and
            temperatures by T_K_0
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

    model_parameters_unfittable = ModelParametersUnfittable(
        N_s=iv_performance_matrix.N_s,
        T_degC_0=iv_performance_matrix.T_degC_0,
        I_sc_A_0=iv_performance_matrix.I_sc_A_0,
    )
    N_s = model_parameters_unfittable.N_s
    T_K_0 = convert_temperature(
        model_parameters_unfittable.T_degC_0, "Celsius", "Kelvin"
    )

    if model_parameters_fittable_ic is None:
        model_parameters_fittable_ic = determine_model_parameters_fittable_ic(
            model_parameters_unfittable=model_parameters_unfittable,
            ivft_data=iv_performance_matrix.ivft_data,
        )

    if model_parameters_fittable_fixed is None:
        model_parameters_fittable_fixed = ModelParametersFittableFixed()

    # FIXME Implement data scaling?
    if normalize_iv_performance_matrix:
        V_V_scale = iv_performance_matrix.V_oc_V_0
        I_A_scale = iv_performance_matrix.I_sc_A_0
        T_K_scale = iv_performance_matrix.T_K_0
    else:
        V_V_scale = 1.0
        I_A_scale = 1.0
        T_K_scale = 1.0

    def f(x: numpy.ndarray, beta: numpy.ndarray) -> numpy.ndarray:
        """
        Implicit system of SDM-derived equations over which model parameters are
        optimized. Note closure over some fixed variables.
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

        scaled_thermal_voltage_V = (N_s * n_0 * k_B_J_per_K * T_K) / q_C

        # Parallel conductance with photoconductive shunt.
        F = I_sc_A / model_parameters_unfittable.I_sc_A_0
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

        # Maximum attained at maximum-power point. (Derivative numerator set to zero.)
        y1 = (I_mp_A * R_s_Ohm_0 - V_mp_V) * (
            I_rs_A
            / scaled_thermal_voltage_V
            * numpy.exp(V_diode_mp_V / scaled_thermal_voltage_V)
            + G_p_S
        ) + I_mp_A

        # Open-circuit point.
        V_diode_oc_V = V_oc_V
        y2 = (
            I_ph_A
            - I_rs_A * numpy.expm1(V_diode_oc_V / scaled_thermal_voltage_V)
            - G_p_S * V_diode_oc_V
        )

        return numpy.vstack((y0, y1, y2))

    xdata = numpy.vstack(
        (
            iv_performance_matrix.I_sc_A,
            iv_performance_matrix.I_mp_A,
            iv_performance_matrix.V_mp_V,
            iv_performance_matrix.V_oc_V,
            iv_performance_matrix.T_K,
        )
    )
    ydata = numpy.zeros((3, xdata.shape[-1]))
    beta0 = numpy.array(
        [
            numpy.log(model_parameters_fittable_ic.I_rs_A_0),
            model_parameters_fittable_ic.n_0,
            model_parameters_fittable_ic.R_s_Ohm_0,
            model_parameters_fittable_ic.G_p_S_0,
            model_parameters_fittable_ic.E_g_eV_0,
        ]
    )
    bounds = (numpy.zeros_like(beta0), numpy.full_like(beta0, numpy.inf))
    bounds[0][0] = -numpy.inf  # Log of reverse saturation current.
    fix_beta = [
        asdict(model_parameters_fittable_fixed)[key]
        for key in asdict(model_parameters_fittable_ic).keys()
    ]

    if odr_options is None:
        odr_options = OdrOptions()

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

    # Transform back fit values. Applies validation.
    model_parameters_fittable = ModelParametersFittable(
        I_rs_A_0=float(numpy.exp(solver_result.beta[0])),
        n_0=float(solver_result.beta[1]),
        R_s_Ohm_0=float(solver_result.beta[2]),
        G_p_S_0=float(solver_result.beta[3]),
        E_g_eV_0=float(solver_result.beta[4]),
    )

    return ModelParametersFitResultODR(
        model_parameters_ic=ModelParameters(
            **asdict(model_parameters_unfittable),
            **asdict(model_parameters_fittable_ic),
        ),
        model_parameters=ModelParameters(
            **asdict(model_parameters_unfittable),
            **asdict(model_parameters_fittable),
        ),
        solver_result=solver_result,
    )


def compute_fit_quality(
    iv_performance_matrix: IVPerformanceMatrix,
    model_parameters: ModelParameters,
    newton_options: Optional[NewtonOptions] = None,
) -> Tuple[dict, IVCurveParametersArray]:
    """Compute quality metrics for the fit against the measured performance matrix."""

    iv_curve_parameters = sde_sim.iv_curve_parameters(
        model_parameters=sdm_ps_ae.compute_sde_model_parameters(
            ft_data=FTData(
                F=iv_performance_matrix.F, T_degC=iv_performance_matrix.T_degC
            ),
            model_parameters=model_parameters,
        ),
        newton_options=newton_options,
    )

    I_sc_pc_error = 100 * (
        iv_curve_parameters.I_sc_A / iv_performance_matrix.I_sc_A - 1
    )
    I_mp_pc_error = 100 * (
        iv_curve_parameters.I_mp_A / iv_performance_matrix.I_mp_A - 1
    )
    P_mp_pc_error = 100 * (
        iv_curve_parameters.P_mp_W / iv_performance_matrix.P_mp_W - 1
    )
    V_mp_pc_error = 100 * (
        iv_curve_parameters.V_mp_V / iv_performance_matrix.V_mp_V - 1
    )
    V_oc_pc_error = 100 * (
        iv_curve_parameters.V_oc_V / iv_performance_matrix.V_oc_V - 1
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

"""
Calibrate simple single-diode model (SDM) using nonlinear least squares (NLLS) and
information from photovoltaic device's specification datasheet (spec sheet).

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Tuple

import numpy
from scipy.constants import convert_temperature
import scipy.optimize


from pvfit.common import k_B_eV_per_K
from pvfit.measurement.iv.types import (
    FTData,
    IVCurveParametersArray,
    IVFTData,
    IVPerformanceMatrix,
    SpecSheetParameters,
)
from pvfit.modeling.dc.common import get_scaled_thermal_voltage
import pvfit.modeling.dc.single_diode.equation.simple.simulation as sde_sim
from pvfit.modeling.dc.single_diode.model.simple import types
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as sdm_ae
from pvfit.modeling.dc.single_diode.model.simple.inference_ic import (
    estimate_model_parameters_fittable_ic,
)

delta_T_degC = 0.1


def fun(
    x: numpy.ndarray,
    *,
    N_s: int,
    T_degC_0: float,
    I_sc_A_0: float,
    I_mp_A_0: float,
    V_mp_V_0: float,
    V_oc_V_0: float,
    dI_sc_dT_A_per_K_0: float,
    dP_mp_dT_W_per_K_0: float,
    dV_oc_dT_V_per_K_0: float,
):
    """Computes all the simultaneous constraints produced by SDM at RC."""
    # Everything measured at RC, where F=1 and T=T_degC_0.
    I_rs_A_0 = numpy.exp(x[0])
    n_0 = x[1]
    R_s_Ohm_0 = x[2]
    G_p_S_0 = x[3]
    E_g_eV_0 = x[4]

    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")
    scaled_thermal_voltage_V_0 = n_0 * get_scaled_thermal_voltage(
        N_s=N_s, T_degC=T_degC_0
    )

    # Derivative of reverse-saturation current w.r.t. temperature at RC.
    dI_rs_1_dT_A_per_K_0 = (I_rs_A_0 / T_K_0) * (
        3 + E_g_eV_0 / (n_0 * k_B_eV_per_K * T_K_0)
    )

    # Photocurrent at RC.
    I_ph_A_0 = (
        I_rs_A_0 * numpy.expm1((I_sc_A_0 * R_s_Ohm_0) / scaled_thermal_voltage_V_0)
        + G_p_S_0 * I_sc_A_0 * R_s_Ohm_0
        + I_sc_A_0
    )

    # Derivative of photocurrent w.r.t. T at RC.
    dI_ph_dT_A_per_K_0 = (
        dI_rs_1_dT_A_per_K_0
        * numpy.expm1((I_sc_A_0 * R_s_Ohm_0) / scaled_thermal_voltage_V_0)
        + ((I_rs_A_0 * R_s_Ohm_0) / (scaled_thermal_voltage_V_0 * T_K_0))
        * numpy.exp((I_sc_A_0 * R_s_Ohm_0) / scaled_thermal_voltage_V_0)
        * (T_K_0 * dI_sc_dT_A_per_K_0 - I_sc_A_0)
        + dI_sc_dT_A_per_K_0 * (G_p_S_0 * R_s_Ohm_0 + 1)
    )

    # Maximum-power power point (zero sum of currents at diode anode).
    V_diode_mp_V_0 = V_mp_V_0 + I_mp_A_0 * R_s_Ohm_0
    y_0 = (
        I_ph_A_0
        - I_rs_A_0 * numpy.expm1(V_diode_mp_V_0 / scaled_thermal_voltage_V_0)
        - G_p_S_0 * V_diode_mp_V_0
    ) - I_mp_A_0

    # Maximum power attained at maximum-power point (zero derivative w.r.t. voltage).
    y_1 = (V_mp_V_0 - I_mp_A_0 * R_s_Ohm_0) * (
        (I_rs_A_0 / scaled_thermal_voltage_V_0)
        * numpy.exp(V_diode_mp_V_0 / scaled_thermal_voltage_V_0)
        + G_p_S_0
    ) - I_mp_A_0

    # Open-circuit voltage point (zero sum of currents at diode anode).
    y_2 = (
        I_ph_A_0
        - I_rs_A_0 * numpy.expm1(V_oc_V_0 / scaled_thermal_voltage_V_0)
        - V_oc_V_0 * G_p_S_0
    )

    # Maximum-power power temperature coefficient.
    maximum_powers = sde_sim.P_mp(
        model_parameters=sdm_ae.compute_sde_model_parameters(
            ft_data=FTData(
                F=1,
                T_degC=numpy.array([T_degC_0 - delta_T_degC, T_degC_0 + delta_T_degC]),
            ),
            model_parameters=types.ModelParameters(
                N_s=N_s,
                T_degC_0=T_degC_0,
                I_sc_A_0=I_sc_A_0,
                I_rs_A_0=I_rs_A_0,
                n_0=n_0,
                R_s_Ohm_0=R_s_Ohm_0,
                G_p_S_0=G_p_S_0,
                E_g_eV_0=E_g_eV_0,
            ),
        ),
        newton_options={"maxiter": 1000},
    )["P_mp_W"]
    y_3 = (
        (maximum_powers[1] - maximum_powers[0]) / (2 * delta_T_degC)
    ) - dP_mp_dT_W_per_K_0

    # Open-circuit voltage temperature coefficient.
    y_4 = (
        (
            dI_ph_dT_A_per_K_0
            - dI_rs_1_dT_A_per_K_0 * numpy.expm1(V_oc_V_0 / scaled_thermal_voltage_V_0)
            + ((I_rs_A_0 * V_oc_V_0) / (scaled_thermal_voltage_V_0 * T_K_0))
            * numpy.exp(V_oc_V_0 / scaled_thermal_voltage_V_0)
        )
        / (
            (I_rs_A_0 / scaled_thermal_voltage_V_0)
            * numpy.exp(V_oc_V_0 / scaled_thermal_voltage_V_0)
            + G_p_S_0
        )
    ) - dV_oc_dT_V_per_K_0

    output = [y_0, y_1, y_2, y_3, y_4]

    return numpy.array(output)


def fit(
    *, spec_sheet_parameters: SpecSheetParameters, method: str = "trf"
) -> types.FitResultLeastSquares:
    """
    Use nonlinear least squares (NLLS) to fit the implicit 6-parameter single-diode
    model (SDM) equivalent-circuit model given module specification (spec) sheet data at
    reference condtions.

    Parameters
    ----------
    spec_sheet_parameters
        Parameters from specification datasheet

    Returns
    -------
    dictionary with the following
        model_parameters
            Model parameters from fit
        model_parameters_fittable_ic
            Model parameters from fit's initial-condition (IC) calculation
        optimize_result
            Nonlinear least squares solver result (for a transformed problem)
    """

    # FUTURE Optionally add one more OC at which to fit just three points, not
    # coefficients. Then scan through IEC 61853-1 matrix for best additional point to
    # add.

    model_parameters_unfittable = types.ModelParametersUnfittable(
        N_s=spec_sheet_parameters.N_s,
        T_degC_0=spec_sheet_parameters.T_degC_0,
    )
    types.validate_model_parameters_unfittable(
        model_parameters_unfittable=model_parameters_unfittable,
    )

    I_sc_A_0 = spec_sheet_parameters.I_sc_A_0
    I_mp_A_0 = spec_sheet_parameters.I_mp_A_0
    V_mp_V_0 = spec_sheet_parameters.V_mp_V_0
    V_oc_V_0 = spec_sheet_parameters.V_oc_V_0
    dI_sc_dT_A_per_degC_0 = spec_sheet_parameters.dI_sc_dT_A_per_degC_0
    dP_mp_dT_W_per_degC_0 = spec_sheet_parameters.dP_mp_dT_W_per_degC_0
    dV_oc_dT_V_per_degC_0 = spec_sheet_parameters.dV_oc_dT_V_per_degC_0
    N_s = model_parameters_unfittable["N_s"]
    T_degC_0 = model_parameters_unfittable["T_degC_0"]

    ivft_data = IVFTData(
        I_A=numpy.array(
            [spec_sheet_parameters.I_sc_A_0, spec_sheet_parameters.I_mp_A_0, 0.0]
        ),
        V_V=numpy.array(
            [0.0, spec_sheet_parameters.V_mp_V_0, spec_sheet_parameters.V_oc_V_0]
        ),
        F=1.0,
        T_degC=T_degC_0,
    )

    model_parameters_fittable_ic = estimate_model_parameters_fittable_ic(
        ivft_data=ivft_data,
        model_parameters_unfittable=model_parameters_unfittable,
        material=spec_sheet_parameters.material,
    )

    x_0 = numpy.array(
        [
            numpy.log(model_parameters_fittable_ic["I_rs_A_0"]),
            model_parameters_fittable_ic["n_0"],
            model_parameters_fittable_ic["R_s_Ohm_0"],
            model_parameters_fittable_ic["G_p_S_0"],
            model_parameters_fittable_ic["E_g_eV_0"],
        ]
    )

    kwargs = {
        "N_s": N_s,
        "T_degC_0": T_degC_0,
        "I_sc_A_0": I_sc_A_0,
        "I_mp_A_0": I_mp_A_0,
        "V_mp_V_0": V_mp_V_0,
        "V_oc_V_0": V_oc_V_0,
        "dI_sc_dT_A_per_K_0": dI_sc_dT_A_per_degC_0,
        "dP_mp_dT_W_per_K_0": dP_mp_dT_W_per_degC_0,
        "dV_oc_dT_V_per_K_0": dV_oc_dT_V_per_degC_0,
    }

    # NLLS
    method_kwargs = {"method": method}

    if method == "trf":
        method_kwargs["jac"] = "3-point"

    if method in ("trf", "dogbox"):
        method_kwargs["bounds"] = ([-numpy.inf, 0, 0, 0, 0], numpy.inf)

    optimize_result = scipy.optimize.least_squares(
        fun,
        x_0,
        **method_kwargs,
        x_scale="jac",
        max_nfev=10000 * (len(kwargs) - 2),
        kwargs=kwargs,
    )

    if not optimize_result.success:
        raise RuntimeError(
            "least_squares solver failed to converge to a solution, "
            f"{optimize_result.status}: {optimize_result.message}"
        )

    # Transform back fit values.
    model_parameters_fittable = types.ModelParametersFittable(
        I_sc_A_0=I_sc_A_0,
        I_rs_A_0=numpy.exp(optimize_result.x[0]),
        n_0=optimize_result.x[1],
        R_s_Ohm_0=optimize_result.x[2],
        G_p_S_0=optimize_result.x[3],
        E_g_eV_0=optimize_result.x[4],
    )

    # Raise if something didn't work. For example, bad user-provided value or something
    # computed as NaN.
    types.validate_model_parameters_fittable(
        model_parameters_fittable=model_parameters_fittable
    )

    return types.FitResultLeastSquares(
        model_parameters_ic=types.ModelParameters(
            **model_parameters_unfittable,
            **model_parameters_fittable_ic,
        ),
        model_parameters=types.ModelParameters(
            **model_parameters_unfittable,
            **model_parameters_fittable,
        ),
        optimize_result=optimize_result,
    )


def compute_fit_quality(
    iv_performance_matrix: IVPerformanceMatrix,
    model_parameters: types.ModelParameters,
) -> Tuple[dict, IVCurveParametersArray]:
    """Compute FIXME"""

    iv_curve_parameters = sde_sim.iv_curve_parameters(
        model_parameters=sdm_ae.compute_sde_model_parameters(
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

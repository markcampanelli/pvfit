"""
Calibrate single-diode model (SDM) from IEC 61853-1 matrix data (or similar) using
orthogonal distance regression (ODR).
"""

from typing import Optional, Tuple
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
from pvfit.measurement.iv.types import IVFTData
import pvfit.modeling.dc.single_diode.equation.simulation as sde_sim
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as ae
from pvfit.modeling.dc.single_diode.model.simple.inference_ic import (
    estimate_model_parameters_fittable_ic,
)
from pvfit.modeling.dc.single_diode.model.simple.types import (
    ModelParameters,
    ModelParametersFittableFixedProvided,
    ModelParametersFittableICProvided,
    ModelParametersUnfittable,
    get_model_parameters_fittable_fixed_default,
    validate_model_parameters_unfittable,
)
from pvfit.types import OdrOptions


def fun(beta, x, N_s, T_K_0):
    """FIXME"""
    I_rs_1_A_0 = numpy.exp(beta[0])
    n_1_0 = beta[1]
    R_s_Ohm_0 = beta[2]
    G_p_S_0 = beta[3]
    E_g_eV_0 = beta[4]

    I_sc_A = x[0, :]
    I_mp_A = x[1, :]
    V_mp_V = x[2, :]
    V_oc_V = x[3, :]
    T_K = x[4, :]

    scaled_thermal_voltage_V = (N_s * n_1_0 * k_B_J_per_K * T_K) / q_C

    # Reverse-saturation current.
    I_rs_1_A = (
        I_rs_1_A_0
        * (T_K / T_K_0) ** 3
        * numpy.exp(E_g_eV_0 / (n_1_0 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))
    )

    # Photocurrent from short-circuit point.
    V_diode_sc_V = I_sc_A * R_s_Ohm_0
    I_ph_A = (
        I_rs_1_A * numpy.expm1(V_diode_sc_V / scaled_thermal_voltage_V)
        + G_p_S_0 * V_diode_sc_V
        + I_sc_A
    )

    # Maximum-power point.
    V_diode_mp_V = V_mp_V + I_mp_A * R_s_Ohm_0
    y0 = (
        I_ph_A
        - I_rs_1_A * numpy.expm1(V_diode_mp_V / scaled_thermal_voltage_V)
        - G_p_S_0 * V_diode_mp_V
        - I_mp_A
    )

    # Maximum attained at maximum-power point.
    y1 = (
        (I_mp_A * R_s_Ohm_0 - V_mp_V)
        * (
            I_rs_1_A
            / scaled_thermal_voltage_V
            * numpy.exp(V_diode_mp_V / scaled_thermal_voltage_V)
            + G_p_S_0
        )
        + I_mp_A
    ) / (
        I_rs_1_A
        / scaled_thermal_voltage_V
        * numpy.exp(V_diode_mp_V / scaled_thermal_voltage_V)
        + G_p_S_0 * R_s_Ohm_0
        + 1
    )

    # Open-circuit point.
    V_diode_oc_V = V_oc_V
    y2 = (
        I_ph_A
        - I_rs_1_A * numpy.expm1(V_diode_oc_V / scaled_thermal_voltage_V)
        - G_p_S_0 * V_diode_oc_V
    )

    return numpy.vstack((y0, y1, y2))


def fit(
    *,
    ivft_data: IVFTData,
    model_parameters_unfittable: ModelParametersUnfittable,
    model_parameters_fittable_ic_provided: Optional[
        ModelParametersFittableICProvided
    ] = None,
    model_parameters_fittable_fixed_provided: Optional[
        ModelParametersFittableFixedProvided
    ] = None,
    material: str = "x-Si",
    normalize_iv_curves: bool = True,
    odr_options: Optional[OdrOptions] = None,
) -> Tuple[ModelParameters, scipy.odr.ODR]:
    """
    Use orthogonal distance regression (ODR) to fit the implicit 6-parameter
    equivalent-circuit single-diode model (SDM) given current-voltage (I-V) curve
    data taken over a range of effective-irradiance ratio and cell temperature (F-T)
    operating conditions.

    FIXME Add inputs and outputs.
    """
    validate_model_parameters_unfittable(
        model_parameters_unfittable=model_parameters_unfittable,
    )
    N_s = model_parameters_unfittable["N_s"]
    T_degC_0 = model_parameters_unfittable["T_degC_0"]
    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")

    model_parameters_fittable_ic = estimate_model_parameters_fittable_ic(
        ivft_data=ivft_data,
        model_parameters_unfittable=model_parameters_unfittable,
        model_parameters_fittable_ic_provided=model_parameters_fittable_ic_provided,
        material=material,
    )

    # if normalize_iv_curve:
    #     V_V_scale = iv_curve_parameters["V_oc_V"]
    #     I_A_scale = iv_curve_parameters["I_sc_A"]
    # else:
    #     V_V_scale = 1.0
    #     I_A_scale = 1.0

    data = scipy.odr.Data(
        numpy.vstack(
            (
                I_sc_A,
                I_mp_A,
                V_mp_V,
                V_oc_V,
                convert_temperature(T_degC, "Celsius", "Kelvin"),
            )
        ),
        3,
    )

    model = scipy.odr.Model(fun, implicit=True, extra_args=(N_s, T_K_0))

    # Prepare initial conditions.
    # FIXME Use constants for STC values.
    I_sc_A_0 = I_sc_A[numpy.logical_and(G_W_per_m2 == 1000, T_degC == 25)].item()
    V_V_matrix = numpy.concatenate((numpy.zeros_like(I_sc_A), V_mp_V, V_oc_V))
    I_A_matrix = numpy.concatenate((I_sc_A, I_mp_A, numpy.zeros_like(V_oc_V)))
    F_matrix = numpy.concatenate((I_sc_A, I_sc_A, I_sc_A)) / I_sc_A_0
    T_degC_matrix = numpy.concatenate((T_degC, T_degC, T_degC))

    fit_prep_result = fit_prep(
        V_V=V_V_matrix,
        I_A=I_A_matrix,
        F=F_matrix,
        T_degC=T_degC_matrix,
        N_s=N_s,
        T_degC_0=T_degC_0,
        material=material,
        model_params_ic=model_params_ic,
        model_params_fixed=model_params_fixed,
    )

    beta0 = numpy.array(
        [
            numpy.log(fit_prep_result["model_params_ic"]["I_rs_1_A_0"]),
            fit_prep_result["model_params_ic"]["n_1_0"],
            fit_prep_result["model_params_ic"]["R_s_Ohm_0"],
            fit_prep_result["model_params_ic"]["G_p_S_0"],
            fit_prep_result["model_params_ic"]["E_g_eV_0"],
        ]
    )

    # Check for provided fit parameters to be fixed, and assign default if None.
    # FIXME I_sc_A_0 Is not used in this fit, but it is in the TypedDict.
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
            # R_s_Ohm was negative.
            ifixb[3] = 0
            beta0[3] = 0.0
            recompute = True

        if output.beta[4] < 0:
            # G_p_S was negative.
            ifixb[4] = 0
            beta0[4] = 0.0
            recompute = True

    # Transform back fit values.
    model_parameters = ModelParameters(
        N_s=N_s,
        T_degC_0=T_degC_0,
        I_sc_A_0=I_sc_A_0,
        I_rs_1_A_0=numpy.exp(output.beta[0]),
        n_1_0=output.beta[1],
        R_s_Ohm_0=output.beta[2],
        G_p_S_0=output.beta[3],
        E_g_eV_0=output.beta[4],
    )

    return model_parameters, odr

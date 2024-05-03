"""
PVfit: Initial conditions (IC) for single-diode model (SDM) inference.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional

import numpy

from pvfit.measurement.iv.computation import estimate_iv_curve_parameters
from pvfit.measurement.iv.types import IVCurve, IVFTData
from pvfit.modeling.dc.common import MATERIALS_INFO, Material
import pvfit.modeling.dc.single_diode.equation.simple.inference_ic as sde_inf_ic
import pvfit.modeling.dc.single_diode.equation.simple.types as sde_types
from pvfit.modeling.dc.single_diode.model.simple.types import (
    ModelParametersFittable,
    ModelParametersFittableProvided,
    ModelParametersUnfittable,
    validate_model_parameters_fittable,
    validate_model_parameters_unfittable,
)


def estimate_model_parameters_fittable_ic(
    *,
    ivft_data: IVFTData,
    model_parameters_unfittable: ModelParametersUnfittable,
    model_parameters_fittable_ic_provided: Optional[
        ModelParametersFittableProvided
    ] = None,
    material: Material = Material.xSi,
) -> ModelParametersFittable:
    """
    Estimate initial conditions (IC) for fittable model parameters.

    Parameters
    ----------
    ivft_data
        I-V-F-T data describing I-V performance over a range of operating conditions
    model_parameters_unfittable
        Model parameters that are are not fittable
    model_parameters_fittable_ic_provided (optional)
        Provided initial conditions (IC) for model parameters that are fittable
    material (optional)
        Material of PV device, required if IC for material band gap not provided

    Returns
    -------
    model_parameters_fittable_ic
        Complete initial conditions (IC) for model parameters that are fittable (complete)
    """
    validate_model_parameters_unfittable(
        model_parameters_unfittable=model_parameters_unfittable
    )

    if model_parameters_fittable_ic_provided is None:
        model_parameters_fittable_ic_provided = ModelParametersFittableProvided()

    I_sc_A_0_ic = model_parameters_fittable_ic_provided.get("I_sc_A_0", None)
    I_rs_A_0_ic = model_parameters_fittable_ic_provided.get("I_rs_A_0", None)
    n_0_ic = model_parameters_fittable_ic_provided.get("n_0", None)
    R_s_Ohm_0_ic = model_parameters_fittable_ic_provided.get("R_s_Ohm_0", None)
    G_p_S_0_ic = model_parameters_fittable_ic_provided.get("G_p_S_0", None)
    E_g_eV_0_ic = model_parameters_fittable_ic_provided.get("E_g_eV_0", None)

    if (
        I_sc_A_0_ic is None
        or I_rs_A_0_ic is None
        or n_0_ic is None
        or R_s_Ohm_0_ic is None
        or G_p_S_0_ic is None
    ):
        # Take data near RC: F within 7.5% of 1 and T_degC within 3 degC of T_degC_0.
        F_tol = 0.075
        T_degC_tol = 3.0
        ref_indices = numpy.logical_and(
            numpy.abs(ivft_data.F - 1) <= F_tol,
            numpy.abs(ivft_data.T_degC - model_parameters_unfittable["T_degC_0"])
            <= T_degC_tol,
        )

        if numpy.sum(ref_indices) >= 3:
            # F-normalize currents to RC (a rough irradiance-only correction).
            iv_curve_0 = IVCurve(
                V_V=ivft_data.V_V[ref_indices],
                I_A=ivft_data.I_A[ref_indices] / ivft_data.F[ref_indices],
            )
            iv_curve_parameters_0 = estimate_iv_curve_parameters(iv_curve=iv_curve_0)
        else:
            raise NotImplementedError(
                "cannot estimate initial conditions with fewer than three distinct I-V "
                "data points sufficiently close to reference conditions"
            )

        model_parameters_unfittable_sde = sde_types.ModelParametersUnfittable(
            N_s=model_parameters_unfittable["N_s"],
            T_degC=model_parameters_unfittable["T_degC_0"],
        )

        model_parameters_fittable_ic_provided_sde_0 = (
            sde_types.ModelParametersFittableProvided()
        )

        if I_sc_A_0_ic is not None:
            model_parameters_fittable_ic_provided_sde_0["I_ph_A"] = I_sc_A_0_ic

        if I_rs_A_0_ic is not None:
            model_parameters_fittable_ic_provided_sde_0["I_rs_A"] = I_rs_A_0_ic

        if n_0_ic is not None:
            model_parameters_fittable_ic_provided_sde_0["n"] = n_0_ic

        if R_s_Ohm_0_ic is not None:
            model_parameters_fittable_ic_provided_sde_0["R_s_Ohm"] = R_s_Ohm_0_ic

        if G_p_S_0_ic is not None:
            model_parameters_fittable_ic_provided_sde_0["G_p_S"] = G_p_S_0_ic

        # Use SDE fit at RC to get ICs for all parameters except E_g_eV_0. Try to
        # respect user-selected ICs, but without fixing for better accomodation.
        model_parameters_sde_ic_0 = sde_inf_ic.estimate_model_parameters_fittable_ic(
            iv_curve_parameters=iv_curve_parameters_0,
            model_parameters_unfittable=model_parameters_unfittable_sde,
            model_parameters_fittable_ic_provided=model_parameters_fittable_ic_provided_sde_0,
        )

        # Update IC's, some of which may have been passed through.
        if I_sc_A_0_ic is None:
            I_sc_A_0_ic = model_parameters_sde_ic_0["I_ph_A"]

        if I_rs_A_0_ic is None:
            I_rs_A_0_ic = model_parameters_sde_ic_0["I_rs_A"]

        if n_0_ic is None:
            n_0_ic = model_parameters_sde_ic_0["n"]

        if R_s_Ohm_0_ic is None:
            R_s_Ohm_0_ic = model_parameters_sde_ic_0["R_s_Ohm"]

        if G_p_S_0_ic is None:
            G_p_S_0_ic = model_parameters_sde_ic_0["G_p_S"]

    # Initial condition for E_g_eV_0.
    if E_g_eV_0_ic is None:
        E_g_eV_0_ic = MATERIALS_INFO[material]["E_g_eV_stc"]

    model_parameters_fittable_ic = ModelParametersFittable(
        I_sc_A_0=I_sc_A_0_ic,
        I_rs_A_0=I_rs_A_0_ic,
        n_0=n_0_ic,
        R_s_Ohm_0=R_s_Ohm_0_ic,
        G_p_S_0=G_p_S_0_ic,
        E_g_eV_0=E_g_eV_0_ic,
    )

    # Raise if something didn't work. For example, bad user-provided value or something
    # computed as NaN.
    validate_model_parameters_fittable(
        model_parameters_fittable=model_parameters_fittable_ic
    )

    return model_parameters_fittable_ic

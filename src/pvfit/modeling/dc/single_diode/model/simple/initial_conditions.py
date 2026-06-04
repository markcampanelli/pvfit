"""
PVfit: Initial conditions (IC) for single-diode model (SDM) inference.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional

import numpy
import scipy.optimize

from pvfit.common import T_degC_abs_zero
from pvfit.measurement.iv.types import FTData, IVCurve, IVData, IVFTData
from pvfit.modeling.dc.common import MATERIALS_INFO, Material
import pvfit.modeling.dc.single_diode.equation.simple.initial_conditions as sde_ic
import pvfit.modeling.dc.single_diode.equation.simple.simulation as sde_sim
import pvfit.modeling.dc.single_diode.equation.simple.types as sde_types
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as sdm_simple_ae
from pvfit.modeling.dc.single_diode.model.simple.types import (
    ModelParameters,
    ModelParametersFittable,
    ModelParametersUnfittable,
    OperatingConditions,
)


def determine_model_parameters_fittable_ic(
    *,
    model_parameters_unfittable: Optional[ModelParametersUnfittable] = None,
    ivft_data: Optional[IVFTData] = None,
    I_rs_A_0: Optional[float] = None,
    n_0: Optional[float] = None,
    R_s_Ohm_0: Optional[float] = None,
    G_p_S_0: Optional[float] = None,
    E_g_eV_0: Optional[float] = None,
    material: Material = Material.xSi,
) -> ModelParametersFittable:
    """
    Estimate initial conditions (IC) for fittable model parameters at reference
    conditions (RC).

    Parameters
    ----------
    model_parameters_unfittable (optional if all ICs provided)
        Model parameters that are are not fittable
    ivft_data (optional if all ICs provided)
        I-V-F-T data describing I-V performance over a range of operating conditions,
        including RC
    I_rs_A_0 (optional)
        Provided IC for reverse-saturation current at RC
    n_0 (optional)
        Provided IC for ideality factor at RC
    R_s_Ohm_0 (optional)
        Provided IC for series resistance at RC
    G_p_S_0 (optional)
        Provided IC for parallel conductance at RC
    E_g_eV_0 (optional)
        Provided IC for material bandgap at RC
    material (optional)
        Material of PV device, required if IC for material bandgap at RC when not
            provided

    Returns
    -------
    model_parameters_fittable_ic
        Initial conditions (IC) for model parameters that are fittable
    """

    if I_rs_A_0 is None or n_0 is None or R_s_Ohm_0 is None or G_p_S_0 is None:
        if model_parameters_unfittable is None or ivft_data is None:
            raise ValueError(
                "must specify both model_parameters_unfittable and ivft_data if any "
                "initial condition other than E_g_eV_0 is None and thus to be "
                "calculated from data"
            )

        # Take data near RC: F within 7.5% of 1 and T_degC within 3 degC of T_degC_0.
        F_tol = 0.075
        T_degC_tol = 3.0
        ref_indices = numpy.logical_and(
            numpy.abs(ivft_data.F - 1) <= F_tol,
            numpy.abs(ivft_data.T_degC - model_parameters_unfittable.T_degC_0)
            <= T_degC_tol,
        )

        if numpy.sum(ref_indices) < 3:
            raise ValueError(
                "cannot estimate initial conditions with fewer than three distinct I-V "
                "data points sufficiently close to reference conditions"
            )

        # F-normalize currents to RC (a rough irradiance-only correction).
        iv_curve = IVCurve(
            V_V=ivft_data.V_V[ref_indices],
            I_A=ivft_data.I_A[ref_indices] / ivft_data.F[ref_indices],
        )
        model_parameters_unfittable_sde = sde_types.ModelParametersUnfittable(
            N_s=model_parameters_unfittable.N_s,
            T_degC=model_parameters_unfittable.T_degC_0,
        )
        model_parameters_fittable_ic_sde_0 = (
            sde_ic.determine_model_parameters_fittable_ic(
                model_parameters_unfittable=model_parameters_unfittable_sde,
                iv_curve=iv_curve,
                I_ph_A=model_parameters_unfittable.I_sc_A_0,
                I_rs_A=I_rs_A_0,
                n=n_0,
                R_s_Ohm=R_s_Ohm_0,
                G_p_S=G_p_S_0,
            )
        )

        # These are "passed through" if specified value is not None.
        I_rs_A_0 = model_parameters_fittable_ic_sde_0.I_rs_A
        n_0 = model_parameters_fittable_ic_sde_0.n
        R_s_Ohm_0 = model_parameters_fittable_ic_sde_0.R_s_Ohm
        G_p_S_0 = model_parameters_fittable_ic_sde_0.G_p_S

    # Initial condition for E_g_eV_0.
    if E_g_eV_0 is None:
        E_g_eV_0 = MATERIALS_INFO[material].E_g_eV_stc

    return ModelParametersFittable(
        I_rs_A_0=I_rs_A_0,
        n_0=n_0,
        R_s_Ohm_0=R_s_Ohm_0,
        G_p_S_0=G_p_S_0,
        E_g_eV_0=E_g_eV_0,
    )


def determine_operating_conditions_ic(
    *,
    model_parameters: Optional[ModelParameters],
    iv_data: Optional[IVData] = None,
    F: Optional[float] = None,
    T_degC: Optional[float] = None,
) -> OperatingConditions:
    """
    Estimate initial conditions (IC) for operating conditions (OC).

    FIXME
    Parameters
    ----------
    model_parameters (optional if all ICs provided)
        Model parameters for device
    iv_data (optional if all ICs provided)
        I-V data describing I-V performance at OC to be inferred
    F (optional)
        Provided IC for effective-irradiance ratio
    T_degC (optional)
        Provided IC for device temperature

    Returns
    -------
    operating_conditions_ic
        IC for OC
    """

    if F is None or T_degC is None:
        if model_parameters is None or iv_data is None:
            raise ValueError(
                "must specify both model_parameters and iv_data if any initial "
                "condition is None and thus to be calculated from data"
            )

        # F IC.
        if F is None:
            # To estimate I_sc_A, get I_A at index of V_V that closest in absolute value to 0.
            I_sc_A_est = iv_data.I_A[numpy.argmin(numpy.abs(iv_data.V_V))]
            F = I_sc_A_est / model_parameters.I_sc_A_0

        # T_degC IC.
        if T_degC is None:
            T_degC = scipy.optimize.least_squares(
                lambda x: sde_sim.I_sum_diode_anode_at_I_V(
                    iv_data=IVData(I_A=iv_data.I_A, V_V=iv_data.V_V),
                    model_parameters=sdm_simple_ae.compute_sde_model_parameters(
                        ft_data=FTData(F=F, T_degC=x),
                        model_parameters=model_parameters,
                    ),
                )["I_sum_diode_anode_A"],
                model_parameters.T_degC_0,
            ).x.item()

    return OperatingConditions(
        F=F,
        T_degC=T_degC,
    )

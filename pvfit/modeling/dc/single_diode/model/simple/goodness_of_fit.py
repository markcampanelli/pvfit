"""
PVfit: Goodness-of-fit metrics for simple single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Tuple

import numpy

from pvfit.measurement.iv.types import (
    FTData,
    IVCurveParametersArray,
    IVPerformanceMatrix,
)
import pvfit.modeling.dc.single_diode.equation.simulation as sde_sim
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as ae
import pvfit.modeling.dc.single_diode.model.simple.types as types


def compute_matrix_mape_mbpe(
    iv_performance_matrix: IVPerformanceMatrix, model_parameters: types.ModelParameters
) -> Tuple[dict, IVCurveParametersArray]:
    """Compute"""

    iv_curve_parameters = sde_sim.iv_curve_parameters(
        model_parameters=ae.compute_sde_model_parameters(
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

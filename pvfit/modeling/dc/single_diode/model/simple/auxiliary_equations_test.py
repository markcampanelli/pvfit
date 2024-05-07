"""
PVfit testing: Auxiliary equations for simple single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy
import pytest

from pvfit.common import T_degC_stc
from pvfit.measurement.iv.types import FTData
from pvfit.modeling.dc.common import MATERIALS_INFO, Material
import pvfit.modeling.dc.single_diode.equation.simple.types as sde_types
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as sdm_ae
from pvfit.modeling.dc.single_diode.model.simple.types import ModelParameters


@pytest.mark.parametrize(
    "params",
    [
        {  # Can handle all python scalar inputs.
            "given": {
                "ft_data": FTData(F=1.25, T_degC=T_degC_stc),
                "model_parameters": ModelParameters(
                    N_s=1,
                    T_degC_0=T_degC_stc,
                    I_sc_A_0=7.0,
                    I_rs_A_0=6.0e-7,
                    n_0=1.25,
                    R_s_Ohm_0=0.0,  # Zero series resistance gives exact solution.
                    G_p_S_0=0.005,
                    E_g_eV_0=MATERIALS_INFO[Material.xSi]["E_g_eV_stc"],
                ),
            },
            "expected": {
                "model_parameters": sde_types.ModelParameters(
                    N_s=numpy.array(1),
                    T_degC=numpy.array(T_degC_stc),
                    I_ph_A=numpy.array(1.25 * 7.0),
                    I_rs_A=numpy.array(6.0e-7),
                    n=numpy.array(1.25),
                    R_s_Ohm=numpy.array(0.0),
                    G_p_S=numpy.array(0.005),
                )
            },
        },
        {  # Can handle all but two python scalar inputs.
            "given": {
                "ft_data": FTData(F=1.25, T_degC=T_degC_stc + 10),
                "model_parameters": ModelParameters(
                    N_s=numpy.array([1, 9]),
                    T_degC_0=T_degC_stc,
                    I_sc_A_0=7.0,
                    I_rs_A_0=6.0e-7,
                    n_0=1.25,
                    R_s_Ohm_0=numpy.array([0.024, 9 * 0.024]),
                    G_p_S_0=0.005,
                    E_g_eV_0=MATERIALS_INFO[Material.xSi]["E_g_eV_stc"],
                ),
            },
            "expected": {
                "model_parameters": sde_types.ModelParameters(
                    N_s=numpy.array([1, 9]),
                    T_degC=numpy.array([T_degC_stc + 10, T_degC_stc + 10]),
                    I_ph_A=numpy.array([8.752198, 8.760598]),
                    I_rs_A=numpy.array([2.056226e-06, 2.056226e-06]),
                    n=numpy.array([1.25, 1.25]),
                    R_s_Ohm=numpy.array([0.024, 9 * 0.024]),
                    G_p_S=numpy.array([0.005, 0.005]),
                )
            },
        },
    ],
)
def test_compute(params):
    given = params["given"]
    expected = params["expected"]

    model_parameters_got = sdm_ae.compute_sde_model_parameters(
        ft_data=given["ft_data"],
        model_parameters=given["model_parameters"],
    )
    model_parameters_expected = expected["model_parameters"]

    assert set(model_parameters_got.keys()) == set(model_parameters_expected.keys())

    for key in model_parameters_got:
        assert isinstance(
            model_parameters_got[key], type(model_parameters_expected[key])
        ), key
        assert (
            model_parameters_got[key].shape == model_parameters_expected[key].shape
        ), key
        assert (
            model_parameters_got[key].dtype == model_parameters_expected[key].dtype
        ), key
        numpy.testing.assert_allclose(
            model_parameters_got[key],
            model_parameters_expected[key],
            rtol=1e-05,
            atol=1e-08,
            err_msg=key,
        )

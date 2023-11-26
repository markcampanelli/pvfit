"""
PVfit: DC modeling for Sandia Array Performance Model (SAPM) simulation.

https://energy.sandia.gov/wp-content/gallery/uploads/SAND-2004_PV-Performance-Array-Model.pdf

WARNING: This code is experimental and not subject to semantic versioning.
"""

from typing import Union

import numpy

from pvfit.common.utils import ensure_numpy_scalars


def F_at_E_T(
    *,
    E_b_W_per_m2,
    E_d_W_per_m2,
    T_c_degC,
    AM_a,
    AOI,
    E_0_W_per_m2,
    T_degC_0,
    f_1_params,
    f_2_params,
    f_d,
    alpha_I_sc_per_degC,
):
    """TODO"""
    # : Union[float, numpy.float64, numpy.ndarray]
    return ensure_numpy_scalars(
        f_1(AM_a, **f_1_params)
        * (E_b_W_per_m2 * f_2(AOI, **f_2_params) + f_d * E_d_W_per_m2)
        / E_0_W_per_m2
        * (1 + alpha_I_sc_per_degC * (T_c_degC - T_degC_0))
    )


def f_1(
    *,
    AM_a: Union[float, numpy.float64, numpy.ndarray],
    a_0: Union[float, numpy.float64, numpy.ndarray],
    a_1: Union[float, numpy.float64, numpy.ndarray],
    a_2: Union[float, numpy.float64, numpy.ndarray],
    a_3: Union[float, numpy.float64, numpy.ndarray],
    a_4: Union[float, numpy.float64, numpy.ndarray],
):
    """TODO"""
    return ensure_numpy_scalars(
        a_0 + AM_a * (a_1 + AM_a * (a_2 + AM_a * (a_3 + AM_a * a_4)))
    )


def f_2(
    *,
    AOI: Union[float, numpy.float64, numpy.ndarray],
    b_0: Union[float, numpy.float64, numpy.ndarray],
    b_1: Union[float, numpy.float64, numpy.ndarray],
    b_2: Union[float, numpy.float64, numpy.ndarray],
    b_3: Union[float, numpy.float64, numpy.ndarray],
    b_4: Union[float, numpy.float64, numpy.ndarray],
    b_5: Union[float, numpy.float64, numpy.ndarray],
):
    """TODO"""
    return ensure_numpy_scalars(
        b_0 + AOI * (b_1 + AOI * (b_2 + AOI * (b_3 + AOI * (b_4 + AOI * b_5))))
    )

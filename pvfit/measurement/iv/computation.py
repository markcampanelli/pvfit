"""
PVfit: Computation on current-voltage (I-V) measurements.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy

from pvfit.measurement.iv.types import IVCurve, IVCurveParametersScalar


def estimate_iv_curve_parameters(*, iv_curve: IVCurve) -> IVCurveParametersScalar:
    """
    Estimate the I-V curve parameters for a single I-V curve without a physical model
    (device illuminated).

    The I-V curve is assumed to have a bare minimum of 3 points distributed between Isc
    and Voc, with at least one maximum-power point (Vmp, Imp) strictly in the 1st
    quadrant, at least one short-circuit point (Vsc, Isc) with Vsc < Vmp and Imp < Isc,
    and at least one open-circuit point (Voc, Ioc) with Vmp < Voc and Ioc < Imp.
    Resulting I-V curve should be well-ordered with expected signs.

    Inputs:
      iv_curve containing:
        V_V voltage data for I-V curve
        I_A current data for I-V curve

    Outputs:
      iv_curve_parameters containing:
        I_sc_A short-circuit current
        R_sc_Ohm resistance at short-circuit
        I_mp_A current at maximum power point
        P_mp_W maximum power
        V_mp_V voltage at maximum power point
        R_oc_Ohm resistance at open-circuit
        V_oc_V open-circuit voltage
        FF fill factor

    Exceptions:
        ValueError if any I-V curve parameter cannot be estimated.
    """
    # Initialize values.
    I_sc_A = float("nan")
    dI_dV_sc_S = float("nan")
    I_x_A = float("nan")
    V_x_V = float("nan")
    I_mp_A = float("nan")
    P_mp_W = float("nan")
    V_mp_V = float("nan")
    I_xx_A = float("nan")
    V_xx_V = float("nan")
    dI_dV_oc_S = float("nan")
    V_oc_V = float("nan")

    # Estimate Pmp with empirical value (no local polynomial fitting).
    I_A = iv_curve.I_A
    V_V = iv_curve.V_V
    P_W = iv_curve.P_W
    P_mp_W_idx = numpy.argmax(P_W)
    I_mp_A = I_A[P_mp_W_idx]
    V_mp_V = V_V[P_mp_W_idx]
    P_mp_W = P_W[P_mp_W_idx]

    if not (0 < V_mp_V and 0 < I_mp_A):
        raise ValueError(
            f"maximum-power point '({V_mp_V}, {I_mp_A})' of I-V cure not strictly in "
            "first quadrant"
        )

    # Take Isc as first current nearest zero voltage that is bigger than Imp.
    for idx in numpy.argsort(numpy.abs(V_V)):
        if I_mp_A < I_A[idx]:
            I_sc_A = float(I_A[idx])
            break

    if numpy.isnan(I_sc_A):
        raise ValueError("could not estimate short-circuit current from I-V curve")

    # Take Voc as first voltage nearest zero current that is bigger than Vmp.
    for idx in numpy.argsort(numpy.abs(I_A)):
        if V_mp_V < V_V[idx]:
            V_oc_V = float(V_V[idx])
            break

    if numpy.isnan(V_oc_V):
        raise ValueError("could not estimate open-circuit voltage from I-V curve")

    V_x_V = V_oc_V / 2
    I_x_A = I_A[numpy.abs(V_V - V_x_V).argmin()]

    if numpy.logical_or(I_sc_A <= I_x_A, I_x_A <= I_mp_A):
        # No data points between 0 and V_mp_V or data are quite noisey.
        I_x_A = (I_sc_A + I_mp_A) / 2

    # Take slope at Isc as 1/2 slope from (0, Isc) to (Vx, Ix).
    dI_dV_sc_S = ((I_sc_A - I_x_A) / -V_x_V) / 2

    V_xx_V = (V_mp_V + V_oc_V) / 2
    I_xx_A = float(I_A[numpy.abs(V_V - V_xx_V).argmin()])

    if numpy.logical_or(I_mp_A <= I_xx_A, I_xx_A <= 0):
        # No data points between V_mp_V and V_oc_V or data are quite noisey.
        I_xx_A = I_mp_A / 2

    # Take slope at Voc as 1/2 slope from (Vxx, Ixx) to (Voc, 0).
    dI_dV_oc_S = (I_xx_A / (V_xx_V - V_oc_V)) / 2

    return IVCurveParametersScalar(
        I_sc_A=I_sc_A,
        R_sc_Ohm=-1 / dI_dV_sc_S,
        V_x_V=V_x_V,
        I_x_A=I_x_A,
        I_mp_A=I_mp_A,
        P_mp_W=P_mp_W,
        V_mp_V=V_mp_V,
        V_xx_V=V_xx_V,
        I_xx_A=I_xx_A,
        R_oc_Ohm=-1 / dI_dV_oc_S,
        V_oc_V=V_oc_V,
        FF=P_mp_W / (I_sc_A * V_oc_V),
    )

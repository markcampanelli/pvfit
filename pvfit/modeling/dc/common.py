"""
PVfit: Common items for DC modeling.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from enum import Enum
from typing import Protocol, TypedDict

import numpy
import scipy.constants


from pvfit.common import k_B_J_per_K, q_C
from pvfit.types import FloatArray, FloatBroadcastable, IntBroadcastable


class Material(Enum):
    """Photovoltaic materials recognized by PVfit."""

    CIGS = "CIGS"  # Copper Indium Gallium Selenide (CIGS).
    CIS = "CIS"  # Copper Indium diSelenide (CIS).
    CdTe = "CdTe"  # Cadmium Telluride (CdTe).
    GaAs = "GaAs"  # Gallium Arsenide (GaAs).
    monoSi = "mono-Si"  # Mono-crystalline Silicon (mono-Si).
    multiSi = "multi-Si"  # Multi-crystalline Silicon (multi-Si).
    polySi = "poly-Si"  # Poly-crystalline Silicon (poly-Si).
    xSi = "x-Si"  # Crystalline Silicon (x-Si).


class MaterialsInfo(TypedDict):
    """Information for photovoltaic materials recognized by PVfit."""

    E_g_eV_stc: float


# Materials information.
MATERIALS_INFO = {
    Material.CIGS: MaterialsInfo(
        # De Soto et al. 2006.
        E_g_eV_stc=1.15,
    ),
    Material.CIS: MaterialsInfo(
        # De Soto et al. 2006.
        E_g_eV_stc=1.010,
    ),
    Material.CdTe: MaterialsInfo(
        # De Soto et al. 2006.
        E_g_eV_stc=1.475,
    ),
    Material.GaAs: MaterialsInfo(
        # At 300 K, Kittel, C., Intro. to Solid State Physics, 6th ed. 1986, p 185.
        E_g_eV_stc=1.43,
    ),
    Material.monoSi: MaterialsInfo(
        # De Soto et al. 2006.
        E_g_eV_stc=1.121,
    ),
    Material.multiSi: MaterialsInfo(
        # De Soto et al. 2006.
        E_g_eV_stc=1.121,
    ),
    Material.polySi: MaterialsInfo(
        # De Soto et al. 2006.
        E_g_eV_stc=1.121,
    ),
    Material.xSi: MaterialsInfo(
        # De Soto et al. 2006.
        E_g_eV_stc=1.121,
    ),
}

# Limits on ideality factor on first diode.
N_IC_MIN = 1.0
N_IC_MAX = 2.0


def get_scaled_thermal_voltage(
    N_s: IntBroadcastable, T_degC: FloatBroadcastable
) -> FloatArray:
    """
    Compute thermal voltage [V], scaled by the number of cells in series in each
    parallel string.
    """
    return numpy.array(
        N_s
        * (
            k_B_J_per_K
            * scipy.constants.convert_temperature(T_degC, "Celsius", "Kelvin")
            / q_C
        )
    )


class IamFunction(Protocol):
    """Type definition for incident-angle modifier functions."""

    def __call__(self, *, angle_deg: FloatArray) -> FloatArray: ...


def iam_factory(*, iam_angle_deg: FloatArray, iam_frac: FloatArray) -> IamFunction:
    """Generate interpolating function for incident-angle modifier (IAM)."""

    def iam(*, angle_deg: FloatArray) -> FloatArray:
        """TODO"""
        if numpy.any(numpy.logical_or(angle_deg < 0, angle_deg > 180)):
            raise ValueError("angle_deg not between 0 and 180, inclusive")

        iam_ = scipy.interpolate.PchipInterpolator(
            iam_angle_deg, iam_frac, extrapolate=False
        )(angle_deg)
        iam_[90 < angle_deg] = 0.0

        return iam_

    return iam


class ConstantThenLinearDegradationFunction(Protocol):
    """Type definition for constant-then-linear degradation functions."""

    def __call__(self, *, years_since_commissioning: FloatArray) -> FloatArray: ...


def constant_then_linear_degredation_factory(
    *, first_year_degredation: float, subsequent_year_degredation: float
) -> ConstantThenLinearDegradationFunction:
    """
    Constant-then-linear degredation, e.g. from a module warranty.

    No degredation before first year. First year has a constant degredation, which is
    continuous to linear degredation in subsequent years.
    """

    def constant_then_linear_degredation(
        *, years_since_commissioning: FloatArray
    ) -> FloatArray:
        """TODO"""
        fractional_degredation = numpy.empty_like(years_since_commissioning)
        fractional_degredation[years_since_commissioning < 0] = 0.0
        fractional_degredation[
            numpy.logical_and(
                0 <= years_since_commissioning, years_since_commissioning <= 1
            )
        ] = first_year_degredation
        fractional_degredation[1 < years_since_commissioning] = (
            first_year_degredation
            + subsequent_year_degredation
            * (years_since_commissioning[1 < years_since_commissioning] - 1)
        )

        return fractional_degredation

    return constant_then_linear_degredation

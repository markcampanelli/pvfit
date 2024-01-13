"""
PVfit: Types for current-voltage (I-V) measurement.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from collections.abc import Iterable
from typing import TypedDict

import numpy

from pvfit.common import T_degC_abs_zero
from pvfit.modeling.dc.common import G_hemi_W_per_m2_stc, T_degC_stc
from pvfit.types import FloatArray, FloatBroadcastable, FloatVector


class IVData:
    """
    Broadcast-compatible current-voltage (I-V) data at possibly many irradiance and
    temperature.
    """

    def __init__(self, *, I_A: FloatBroadcastable, V_V: FloatBroadcastable) -> None:
        """
        Parameters
        ----------
        I_A
            Terminal currents [A]
        V_A
            Terminal voltages [V]
        """
        self._I_A, self._V_V = numpy.broadcast_arrays(I_A, V_V)

    @property
    def I_A(self) -> FloatArray:
        """Get terminal currents."""
        return self._I_A

    @property
    def V_V(self) -> FloatArray:
        """Get terminal voltages."""
        return self._V_V

    @property
    def P_W(self) -> FloatArray:
        """Get terminal powers."""
        return numpy.array(self._I_A * self._V_V)


class FTData:
    """Broadcast-compatible operation conditions (F-T) data."""

    def __init__(self, *, F: FloatBroadcastable, T_degC: FloatBroadcastable) -> None:
        """
        Parameters
        ----------
        F
            Effective-irradiance ratios [·]
        T_degC
            Cell temperatures [°C]
        """
        if numpy.any(F < 0):
            raise ValueError("F less than zero")

        if numpy.any(T_degC <= T_degC_abs_zero):
            raise ValueError("T_degC less than or equal to absolute zero temperature")

        self._F, self._T_degC = numpy.broadcast_arrays(F, T_degC)

    @property
    def F(self) -> FloatArray:
        """Get effective-irradiance ratios."""
        return self._F

    @property
    def T_degC(self) -> FloatArray:
        """Get cell temperatures."""
        return self._T_degC


class IVFTData(IVData, FTData):
    """Broadcast-compatible combined I-V and operating conditions (I-V-F-T) data."""

    def __init__(
        self,
        *,
        I_A: FloatBroadcastable,
        V_V: FloatBroadcastable,
        F: FloatBroadcastable,
        T_degC: FloatBroadcastable,
    ) -> None:
        """
        Parameters
        ----------
        I_A
            Terminal currents [A]
        V_A
            Terminal voltages [V]
        F
            Effective-irradiance ratios [·]
        T_degC
            Cell temperatures [°C]
        """
        IVData.__init__(self, I_A=I_A, V_V=V_V)
        FTData.__init__(self, F=F, T_degC=T_degC)
        # After super initializations, broadcast everything together.
        self._I_A, self._V_V, self._F, self._T_degC = numpy.broadcast_arrays(
            self.I_A, self.V_V, F, T_degC
        )


class IVCurve(IVData):
    """
    Current-voltage (I-V) curve in positive power quadrant at one irradiance and
    temperature.
    """

    def __init__(self, *, V_V: FloatVector, I_A: FloatVector) -> None:
        """Initialize I-V curve with validation."""
        super().__init__(V_V=V_V, I_A=I_A)

        if self.V_V.ndim != 1:
            raise ValueError("V_V is not one dimensional")

        if self.I_A.ndim != 1:
            raise ValueError("I_A is not one dimensional")

        if self.V_V.size == 0:
            raise ValueError("V_V is empty")

        if self.I_A.size == 0:
            raise ValueError("I_A is empty")

        if self.V_V.size != self.I_A.size:
            raise ValueError("V_V and I_A have different lengths")

        if numpy.unique(self.V_V).size < 3:
            raise ValueError("fewer than three unique voltages in I-V curve")

        if numpy.unique(self.I_A).size < 3:
            raise ValueError("fewer than three unique currents in I-V curve")

        if numpy.all(self.P_W <= 0):
            raise ValueError("I-V curve has no points in positive-power quadrant")


class IVPerformanceMatrix():
    """I-V performance matrix data."""

    def __init__(
        self,
        *,
        iv_curves: Iterable[IVCurve],
        G_W_per_m2: Iterable[float],
        T_degC: Iterable[float],
        G_W_per_m2_0: float = G_hemi_W_per_m2_stc,
        T_degC_0: float = T_degC_stc,
    ) -> None:
        """
        Parameters
        ----------
        iv_curves
            I-V curve at each operating condition, each minimally with Isc, Pmp, and Voc
        G_W_per_m2
            Plane of array irradiance [W/m^2]
        T_degC
            Cell temperatures [°C]
        G_W_per_m2
            Reference plane of array irradiance [W/m^2]
        T_degC
            Reference cell temperatures [°C]
        """
        # FIXME
        # Validations
        # Ensure same iterable lengths.
        # Check that I-V curves have an Isc, Voc, and separate Pmp point. If multiple,
        # nonzero powers, then take max power.


class IVCurveParametersScalar(TypedDict):
    """I-V curve parameters (at one operating condition)."""

    I_sc_A: float
    R_sc_Ohm: float
    V_x_V: float
    I_x_A: float
    V_mp_V: float
    P_mp_W: float
    I_mp_A: float
    V_xx_V: float
    I_xx_A: float
    R_oc_Ohm: float
    V_oc_V: float
    FF: float


class IVCurveParametersArray(TypedDict):
    """I-V curve parameters (at one/more operating conditions)."""

    I_sc_A: FloatArray
    R_sc_Ohm: FloatArray
    V_x_V: FloatArray
    I_x_A: FloatArray
    V_mp_V: FloatArray
    P_mp_W: FloatArray
    I_mp_A: FloatArray
    V_xx_V: FloatArray
    I_xx_A: FloatArray
    R_oc_Ohm: FloatArray
    V_oc_V: FloatArray
    FF: FloatArray

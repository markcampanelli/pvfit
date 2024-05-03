"""
PVfit: Types for current-voltage (I-V) measurement.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import TypedDict

import numpy
from scipy.constants import convert_temperature

from pvfit.common import T_degC_abs_zero
from pvfit.modeling.dc.common import Material
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


class IVPerformanceMatrix:
    """I-V performance matrix data."""

    def __init__(
        self,
        *,
        material: Material,
        N_s: int,
        I_sc_A: FloatVector,
        I_mp_A: FloatVector,
        V_mp_V: FloatVector,
        V_oc_V: FloatVector,
        E_W_per_m2: FloatVector,
        T_degC: FloatVector,
        E_W_per_m2_0: float,
        T_degC_0: float,
    ) -> None:
        """
        Parameters
        ----------
        material
            material of PV device
        N_s
            number of cells in series in each parallel string
        I_sc_A
            Currents at short-circuit [A]
        I_mp_A
            Currents at maximum power [A]
        V_mp_V
            Voltages at maximum power [V]
        V_oc_V
            Voltages at open-circuit [V]
        E_W_per_m2
            Plane of array irradiances [W/m^2]
        T_degC
            Cell temperatures [°C]
        E_W_per_m2_0
            Reference plane of array irradiance [W/m^2]
        T_degC_0
            Reference cell temperatures [°C]
        """
        if (
            len(I_sc_A)
            != len(I_mp_A)
            != len(V_mp_V)
            != len(V_oc_V)
            != len(E_W_per_m2)
            != len(T_degC)
        ):
            raise ValueError("input collections must all have the same length")

        self._material = material
        self._N_s = N_s
        self._I_sc_A = I_sc_A
        self._I_mp_A = I_mp_A
        self._V_mp_V = V_mp_V
        self._V_oc_V = V_oc_V
        self._E_W_per_m2 = E_W_per_m2
        self._T_degC = T_degC
        self._E_W_per_m2_0 = E_W_per_m2_0
        self._T_degC_0 = T_degC_0
        ref_idx = numpy.logical_and(E_W_per_m2 == E_W_per_m2_0, T_degC == T_degC_0)
        self._I_sc_A_0 = I_sc_A[ref_idx].item()
        self._I_mp_A_0 = I_mp_A[ref_idx].item()
        self._V_mp_V_0 = V_mp_V[ref_idx].item()
        self._V_oc_V_0 = V_oc_V[ref_idx].item()

    @property
    def material(self) -> Material:
        return self._material

    @property
    def N_s(self) -> int:
        return self._N_s

    @property
    def I_sc_A(self) -> FloatVector:
        return self._I_sc_A

    @property
    def I_mp_A(self) -> FloatVector:
        return self._I_mp_A

    @property
    def P_mp_W(self) -> FloatVector:
        return self._I_mp_A * self._V_mp_V

    @property
    def V_mp_V(self) -> FloatVector:
        return self._V_mp_V

    @property
    def V_oc_V(self) -> FloatVector:
        return self._V_oc_V

    @property
    def E_W_per_m2(self) -> FloatVector:
        return self._E_W_per_m2

    @property
    def F(self) -> FloatVector:
        return self._I_sc_A / self._I_sc_A_0

    @property
    def T_degC(self) -> FloatVector:
        return self._T_degC

    @property
    def T_K(self) -> FloatVector:
        return convert_temperature(self._T_degC, "Celsius", "Kelvin")

    @property
    def E_W_per_m2_0(self) -> float:
        return self._E_W_per_m2_0

    @property
    def T_degC_0(self) -> float:
        return self._T_degC_0

    @property
    def T_K_0(self) -> float:
        return convert_temperature(self._T_degC_0, "Celsius", "Kelvin")

    @property
    def I_sc_A_0(self) -> float:
        return self._I_sc_A_0

    @property
    def I_mp_A_0(self) -> float:
        return self._I_mp_A_0

    @property
    def P_mp_W_0(self) -> float:
        return self._I_mp_A_0 * self._V_mp_V_0

    @property
    def V_mp_V_0(self) -> float:
        return self._V_mp_V_0

    @property
    def V_oc_V_0(self) -> float:
        return self._V_oc_V_0

    @property
    def ivft_data(self) -> IVFTData:
        I_A = []
        V_V = []
        F = []
        T_degC = []

        for I_sc_A, I_mp_A, V_mp_V, V_oc_V, T_degC_ in zip(
            self._I_sc_A, self._I_mp_A, self._V_mp_V, self._V_oc_V, self._T_degC
        ):
            I_A.extend([I_sc_A, I_mp_A, 0.0])
            V_V.extend([0.0, V_mp_V, V_oc_V])
            F.extend([I_sc_A, I_sc_A, I_sc_A])  # Currents to be normalized by I_sc_A_0.
            T_degC.extend([T_degC_, T_degC_, T_degC_])

        I_A = numpy.array(I_A)
        V_V = numpy.array(V_V)
        F = numpy.array(F) / self._I_sc_A_0
        T_degC = numpy.array(T_degC)

        return IVFTData(I_A=I_A, V_V=V_V, F=F, T_degC=T_degC)


class SpecSheetParameters:
    """
    Performance parameters at specified reference conditions, typically found on the
    specification datasheet of a photovoltaic module.
    """

    def __init__(
        self,
        *,
        material: Material,
        N_s: int,
        I_sc_A_0: float,
        I_mp_A_0: float,
        V_mp_V_0: float,
        V_oc_V_0: float,
        dI_sc_dT_A_per_degC_0: float,
        dP_mp_dT_W_per_degC_0: float,
        dV_oc_dT_V_per_degC_0: float,
        E_W_per_m2_0: float,
        T_degC_0: float,
    ) -> None:
        """
        Parameters
        ----------
        material
            material of PV device
        N_s
            number of cells in series in each parallel string
        I_sc_A_0
            Current at short-circuit at reference condition [A]
        I_mp_A_0
            Current at maximum power at reference condition [A]
        V_mp_V_0
            Voltage at maximum power at reference condition [V]
        V_oc_V_0
            Voltage at open-circuit at reference condition [V]
        dI_sc_dT_A_per_degC_0
            Derivative of current at short-circuit with respect to temperature at
                reference condition [A/°C]
        dP_mp_dT_W_per_degC_0
            Derivative of maximum power with respect to temperature at reference
                condition [W/°C]
        dV_oc_dT_V_per_degC_0
            Derivative of voltage at open-circuit with respect to temperature at
                reference condition [V/°C]
        E_W_per_m2_0
            Reference plane of array irradiance, defaults to STC [W/m^2]
        T_degC_0
            Reference cell temperatures, defaults to STC [°C]
        """
        self._material = material
        self._N_s = N_s
        self._I_sc_A_0 = I_sc_A_0
        self._I_mp_A_0 = I_mp_A_0
        self._V_mp_V_0 = V_mp_V_0
        self._V_oc_V_0 = V_oc_V_0
        self._dI_sc_dT_A_per_degC_0 = dI_sc_dT_A_per_degC_0
        self._dP_mp_dT_W_per_degC_0 = dP_mp_dT_W_per_degC_0
        self._dV_oc_dT_V_per_degC_0 = dV_oc_dT_V_per_degC_0
        self._E_W_per_m2_0 = E_W_per_m2_0
        self._T_degC_0 = T_degC_0

    @property
    def material(self) -> Material:
        return self._material

    @property
    def N_s(self) -> int:
        return self._N_s

    @property
    def I_sc_A_0(self) -> float:
        return self._I_sc_A_0

    @property
    def I_mp_A_0(self) -> float:
        return self._I_mp_A_0

    @property
    def V_mp_V_0(self) -> float:
        return self._V_mp_V_0

    @property
    def V_oc_V_0(self) -> float:
        return self._V_oc_V_0

    @property
    def dI_sc_dT_A_per_degC_0(self) -> float:
        return self._dI_sc_dT_A_per_degC_0

    @property
    def dP_mp_dT_W_per_degC_0(self) -> float:
        return self._dP_mp_dT_W_per_degC_0

    @property
    def dV_oc_dT_V_per_degC_0(self) -> float:
        return self._dV_oc_dT_V_per_degC_0

    @property
    def E_W_per_m2_0(self) -> float:
        return self._E_W_per_m2_0

    @property
    def T_degC_0(self) -> float:
        return self._T_degC_0

    @property
    def T_K_0(self) -> float:
        return convert_temperature(self._T_degC_0, "Celsius", "Kelvin")

    @property
    def P_mp_W_0(self) -> float:
        return self.I_mp_A_0 * self.V_mp_V_0


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

"""
PVfit: Spectral correction types.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy

from pvfit.common import c_m_per_s, h_J_s, q_C
from pvfit.types import FloatArray, FloatVector


class DataFunction:
    r"""
    Store data representing one/more functions in :math:`\mathbb{R}^2` with common,
    monotonic increasing domain values. All data are finite floating point values.

    TODO Describe interface.
    """

    def __init__(self, *, x: FloatVector, y: FloatArray) -> None:
        # Copies inputs and sorts on increasing x values.
        x = numpy.asarray_chkfinite(x, dtype=float)
        if x.size == 0:
            raise ValueError("x must have at least one element.")
        if 1 < x.ndim:
            raise ValueError("x cannot have dimension greater than one.")
        x_size = x.size
        x, x_argsort = numpy.unique(x, return_index=True)
        if x.size != x_size:
            raise ValueError("x values must be unique.")
        if y.shape[-1] != x_size:
            raise ValueError("last dimension of y must equal size of x.")
        self._x = x
        y = numpy.asarray_chkfinite(y, dtype=float)
        self._y = y[..., x_argsort]

    @property
    def x(self) -> FloatVector:
        """Return domain x."""
        return self._x

    @property
    def y(self) -> FloatArray:
        """Return codomain(s) y."""
        return self._y

    @property
    def num_functions(self) -> int:
        """Return the number of functions represented."""
        return max(1, self._y.ndim - 1)

    def __eq__(self, obj):
        """Check object equality."""
        return (
            isinstance(obj, DataFunction)
            and numpy.array_equal(self.x, obj.x)
            and numpy.array_equal(self.y, obj.y)
        )


class DataFunctionNonnegativeXNonnegativeY(DataFunction):
    r"""
    Store data representing a function in :math:`\mathbb{R}^2` with :math:`0 \leq x` and
    :math:`0 \leq y`.

    TODO Describe interface.
    """

    def __init__(self, *, x: FloatVector, y: FloatArray):
        super().__init__(x=x, y=y)
        if numpy.any(self.x < 0):
            raise ValueError("x values must all be non-negative.")
        if numpy.any(self.y < 0):
            raise ValueError("y values must all be non-negative.")


class QuantumEfficiency(DataFunctionNonnegativeXNonnegativeY):
    """
    Store data representing a quantum efficiency (QE) curve(s).

    Any function value y at x=0 is interpreted as the (finite) limit as x->0^+.

    TODO Describe interface and units [nm] and [1] or [%].
    """

    def __init__(
        self, *, lambda_nm: FloatVector, QE: FloatArray, is_percent: bool = False
    ):
        super().__init__(x=lambda_nm, y=QE)
        # Do not convert raw data. Instead track if it is given as a percent.
        self.is_percent = is_percent

    @property
    def lambda_nm(self) -> FloatVector:
        """Return wavelengths."""
        return self.x

    @property
    def QE(self) -> FloatArray:
        """Return QE(s) as fraction."""
        if self.is_percent:
            return self.y / 100
        else:
            return self.y

    @property
    def QE_percent(self) -> FloatArray:
        """Return QE(s) as percent."""
        if self.is_percent:
            return self.y
        else:
            return 100 * self.y

    @property
    def S_A_per_W(self) -> "SpectralResponsivity":
        """
        Convert quantum efficiency (QE) curve(s) to spectral responsivity (SR) curve(s).

        TODO Describe interface.
        """
        return SpectralResponsivity(
            lambda_nm=self.lambda_nm,
            S_A_per_W=self.QE * self.lambda_nm * 1.0e-9 * q_C / (h_J_s * c_m_per_s),
        )


class SpectralIrradiance(DataFunctionNonnegativeXNonnegativeY):
    """
    Store data representing spectral irradiance curve(s).

    TODO Describe interface and units [nm] and [A/W/m^2].
    """

    def __init__(self, *, lambda_nm: FloatVector, E_W_per_m2_nm: FloatArray):
        super().__init__(x=lambda_nm, y=E_W_per_m2_nm)

    @property
    def lambda_nm(self) -> FloatVector:
        """Return wavelengths."""
        return self.x

    @property
    def E_W_per_m2_nm(self) -> FloatArray:
        """Return spectral irradiance curve(s)."""
        return self.y

    @property
    def E_total_W_per_m2(self) -> FloatArray:
        """Return integrated total irradiance(s)."""
        return numpy.trapz(self.E_W_per_m2_nm, x=self.lambda_nm)

    # FIXME Should this be a method?
    def get_E_total_subinterval_W_per_m2(
        self, *, lambda_min_nm: float, lambda_max_nm: float
    ) -> FloatArray:
        """Return integrated total irradiance(s) on subinterval."""
        if lambda_min_nm < self.lambda_nm[0] or lambda_max_nm > self.lambda_nm[-1]:
            raise ValueError("subinterval not contained in non-tail wavelength domain")

        subinterval_idx = numpy.logical_and(
            lambda_min_nm <= self.lambda_nm, self.lambda_nm <= lambda_max_nm
        )

        return numpy.trapz(
            self.E_W_per_m2_nm[..., subinterval_idx], x=self.lambda_nm[subinterval_idx]
        )


class SpectralIrradianceWithTail(SpectralIrradiance):
    """
    Store data representing spectral irradiance curve(s), including nonzero integrated
    tail irradiance to positive infinity after largest wavelength in domian.

    TODO Describe interface and units [nm] and [A/W/m^2].
    """

    def __init__(
        self,
        *,
        lambda_nm: FloatVector,
        E_W_per_m2_nm: FloatArray,
        E_tail_W_per_m2: FloatArray,
    ):
        super().__init__(lambda_nm=lambda_nm, E_W_per_m2_nm=E_W_per_m2_nm)

        if numpy.any(E_tail_W_per_m2 <= 0):
            raise ValueError("E_tail_W_per_m2 must be positive.")

        if E_tail_W_per_m2.ndim != E_W_per_m2_nm.ndim - 1:
            raise ValueError(
                "E_tail_W_per_m2 must have exactly one fewer dimension than "
                "E_W_per_m2_nm."
            )

        self._E_tail_W_per_m2 = E_tail_W_per_m2

    @property
    def E_tail_W_per_m2(self) -> FloatVector:
        """Return integrated tail irradiance(s)."""
        return self._E_tail_W_per_m2

    @property
    def E_total_W_per_m2(self) -> FloatArray:
        """Return integrated total irradiance(s), including integrated tail(s)."""
        return super().E_total_W_per_m2 + self._E_tail_W_per_m2


class SpectralResponsivity(DataFunctionNonnegativeXNonnegativeY):
    """
    Store data representing a spectral responsivity (SR) curve(s).

    TODO Describe interface and units [nm] and [A/W].
    """

    def __init__(self, *, lambda_nm: FloatVector, S_A_per_W: FloatArray):
        super().__init__(x=lambda_nm, y=S_A_per_W)

    @property
    def lambda_nm(self) -> FloatVector:
        """Return wavelengths."""
        return self.x

    @property
    def S_A_per_W(self) -> FloatArray:
        """Return spectral responsivity curve(s)."""
        return self.y

    @property
    def QE(self) -> "QuantumEfficiency":
        """
        Convert spectral responsivity (SR) curve(s) to quantum efficiency (QE) curve(s).

        TODO Describe interface.
        """
        return QuantumEfficiency(
            lambda_nm=self.lambda_nm,
            QE=self.S_A_per_W * h_J_s * c_m_per_s / (self.lambda_nm * 1.0e-9 * q_C),
        )

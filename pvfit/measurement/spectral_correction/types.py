"""
PVfit: Spectral correction types.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy

from pvfit.common import c_m_per_s, h_J_s, q_C
from pvfit.types import FloatVector


class DataFunction:
    r"""
    Store data representing one/more functions in :math:`\mathbb{R}^2` with common,
    monotonic increasing domain values.

    TODO Describe interface.
    """

    def __init__(self, *, x: FloatVector, y: FloatVector) -> None:
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
        self.x = x
        y = numpy.asarray_chkfinite(y, dtype=float)
        self.y = y[..., x_argsort]

    def __eq__(self, obj):
        return (
            isinstance(obj, DataFunction)
            and numpy.array_equal(self.x, obj.x)
            and numpy.array_equal(self.y, obj.y)
        )


class DataFunctionPositiveXNonnegativeY(DataFunction):
    r"""
    Store data representing a function in :math:`\mathbb{R}^2` with :math:`0 < x` and
    :math:`0 \leq y`.

    TODO Describe interface.
    """

    def __init__(self, *, x: FloatVector, y: FloatVector):
        super().__init__(x=x, y=y)
        if numpy.any(self.x <= 0):
            raise ValueError("x values must all be positive.")
        if numpy.any(self.y < 0):
            raise ValueError("y values must all be non-negative.")


class QuantumEfficiency(DataFunctionPositiveXNonnegativeY):
    """
    Store data representing a quantum efficiency (QE) curve.

    TODO Describe interface and units [nm] and [1] or [%].
    """

    def __init__(
        self, *, lambda_nm: FloatVector, QE: FloatVector, is_percent: bool = False
    ):
        super().__init__(x=lambda_nm, y=QE)
        # Do not convert raw data. Instead track if it is given as a percent.
        self.is_percent = is_percent

    @property
    def lambda_nm(self) -> FloatVector:
        """Return wavelengths."""
        return self.x

    @property
    def QE(self) -> FloatVector:
        """Return QE as fraction."""
        if self.is_percent:
            return self.y / 100
        else:
            return self.y

    @property
    def QE_percent(self) -> FloatVector:
        """Return QE as percent."""
        if self.is_percent:
            return self.y
        else:
            return 100 * self.y

    @property
    def S_A_per_W(self) -> "SpectralResponsivity":
        """
        Convert quantum efficiency (QE) curve to spectral responsivity (SR) curve.

        TODO Describe interface.
        """
        return SpectralResponsivity(
            lambda_nm=self.lambda_nm,
            S_A_per_W=self.QE * self.lambda_nm * 1.0e-9 * q_C / (h_J_s * c_m_per_s),
        )


class SpectralIrradiance(DataFunctionPositiveXNonnegativeY):
    """
    Store data representing a spectral irradiance curve.

    TODO Describe interface and units [nm] and [A/W/m^2].
    """

    def __init__(self, *, lambda_nm: FloatVector, E_W_per_m2_nm: FloatVector):
        super().__init__(x=lambda_nm, y=E_W_per_m2_nm)

    @property
    def lambda_nm(self) -> FloatVector:
        return self.x

    @property
    def E_W_per_m2_nm(self) -> FloatVector:
        return self.y


class SpectralResponsivity(DataFunctionPositiveXNonnegativeY):
    """
    Store data representing a spectral responsivity (SR) curve.

    TODO Describe interface and units [nm] and [A/W].
    """

    def __init__(self, *, lambda_nm: FloatVector, S_A_per_W: FloatVector):
        super().__init__(x=lambda_nm, y=S_A_per_W)

    @property
    def lambda_nm(self) -> FloatVector:
        return self.x

    @property
    def S_A_per_W(self) -> FloatVector:
        return self.y

    @property
    def QE(self) -> "QuantumEfficiency":
        """
        Convert spectral responsivity (SR) curve to quantum efficiency (QE) curve.

        TODO Describe interface.
        """
        return QuantumEfficiency(
            lambda_nm=self.lambda_nm,
            QE=self.S_A_per_W * h_J_s * c_m_per_s / (self.lambda_nm * 1.0e-9 * q_C),
        )

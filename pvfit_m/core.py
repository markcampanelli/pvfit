import numpy
import scipy.constants

q_C = scipy.constants.e
c_m_per_s = scipy.constants.c
h_J_s = scipy.constants.h


class DataCurve:
    """
    Store data representing a curve in R^2 with monotonic increasing x values.

    TODO Describe interface.
    """
    def __init__(self, *, x: numpy.ndarray, y: numpy.ndarray):
        # Copies inputs and sorts on increasing x values.
        x = numpy.asarray_chkfinite(x)
        y = numpy.asarray_chkfinite(y)
        x_argsort = numpy.argsort(x)
        self.x = x[x_argsort]
        self.y = y[x_argsort]

    def __eq__(self, obj):
        return isinstance(obj, DataCurve) and numpy.array_equal(self.x, obj.x) and numpy.array_equal(self.y, obj.y)


class DataCurvePositiveXNonnegativeY(DataCurve):
    """
    Store data representing a curve in R^2 with 0<x and 0<=y.

    TODO Describe interface.
    """
    def __init__(self, *, x: numpy.ndarray, y: numpy.ndarray):
        super().__init__(x=x, y=y)
        if numpy.any(self.x <= 0):
            raise ValueError("x values must all be positive.")
        if numpy.any(self.y < 0):
            raise ValueError("y values must all be non-negative.")


class QuantumEfficiency(DataCurvePositiveXNonnegativeY):
    """
    Store data representing a quantum efficiency (QE) curve.

    TODO Describe interface and units [nm] and [1] or [%].
    """
    def __init__(self, *, lambda_: numpy.ndarray, qe: numpy.ndarray, is_percent: bool = False):
        super().__init__(x=lambda_, y=qe)
        # Do not convert raw data. Instead track if it is given as a percent.
        self.is_percent = is_percent

    @property
    def lambda_(self):
        """Return wavelengths."""
        return self.x

    @property
    def qe(self):
        """Return QE as fraction."""
        if self.is_percent:
            return self.y/100
        else:
            return self.y

    @property
    def qe_as_percent(self):
        """Return QE as percent."""
        if self.is_percent:
            return self.y
        else:
            return 100*self.y


class SpectralIrradiance(DataCurvePositiveXNonnegativeY):
    """
    Store data representing a spectral irradiance curve.

    TODO Describe interface and units [nm] and [A/W/m^2].
    """
    def __init__(self, *, lambda_: numpy.ndarray, ir: numpy.ndarray):
        super().__init__(x=lambda_, y=ir)

    @property
    def lambda_(self):
        return self.x

    @property
    def ir(self):
        return self.y


class SpectralResponsivity(DataCurvePositiveXNonnegativeY):
    """
    Store data representing a spectral responsivity (SR) curve.

    TODO Describe interface and units [nm] and [A/W].
    """
    def __init__(self, *, lambda_: numpy.ndarray, sr: numpy.ndarray):
        super().__init__(x=lambda_, y=sr)

    @property
    def lambda_(self):
        return self.x

    @property
    def sr(self):
        return self.y


def convert_qe_to_sr(*, qe: QuantumEfficiency) -> SpectralResponsivity:
    """Convert quantum efficiency (QE) curve to spectral responsivity (SR) curve."""
    return SpectralResponsivity(lambda_=qe.lambda_, sr=qe.qe * qe.lambda_ * 1.e-9 * q_C / (h_J_s * c_m_per_s))


def compute_m(*, sr_td: SpectralResponsivity, ir_td: SpectralIrradiance, sr_rd: SpectralResponsivity,
              ir_rd: SpectralIrradiance, ir_0: SpectralIrradiance) -> float:
    """Compute spectral mismatch correction factor (M) between a reference device (RD) and test device (TD)."""
    raise NotImplementedError

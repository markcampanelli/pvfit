import numpy
import scipy.constants

# Constants with explicit units.
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
        x = numpy.asarray_chkfinite(x, dtype=float)
        if x.size == 0:
            raise ValueError("x must have at least one element.")
        if 1 < x.ndim:
            raise ValueError("x cannot have dimension greater than one.")
        x_size = x.size
        x, x_argsort = numpy.unique(x, return_index=True)
        if x.size != x_size:
            raise ValueError("x values must be unique.")
        self.x = x[x_argsort]
        y = numpy.asarray_chkfinite(y, dtype=float)
        # This will raise if x and y are mismatched.
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


def compute_m(*, sr_td: SpectralResponsivity, ir_td: SpectralIrradiance, sr_rd: SpectralResponsivity,
              ir_rd: SpectralIrradiance, ir_0: SpectralIrradiance) -> float:
    """Compute spectral mismatch correction factor (M) between a reference device (RD) and test device (TD)."""
    return ((inner_product(dc1=sr_td, dc2=ir_td) * inner_product(dc1=sr_rd, dc2=ir_0)) /
            (inner_product(dc1=sr_td, dc2=ir_0) * inner_product(dc1=sr_rd, dc2=ir_rd)))


def convert_qe_to_sr(*, qe: QuantumEfficiency) -> SpectralResponsivity:
    """Convert quantum efficiency (QE) curve to spectral responsivity (SR) curve."""
    return SpectralResponsivity(lambda_=qe.lambda_, sr=qe.qe * qe.lambda_ * 1.e-9 * q_C / (h_J_s * c_m_per_s))


def convert_sr_to_qe(*, sr: SpectralResponsivity) -> QuantumEfficiency:
    """Convert spectral responsivity (SR) curve to curve quantum efficiency (QE)."""
    return QuantumEfficiency(lambda_=sr.lambda_, qe=sr.sr * h_J_s * c_m_per_s / (sr.lambda_ * 1.e-9 * q_C))


def inner_product(*, dc1: DataCurve, dc2: DataCurve) -> float:
    """Compute inner product of two curves as an integral over the common interval of thier domain of defintion."""
    # TODO Warn if computation appears innacurate due to missing non-zero data at end(s) of common interval,
    #  which should include when there is no overlap of intervals or only one point of overlap.
    x_min = numpy.maximum(dc1.x[0], dc2.x[0])
    x_max = numpy.minimum(dc1.x[-1], dc2.x[-1])
    x_union = numpy.union1d(dc1.x, dc2.x)
    x_union = x_union[numpy.logical_and(x_min <= x_union, x_union <= x_max)]
    y1 = numpy.interp(x_union, dc1.x, dc1.y, left=float('nan'), right=float('nan'))
    y2 = numpy.interp(x_union, dc2.x, dc2.y, left=float('nan'), right=float('nan'))

    slopes1 = (y1[1:] - y1[:-1]) / (x_union[1:] - x_union[:-1])
    intercepts1 = y1[:-1] - slopes1 * x_union[:-1]
    slopes2 = (y2[1:] - y2[:-1]) / (x_union[1:] - x_union[:-1])
    intercepts2 = y2[:-1] - slopes2 * x_union[:-1]

    A = intercepts1 * intercepts2
    B = (slopes1 * intercepts2 + slopes2 * intercepts1) / 2
    C = slopes1 * slopes2 / 3

    x_union_squared = x_union * x_union
    x_union_cubed = x_union_squared * x_union
    return float(numpy.sum(C * (x_union_cubed[1:] - x_union_cubed[:-1]) +
                           B * (x_union_squared[1:] - x_union_squared[:-1]) +
                           A * (x_union[1:] - x_union[:-1])))

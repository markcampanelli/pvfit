import warnings

import numpy
import scipy.interpolate

# Constants with explicit units.
q_C = 1.6021766208e-19  # From scipy.constants.e
c_m_per_s = 299792458.0  # From scipy.constants.c
h_J_s = 6.62607004e-34  # From scipy.constants.h


class DataFunction:
    r"""
    Store data representing one/more functions in :math:`\mathbb{R}^2`
    with common, monotonic increasing domain values.

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
        # This will raise if x and y are not broadcast compatible.
        self.y = y[..., x_argsort]

    def __eq__(self, obj):
        return isinstance(obj, DataFunction) and numpy.array_equal(self.x, obj.x) and numpy.array_equal(self.y, obj.y)


class DataFunctionPositiveXNonnegativeY(DataFunction):
    r"""
    Store data representing a function in :math:`\mathbb{R}^2` with
    :math:`0 < x` and :math:`0 \leq y`.

    TODO Describe interface.
    """
    def __init__(self, *, x: numpy.ndarray, y: numpy.ndarray):
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
    def __init__(self, *, lambda_nm: numpy.ndarray, qe: numpy.ndarray, is_percent: bool = False):
        super().__init__(x=lambda_nm, y=qe)
        # Do not convert raw data. Instead track if it is given as a percent.
        self.is_percent = is_percent

    @property
    def lambda_nm(self) -> numpy.ndarray:
        """Return wavelengths."""
        return self.x

    @property
    def qe(self) -> numpy.ndarray:
        """Return QE as fraction."""
        if self.is_percent:
            return self.y/100
        else:
            return self.y

    @property
    def qe_percent(self) -> numpy.ndarray:
        """Return QE as percent."""
        if self.is_percent:
            return self.y
        else:
            return 100*self.y

    @property
    def sr_A_per_W(self) -> "SpectralResponsivity":
        """
        Convert quantum efficiency (QE) curve to spectral responsivity
        (SR) curve.

        TODO Describe interface.
        """
        return SpectralResponsivity(
            lambda_nm=self.lambda_nm, sr_A_per_W=self.qe * self.lambda_nm * 1.e-9 * q_C / (h_J_s * c_m_per_s))


class SpectralIrradiance(DataFunctionPositiveXNonnegativeY):
    """
    Store data representing a spectral irradiance curve.

    TODO Describe interface and units [nm] and [A/W/m^2].
    """
    def __init__(self, *, lambda_nm: numpy.ndarray, si_W_per_m2_nm: numpy.ndarray):
        super().__init__(x=lambda_nm, y=si_W_per_m2_nm)

    @property
    def lambda_nm(self) -> numpy.ndarray:
        return self.x

    @property
    def si_W_per_m2_nm(self) -> numpy.ndarray:
        return self.y


class SpectralResponsivity(DataFunctionPositiveXNonnegativeY):
    """
    Store data representing a spectral responsivity (SR) curve.

    TODO Describe interface and units [nm] and [A/W].
    """
    def __init__(self, *, lambda_nm: numpy.ndarray, sr_A_per_W: numpy.ndarray):
        super().__init__(x=lambda_nm, y=sr_A_per_W)

    @property
    def lambda_nm(self) -> numpy.ndarray:
        return self.x

    @property
    def sr_A_per_W(self) -> numpy.ndarray:
        return self.y

    @property
    def qe(self) -> "QuantumEfficiency":
        """
        Convert spectral responsivity (SR) curve to quantum efficiency
        (QE) curve.

        TODO Describe interface.
        """
        return QuantumEfficiency(
            lambda_nm=self.lambda_nm, qe=self.sr_A_per_W * h_J_s * c_m_per_s / (self.lambda_nm * 1.e-9 * q_C))


def inner_product(*, df1: DataFunction, df2: DataFunction) -> numpy.ndarray:
    r"""
    Compute inner product of two data functions.

    The inner product of two data functions is the integral of the product
    of the two functions over their common domain of defintion. Because
    the data function model is piecewise linear, an algebraic solution
    exists and is used for the computation.  See the :class:`DataFunction`
    class for details on the model that informs the computation.

    Parameters
    ----------
    df1
        First data function.
    df2
        Second data function.

    Returns
    -------
    inner_product : np.ndarray
        Integral of the product of the two data functions over their
        common domain.

    Warns
    ------
    UserWarning
        If `inner_product` is non-finite or is zero due to no domain overlap.

    Notes
    -----
    The inner product is computed as--

    .. math:: \int_{x=x_1}^{x_2} f_1(x) \, f_{2}(x) \, \mathrm{d}x,

    where the interval of integration :math:`[x_1, x_2]` is the common
    domain of the two data functions. If the domains do not overlap, then
    zero is returned.
    """
    x_min = numpy.maximum(df1.x[0], df2.x[0])
    x_max = numpy.minimum(df1.x[-1], df2.x[-1])
    if x_max <= x_min:
        warnings.warn("DataFunction domains do not overlap.")
        return numpy.zeros(numpy.broadcast(df1.y, df2.y).shape)
    x_union = numpy.union1d(df1.x, df2.x)
    x_union = x_union[numpy.logical_and(x_min <= x_union, x_union <= x_max)]
    y1 = scipy.interpolate.interp1d(df1.x, df1.y, copy=False, assume_sorted=True)(x_union)
    y2 = scipy.interpolate.interp1d(df2.x, df2.y, copy=False, assume_sorted=True)(x_union)

    slopes1 = (y1[..., 1:] - y1[..., :-1]) / (x_union[1:] - x_union[:-1])
    intercepts1 = y1[..., :-1] - slopes1 * x_union[:-1]
    slopes2 = (y2[..., 1:] - y2[..., :-1]) / (x_union[1:] - x_union[:-1])
    intercepts2 = y2[..., :-1] - slopes2 * x_union[:-1]

    A = intercepts1 * intercepts2
    B = (slopes1 * intercepts2 + slopes2 * intercepts1) / 2
    C = slopes1 * slopes2 / 3
    x_union_squared = x_union * x_union
    x_union_cubed = x_union_squared * x_union

    inner_product = numpy.array(numpy.sum(C * (x_union_cubed[1:] - x_union_cubed[:-1]) +
                                          B * (x_union_squared[1:] - x_union_squared[:-1]) +
                                          A * (x_union[1:] - x_union[:-1]), axis=-1))
    if not numpy.all(numpy.isfinite(inner_product)):
        warnings.warn("Non-finite inner product detected.")

    return inner_product


def m(*, sr_td: SpectralResponsivity, si_td: SpectralIrradiance, sr_rd: SpectralResponsivity,
      si_rd: SpectralIrradiance, si_0: SpectralIrradiance) -> numpy.ndarray:
    r"""
    Compute spectral mismatch correction factor (:math:`M`).

    The spectral mismatch is between a photovoltaic (PV) test device (TD)
    and a PV reference device (RD), each illuminated by a (possibly
    different) spectral irradiance. The correction is to a particular
    reference spectrum. The spectral responsivities are each at a
    particular (non-explicit) temperature, which may not be the reference
    condition. M should be strictly positive, but could evalute to be
    zero, infinite, or NaN depending on possible zero values of the
    component integrals. See the :class:`SpectralIrradiance` and
    :class:`SpectralResponsivity` classes for details on the data function
    models that inform the computation, which includes vectorized
    computations.

    Parameters
    ----------
    sr_td
        Spectral responsivity of TD [A/W].
    si_td
        Spectral irradiance illuminating TD [W/m2/nm].
    sr_rd
        Spectral responsivity of RD [A/W].
    si_rd
        Spectral irradiance illuminating RD [W/m2/nm].
    si_0
        Spectral irradiance at reference conditions [W/m2/nm].

    Returns
    -------
    M : np.ndarray
        Spectral mismatch correction factor (:math:`M`).

    Warns
    ------
    UserWarning
        If :math:`M` is computed as zero, infinite, or NaN.

    Notes
    -----
    M is computed as--

    .. math:: M = \frac{
        \int_{\lambda=0}^\infty S_\mathrm{TD}(\lambda) \,
        E_\mathrm{TD}(\lambda) \, \mathrm{d}\lambda \,
        \int_{\lambda=0}^\infty S_\mathrm{RD}(\lambda) \,
        E_\mathrm{0}(\lambda) \, \mathrm{d}\lambda}
        {\int_{\lambda=0}^\infty S_\mathrm{TD}(\lambda) \,
        E_\mathrm{0}(\lambda) \, \mathrm{d}\lambda \,
        \int_{\lambda=0}^\infty S_\mathrm{RD}(\lambda) \,
        E_\mathrm{RD}(\lambda) \, \mathrm{d}\lambda}.

    See Also
    --------
    inner_product : The function used to compute the integrals of the
        products of two data functions.
    """
    # TODO Warn if computation appears innacurate due to missing non-zero data at end(s) of common domain intervals.
    M = numpy.array((inner_product(df1=sr_td, df2=si_td) * inner_product(df1=sr_rd, df2=si_0)) /
                    (inner_product(df1=sr_td, df2=si_0) * inner_product(df1=sr_rd, df2=si_rd)))
    if not numpy.all(numpy.isfinite(M)):
        warnings.warn("Non-finite M detected.")
    if not numpy.all(M != 0):
        warnings.warn("Zero M detected.")

    return M

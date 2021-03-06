import warnings

import numpy
import scipy.constants
import scipy.interpolate

from pvfit.common.constants import c_m_per_s, h_J_s, q_C


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
        if y.shape[-1] != x_size:
            raise ValueError("last dimension of y must equal size of x.")
        self.x = x
        y = numpy.asarray_chkfinite(y, dtype=float)
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
    def __init__(self, *, lambda_nm: numpy.ndarray, QE: numpy.ndarray, is_percent: bool = False):
        super().__init__(x=lambda_nm, y=QE)
        # Do not convert raw data. Instead track if it is given as a percent.
        self.is_percent = is_percent

    @property
    def lambda_nm(self) -> numpy.ndarray:
        """Return wavelengths."""
        return self.x

    @property
    def QE(self) -> numpy.ndarray:
        """Return QE as fraction."""
        if self.is_percent:
            return self.y/100
        else:
            return self.y

    @property
    def QE_percent(self) -> numpy.ndarray:
        """Return QE as percent."""
        if self.is_percent:
            return self.y
        else:
            return 100*self.y

    @property
    def S_A_per_W(self) -> "SpectralResponsivity":
        """
        Convert quantum efficiency (QE) curve to spectral responsivity
        (SR) curve.

        TODO Describe interface.
        """
        return SpectralResponsivity(
            lambda_nm=self.lambda_nm, S_A_per_W=self.QE * self.lambda_nm * 1.e-9 * q_C / (h_J_s * c_m_per_s))


class SpectralIrradiance(DataFunctionPositiveXNonnegativeY):
    """
    Store data representing a spectral irradiance curve.

    TODO Describe interface and units [nm] and [A/W/m^2].
    """
    def __init__(self, *, lambda_nm: numpy.ndarray, E_W_per_m2_nm: numpy.ndarray):
        super().__init__(x=lambda_nm, y=E_W_per_m2_nm)

    @property
    def lambda_nm(self) -> numpy.ndarray:
        return self.x

    @property
    def E_W_per_m2_nm(self) -> numpy.ndarray:
        return self.y


class SpectralResponsivity(DataFunctionPositiveXNonnegativeY):
    """
    Store data representing a spectral responsivity (SR) curve.

    TODO Describe interface and units [nm] and [A/W].
    """
    def __init__(self, *, lambda_nm: numpy.ndarray, S_A_per_W: numpy.ndarray):
        super().__init__(x=lambda_nm, y=S_A_per_W)

    @property
    def lambda_nm(self) -> numpy.ndarray:
        return self.x

    @property
    def S_A_per_W(self) -> numpy.ndarray:
        return self.y

    @property
    def QE(self) -> "QuantumEfficiency":
        """
        Convert spectral responsivity (SR) curve to quantum efficiency
        (QE) curve.

        TODO Describe interface.
        """
        return QuantumEfficiency(
            lambda_nm=self.lambda_nm, QE=self.S_A_per_W * h_J_s * c_m_per_s / (self.lambda_nm * 1.e-9 * q_C))


def inner_product(*, f1: DataFunction, f2: DataFunction) -> numpy.ndarray:
    r"""
    Compute inner product of two data functions.

    The inner product of two data functions is the integral of the product
    of the two functions over their common domain of defintion. Because
    the data function model is piecewise linear, an algebraic solution
    exists and is used for the computation.  See the :class:`DataFunction`
    class for details on the model that informs the computation.

    Parameters
    ----------
    f1
        First data function.
    f2
        Second data function.

    Returns
    -------
    inner_product : numpy.ndarray
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
    x_min = numpy.maximum(f1.x[0], f2.x[0])
    x_max = numpy.minimum(f1.x[-1], f2.x[-1])
    if x_max <= x_min:
        warnings.warn("DataFunction domains do not overlap.")
        return numpy.zeros(numpy.broadcast(f1.y, f2.y).shape[:-1])
    x_union = numpy.union1d(f1.x, f2.x)
    x_union = x_union[numpy.logical_and(x_min <= x_union, x_union <= x_max)]
    y1 = scipy.interpolate.interp1d(f1.x, f1.y, copy=False, assume_sorted=True)(x_union)
    y2 = scipy.interpolate.interp1d(f2.x, f2.y, copy=False, assume_sorted=True)(x_union)

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


def M(*, S_TD_OC: SpectralResponsivity, E_TD_OC: SpectralIrradiance, S_TD_RC: SpectralResponsivity,
      E_TD_RC: SpectralIrradiance, S_RD_OC: SpectralResponsivity, E_RD_OC: SpectralIrradiance,
      S_RD_RC: SpectralResponsivity, E_RD_RC: SpectralIrradiance) -> numpy.ndarray:
    r"""
    Compute spectral mismatch correction factor (:math:`M`).

    The spectral mismatch is between a photovoltaic (PV) test device (TD)
    and a PV reference device (RD), each at a particular (non-explicit)
    temperature and illuminated by a (possibly different) spectral
    irradiance at operating condition (OC). The corresponding reference
    condition (RC) of each device need not be the same, but often are.
    :math:`M` should be strictly positive, but could evalute to
    be zero, infinite, or NaN depending on possible zero values of the
    component integrals. See the :class:`SpectralIrradiance` and
    :class:`SpectralResponsivity` classes for details on the data function
    models that inform the computation, which includes vectorized
    computations.

    Parameters
    ----------
    S_TD_OC
        Spectral responsivity of TD at OC [A/W].
    E_TD_OC
        Spectral irradiance illuminating TD at OC [W/m2/nm].
    S_TD_RC
        Spectral responsivity of TD at RC [A/W].
    E_TD_RC
        Spectral irradiance illuminating TD at RC [W/m2/nm].
    S_RD_OC
        Spectral responsivity of RD at OC [A/W].
    E_RD_OC
        Spectral irradiance illuminating RD at OC [W/m2/nm].
    S_RD_RC
        Spectral responsivity of RD at RC [A/W].
    E_RD_RC
        Spectral irradiance illuminating RD at RC [W/m2/nm].

    Returns
    -------
    M : numpy.ndarray
        Spectral mismatch correction factor (:math:`M`).

    Warns
    ------
    UserWarning
        If :math:`M` is computed as non-positive, infinite, or NaN.

    See Also
    --------
    inner_product : The function used to compute the integrals of the
        products of two data functions.

    Notes
    -----
    :math:`M` is defined by this relationship between the short-circuit
    currents (:math:`I_\mathrm{sc}`) of a TD and a RD at their
    respective OC and RC--

    .. math:: \frac{I_\mathrm{sc,TD,OC}}{I_\mathrm{sc,TD,RC}} =
        M \frac{I_\mathrm{sc,RD,OC}}{I_\mathrm{sc,RD,RC}},

    so that, under linearity and homogeneity assumption, :math:`M` is
    computed as--

    .. math:: M &= \frac{I_\mathrm{sc,TD,OC} I_\mathrm{sc,RD,RC}}
        {I_\mathrm{sc,TD,RC} I_\mathrm{sc,RD,OC}} \\
        &= \frac{
        \int_{\lambda=0}^\infty S_\mathrm{TD}(T_\mathrm{TD,OC}, \lambda)
        E_\mathrm{TD,OC}(\lambda) \, \mathrm{d}\lambda \,
        \int_{\lambda=0}^\infty S_\mathrm{RD}(T_\mathrm{RD,RC}, \lambda)
        E_\mathrm{RD,RC}(\lambda) \, \mathrm{d}\lambda}{
        \int_{\lambda=0}^\infty S_\mathrm{TD}(T_\mathrm{TD,RC}, \lambda)
        E_\mathrm{TD,RC}(\lambda) \, \mathrm{d}\lambda \,
        \int_{\lambda=0}^\infty S_\mathrm{RD}(T_\mathrm{RD,OC}, \lambda)
        E_\mathrm{RD,OC}(\lambda) \, \mathrm{d}\lambda},

    where any pertinent constant scaling factors cancel out between
    numerator and denominator, such as device areas, curve measurement
    scaling errors, and unit conversions [1]_.

    References
    ----------
    .. [1] M. Campanelli and B. Hamadani, "Calibration of a single‐diode
        performance model without a short‐circuit temperature
        coefficient," Energy Science & Engineering, vol. 6, no. 4,
        pp. 222-238, 2018. https://doi.org/10.1002/ese3.190.

    """
    # TODO Warn if computation appears innacurate due to missing non-zero data at end(s) of common domain intervals.
    M = numpy.array((inner_product(f1=S_TD_OC, f2=E_TD_OC) * inner_product(f1=S_RD_RC, f2=E_RD_RC)) /
                    (inner_product(f1=S_TD_RC, f2=E_TD_RC) * inner_product(f1=S_RD_OC, f2=E_RD_OC)))
    if not numpy.all(numpy.isfinite(M)):
        warnings.warn("Non-finite M detected.")
    if not numpy.all(0 < M):
        warnings.warn("Non-positive M detected.")

    return M

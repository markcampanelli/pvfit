"""
PVfit: Spectral correction computations.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import warnings

import numpy
import scipy.interpolate

from pvfit.measurement.spectral_correction.types import (
    DataFunction,
    SpectralIrradiance,
    SpectralResponsivity,
)


def inner_product(*, f1: DataFunction, f2: DataFunction) -> numpy.ndarray:
    r"""
    Compute inner product of two data functions.

    The inner product of two data functions is the integral of the product of the two
    functions over their common domain of defintion. Because the data function model is
    piecewise linear, an algebraic solution exists and is used for the computation.  See
    the :class:`DataFunction` class for details on the model that informs the
    computation.

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

    where the interval of integration :math:`[x_1, x_2]` is the common domain of the two
    data functions. If the domains do not overlap, then zero is returned.
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

    inner_product_ = numpy.array(
        numpy.sum(
            C * (x_union_cubed[1:] - x_union_cubed[:-1])
            + B * (x_union_squared[1:] - x_union_squared[:-1])
            + A * (x_union[1:] - x_union[:-1]),
            axis=-1,
        )
    )
    if not numpy.all(numpy.isfinite(inner_product_)):
        warnings.warn("Non-finite inner product detected.")

    return inner_product_


def M(
    *,
    S_TD_OC: SpectralResponsivity,
    E_TD_OC: SpectralIrradiance,
    S_TD_RC: SpectralResponsivity,
    E_TD_RC: SpectralIrradiance,
    S_RD_OC: SpectralResponsivity,
    E_RD_OC: SpectralIrradiance,
    S_RD_RC: SpectralResponsivity,
    E_RD_RC: SpectralIrradiance,
) -> numpy.ndarray:
    r"""
    Compute spectral mismatch correction factor (:math:`M`).

    The spectral mismatch is between a photovoltaic (PV) test device (TD) and a PV
    reference device (RD), each at a particular (non-explicit) temperature and
    illuminated by a (possibly different) spectral irradiance at operating condition
    (OC). The corresponding reference condition (RC) of each device need not be the
    same, but often are. :math:`M` should be strictly positive, but could evalute to be
    zero, infinite, or NaN depending on possible zero values of the component integrals.
    See the :class:`SpectralIrradiance` and :class:`SpectralResponsivity` classes for
    details on the data function models that inform the computation, which includes
    vectorized computations.

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
    inner_product : The function used to compute the integrals of the products of two
    data functions.

    Notes
    -----
    :math:`M` is defined by this relationship between the short-circuit currents
    (:math:`I_\mathrm{sc}`) of a TD and a RD at their respective OC and RC--

    .. math:: \frac{I_\mathrm{sc,TD,OC}}{I_\mathrm{sc,TD,RC}} = M \frac{I_\mathrm{sc,RD,OC}}{I_\mathrm{sc,RD,RC}},

    so that, under linearity and homogeneity assumption, :math:`M` is computed as--

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

    where any pertinent constant scaling factors cancel out between numerator and
    denominator, such as device areas, curve measurement scaling errors, and unit
    conversions [1]_.

    References
    ----------
    .. [1] M. Campanelli and B. Hamadani, "Calibration of a single‐diode performance
        model without a short‐circuit temperature coefficient," Energy Science &
        Engineering, vol. 6, no. 4, pp. 222-238, 2018. https://doi.org/10.1002/ese3.190.

    """
    # TODO Warn if computation appears innacurate due to missing non-zero data at end(s)
    # of common domain intervals.
    M_ = numpy.array(
        (inner_product(f1=S_TD_OC, f2=E_TD_OC) * inner_product(f1=S_RD_RC, f2=E_RD_RC))
        / (
            inner_product(f1=S_TD_RC, f2=E_TD_RC)
            * inner_product(f1=S_RD_OC, f2=E_RD_OC)
        )
    )
    if not numpy.all(numpy.isfinite(M_)):
        warnings.warn("Non-finite M detected.")
    if not numpy.all(0 < M_):
        warnings.warn("Non-positive M detected.")

    return M_

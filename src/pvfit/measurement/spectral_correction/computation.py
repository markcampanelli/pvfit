"""
PVfit: Spectral correction computations.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional, Union
import warnings

import numpy
import scipy.interpolate

from pvfit.measurement.spectral_correction.types import (
    DataFunction,
    FloatArray,
    SpectralIrradiance,
    SpectralIrradianceWithTail,
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


def extrapolate_spectral_irradiance(
    *,
    spectral_irradiance: SpectralIrradiance,
    extrapolant: Union[SpectralIrradiance, SpectralIrradianceWithTail],
    E_total_W_per_m2: Optional[FloatArray] = None,
) -> Union[SpectralIrradiance, SpectralIrradianceWithTail]:
    """
    Extend the ends of a given spectral irradiance curve(s) by a given complete spectral
    irradiance curve scaled so that the specified total irradiance is acheived.
    """
    # 1) Find left wavelength endpoint and interpolate extrapolation spectrum to that value.
    # 2) Find right wavelength endpoint and interpolate extrapolation spectrum to that value.
    # 3) Find total irradiance of interpolant.
    # 4) Find total irradiance of extrapolant (outside of interpolant).
    # 5) Scale extrapolant so that irradiance of interpolant + irradiance of extrapolant equals specified total.
    # 6) Create output spectral irradiance using combined result.

    if extrapolant.num_functions != 1:
        raise ValueError("must extrapolate using only one spectrum")

    lambda_min = spectral_irradiance.lambda_nm[0]
    lambda_max = spectral_irradiance.lambda_nm[-1]
    lambda_range = lambda_max - lambda_min

    if extrapolant.lambda_nm[0] > lambda_min or extrapolant.lambda_nm[-1] < lambda_max:
        raise ValueError(
            "spectral_irradiance domain not a subdomain of extrapolant domain"
        )

    extrapolant_interpolator = scipy.interpolate.interp1d(
        extrapolant.lambda_nm, extrapolant.E_W_per_m2_nm, copy=False
    )

    # Match integrals of lower overlap.
    lambda_lower_min = lambda_min
    fraction_lower = 0.075
    lambda_lower_max = spectral_irradiance.lambda_nm[
        numpy.searchsorted(
            spectral_irradiance.lambda_nm,
            min(lambda_min + fraction_lower * lambda_range, lambda_max),
            side="left",
        )
    ]
    E_total_subinterval_lower = spectral_irradiance.get_E_total_subinterval_W_per_m2(
        lambda_min_nm=lambda_lower_min, lambda_max_nm=lambda_lower_max
    )

    if extrapolant.lambda_nm[0] < lambda_min:
        # Lower portion of spectral_irradiance can be extended.
        # Create sub-spectra for overlapping portions and match integrals.
        lower_idx = numpy.logical_and(
            lambda_lower_min <= extrapolant.lambda_nm,
            extrapolant.lambda_nm <= lambda_lower_max,
        )
        lambda_lower = extrapolant.lambda_nm[lower_idx]
        E_lower = extrapolant.E_W_per_m2_nm[..., lower_idx]

        if lambda_lower_min not in lambda_lower:
            lambda_lower = numpy.insert(lambda_lower, 0, lambda_lower_min)
            E_lower = numpy.insert(
                E_lower, 0, extrapolant_interpolator(lambda_lower_min), axis=-1
            )

        if lambda_lower_max not in lambda_lower:
            lambda_lower = numpy.insert(lambda_lower, -1, lambda_lower_max)
            E_lower = numpy.insert(
                E_lower, -1, extrapolant_interpolator(lambda_lower_max), axis=-1
            )

        E_total_lower = SpectralIrradiance(
            lambda_nm=lambda_lower, E_W_per_m2_nm=E_lower
        ).E_total_W_per_m2

        extrapolant_scaling_lower = E_total_subinterval_lower / E_total_lower
    else:
        lambda_lower = numpy.empty([lambda_lower_min])
        E_lower = spectral_irradiance.E_W_per_m2_nm[..., 0]
        E_total_lower = numpy.zeros_like(E_lower)
        extrapolant_scaling_lower = 0.0

    # Match integrals of upper overlap.
    fraction_upper = 0.185
    lambda_upper_min = spectral_irradiance.lambda_nm[
        numpy.searchsorted(
            spectral_irradiance.lambda_nm,
            max(lambda_min, lambda_max - fraction_upper * lambda_range),
            side="right",
        )
        - 1
    ]
    lambda_upper_max = lambda_max
    E_total_subinterval_upper = spectral_irradiance.get_E_total_subinterval_W_per_m2(
        lambda_min_nm=lambda_upper_min, lambda_max_nm=lambda_upper_max
    )

    if lambda_max < extrapolant.lambda_nm[-1]:
        # Upper portion of spectral_irradiance can be extended.
        # Create sub-spectra for overlapping portions and match integrals.
        upper_idx = numpy.logical_and(
            lambda_upper_min <= extrapolant.lambda_nm,
            extrapolant.lambda_nm <= lambda_upper_max,
        )
        lambda_upper = extrapolant.lambda_nm[upper_idx]
        E_upper = extrapolant.E_W_per_m2_nm[..., upper_idx]

        if lambda_upper_min not in lambda_upper:
            lambda_upper = numpy.insert(lambda_upper, 0, lambda_upper_min)
            E_upper = numpy.insert(
                E_upper, 0, extrapolant_interpolator(lambda_upper_min), axis=-1
            )

        if lambda_upper_max not in lambda_upper:
            lambda_upper = numpy.insert(lambda_upper, -1, lambda_upper_max)
            E_upper = numpy.insert(
                E_upper, -1, extrapolant_interpolator(lambda_upper_max), axis=-1
            )

        E_total_upper = SpectralIrradiance(
            lambda_nm=lambda_upper, E_W_per_m2_nm=E_upper
        ).E_total_W_per_m2

        extrapolant_scaling_upper = E_total_subinterval_upper / E_total_upper
    else:
        lambda_upper = numpy.empty([lambda_upper_max])
        E_upper = spectral_irradiance.E_W_per_m2_nm[..., -1]
        E_total_upper = numpy.zeros_like(E_upper)
        extrapolant_scaling_upper = 0.0

    # Stitch together spectra.
    lambda_lower_tail_idx = extrapolant.lambda_nm < lambda_min
    lambda_upper_tail_idx = lambda_max < extrapolant.lambda_nm

    lambda_nm = numpy.concatenate(
        (
            extrapolant.lambda_nm[lambda_lower_tail_idx],
            spectral_irradiance.lambda_nm,
            extrapolant.lambda_nm[lambda_upper_tail_idx],
        )
    )

    lower_shape = (
        *spectral_irradiance.E_W_per_m2_nm.shape[:-1],
        numpy.count_nonzero(lambda_lower_tail_idx),
    )
    upper_shape = (
        *spectral_irradiance.E_W_per_m2_nm.shape[:-1],
        numpy.count_nonzero(lambda_upper_tail_idx),
    )

    E_W_per_m2_nm = numpy.concatenate(
        (
            numpy.expand_dims(extrapolant_scaling_lower, -1)
            * numpy.broadcast_to(
                extrapolant.E_W_per_m2_nm[lambda_lower_tail_idx], lower_shape
            ),
            spectral_irradiance.E_W_per_m2_nm,
            numpy.expand_dims(extrapolant_scaling_upper, -1)
            * numpy.broadcast_to(
                extrapolant.E_W_per_m2_nm[lambda_upper_tail_idx], upper_shape
            ),
        ),
        axis=-1,
    )

    if isinstance(extrapolant, SpectralIrradianceWithTail):
        spectral_irradiance_extended = SpectralIrradianceWithTail(
            lambda_nm=lambda_nm,
            E_W_per_m2_nm=E_W_per_m2_nm,
            E_tail_W_per_m2=extrapolant_scaling_upper * extrapolant.E_tail_W_per_m2,
        )
    else:
        spectral_irradiance_extended = SpectralIrradiance(
            lambda_nm=lambda_nm,
            E_W_per_m2_nm=E_W_per_m2_nm,
        )

    if E_total_W_per_m2 is not None:
        # Scale extended spectral irradiance to provided total irradiance.
        total_scaling = E_total_W_per_m2 / spectral_irradiance_extended.E_total_W_per_m2

        if isinstance(spectral_irradiance_extended, SpectralIrradianceWithTail):
            spectral_irradiance_extended = SpectralIrradianceWithTail(
                lambda_nm=spectral_irradiance_extended.lambda_nm,
                E_W_per_m2_nm=numpy.expand_dims(total_scaling, -1)
                * spectral_irradiance_extended.E_W_per_m2_nm,
                E_tail_W_per_m2=total_scaling
                * spectral_irradiance_extended.E_tail_W_per_m2,
            )
        else:
            spectral_irradiance_extended = SpectralIrradiance(
                lambda_nm=spectral_irradiance_extended.lambda_nm,
                E_W_per_m2_nm=numpy.expand_dims(total_scaling, -1)
                * spectral_irradiance_extended.E_W_per_m2_nm,
            )

    return spectral_irradiance_extended

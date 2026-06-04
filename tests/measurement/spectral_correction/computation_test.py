"""
PVfit testing: Spectral correction computations.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import sys

import numpy
import pytest

from pvfit.measurement.spectral_correction import computation
from pvfit.measurement.spectral_correction.types import DataFunction


def test_inner_product():
    """Test computation of inner product of two DataFunctions."""
    # Scalar-like computation, two constant lines.
    x1 = numpy.array([100, 200, 300])
    f1 = DataFunction(x=x1, y=numpy.ones_like(x1))
    x2 = numpy.array([50, 150, 250])
    f2 = DataFunction(x=x2, y=numpy.full_like(x2, 2))
    inner_product_expected = 2.0 * (250 - 100)
    inner_product = computation.inner_product(f1=f1, f2=f2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, computation.inner_product(f1=f2, f2=f1))

    # Scalar-like computation, two non-constant lines.
    x1 = numpy.array([100, 200, 300])
    f1 = DataFunction(x=x1, y=numpy.array([0, 1, 2]))
    x2 = numpy.array([50, 150, 250, 350])
    f2 = DataFunction(x=x2, y=numpy.array([3, 2, 1, 0]))
    inner_product_expected = (
        -(300**3 - 100**3) / 30000
        + 45 / 2000 * (300**2 - 100**2)
        - 35 / 10 * (300 - 100)
    )
    inner_product = computation.inner_product(f1=f1, f2=f2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, computation.inner_product(f1=f2, f2=f1))

    # Compatible vectorized computation, time-series like.
    f1 = DataFunction(x=x1, y=numpy.array([[0, 1, 2], [0, 1, 2]]))
    f2 = DataFunction(x=x2, y=numpy.array([[3, 2, 1, 0], [3, 2, 1, 0]]))
    inner_product = computation.inner_product(f1=f1, f2=f2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, (2,))
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, computation.inner_product(f1=f2, f2=f1))

    # Compatible vectorized computation, table like.
    f1 = DataFunction(
        x=x1, y=numpy.array([[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])
    )
    f2 = DataFunction(
        x=x2,
        y=numpy.array([[[3, 2, 1, 0], [3, 2, 1, 0]], [[3, 2, 1, 0], [3, 2, 1, 0]]]),
    )
    inner_product = computation.inner_product(f1=f1, f2=f2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, (2, 2))
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, computation.inner_product(f1=f2, f2=f1))

    # Incompatible vectorized computation because of shape mismatch in multi-curves.
    f1 = DataFunction(
        x=x1,
        y=numpy.array(
            [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]]
        ),
    )
    f2 = DataFunction(
        x=x2,
        y=numpy.array([[[3, 2, 1, 0], [3, 2, 1, 0]], [[3, 2, 1, 0], [3, 2, 1, 0]]]),
    )
    with pytest.raises(ValueError):
        # Cannot broadcast in computation.
        inner_product = computation.inner_product(f1=f1, f2=f2)
    # Non-overlapping domains.
    # No broadcast case.
    x1 = numpy.array([200, 300])
    f1 = DataFunction(x=x1, y=numpy.array([1, 2]))
    x2 = numpy.array([50, 150])
    f2 = DataFunction(x=x2, y=numpy.array([3, 2]))
    inner_product_expected = 0.0
    with pytest.warns(Warning) as record:
        inner_product = computation.inner_product(f1=f1, f2=f2)
    assert len(record) == 1
    assert record[0].message.args[0] == "DataFunction domains do not overlap."
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_equal(inner_product, inner_product_expected)
    # Broadcast case.
    x1 = numpy.array([200, 300])
    f1 = DataFunction(x=x1, y=numpy.array([1, 2]))
    x2 = numpy.array([50, 150])
    f2 = DataFunction(x=x2, y=numpy.array([[3, 2], [5, 7]]))
    inner_product_expected = numpy.zeros((2,))
    with pytest.warns(Warning) as record:
        inner_product = computation.inner_product(f1=f1, f2=f2)
    assert len(record) == 1
    assert record[0].message.args[0] == "DataFunction domains do not overlap."
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, (2,))
    numpy.testing.assert_equal(inner_product, inner_product_expected)
    # Infinite computation.
    f = DataFunction(
        x=numpy.array([0, 1]), y=numpy.array([sys.float_info.max, sys.float_info.max])
    )
    with pytest.warns(Warning) as record:
        inner_product = computation.inner_product(f1=f, f2=f)
    assert len(record) == 2
    assert record[0].message.args[0] == "overflow encountered in multiply"
    assert record[1].message.args[0] == "Non-finite inner product detected."
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_equal(inner_product, numpy.inf)


def test_M():
    """Test computation of M."""
    lambda_nm = numpy.array([100, 200, 300], dtype=float)
    # Scalar-like computation.
    shape = ()
    S_TD = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 2)
    )
    E_TD = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    S_RD = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 0.5)
    )
    E_RD = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    E_RC = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    M_expected = numpy.ones(shape)
    M = computation.M(
        S_TD_OC=S_TD,
        E_TD_OC=E_TD,
        S_TD_RC=S_TD,
        E_TD_RC=E_RC,
        S_RD_OC=S_RD,
        E_RD_OC=E_RD,
        S_RD_RC=S_RD,
        E_RD_RC=E_RC,
    )
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Vectorized computation, time-series like.
    shape = (2,)
    S_TD = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 2)
    )
    E_TD = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.ones((2, len(lambda_nm)))
    )
    S_RD = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 0.5)
    )
    E_RD = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    E_RC = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    M_expected = numpy.ones(shape)
    M = computation.M(
        S_TD_OC=S_TD,
        E_TD_OC=E_TD,
        S_TD_RC=S_TD,
        E_TD_RC=E_RC,
        S_RD_OC=S_RD,
        E_RD_OC=E_RD,
        S_RD_RC=S_RD,
        E_RD_RC=E_RC,
    )
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Vectorized computation, table like.
    shape = (2, 2)
    S_TD = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=2 * numpy.ones((2, 2, len(lambda_nm)))
    )
    E_TD = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    S_RD = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 0.5)
    )
    E_RD = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    E_RC = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    M_expected = numpy.ones(shape)
    M = computation.M(
        S_TD_OC=S_TD,
        E_TD_OC=E_TD,
        S_TD_RC=S_TD,
        E_TD_RC=E_RC,
        S_RD_OC=S_RD,
        E_RD_OC=E_RD,
        S_RD_RC=S_RD,
        E_RD_RC=E_RC,
    )
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Scaling invariance.
    shape = ()
    S_TD_1 = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 2)
    )
    E_TD = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    S_RD = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 0.5)
    )
    E_RD = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    E_RC = computation.SpectralIrradiance(
        lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1)
    )
    # Scale S_TD_1 by 1/2.
    S_TD_2 = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 1)
    )
    M_expected = numpy.ones(shape)
    M_1 = computation.M(
        S_TD_OC=S_TD_1,
        E_TD_OC=E_TD,
        S_TD_RC=S_TD_1,
        E_TD_RC=E_RC,
        S_RD_OC=S_RD,
        E_RD_OC=E_RD,
        S_RD_RC=S_RD,
        E_RD_RC=E_RC,
    )
    M_2 = computation.M(
        S_TD_OC=S_TD_2,
        E_TD_OC=E_TD,
        S_TD_RC=S_TD_2,
        E_TD_RC=E_RC,
        S_RD_OC=S_RD,
        E_RD_OC=E_RD,
        S_RD_RC=S_RD,
        E_RD_RC=E_RC,
    )
    numpy.testing.assert_equal(M_1, M_2)

    # Infinite Integrals.
    # Infinite M.
    S_TD = computation.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, sys.float_info.max)
    )
    with pytest.warns(Warning) as record:
        M = computation.M(
            S_TD_OC=S_TD,
            E_TD_OC=E_TD,
            S_TD_RC=S_TD_1,
            E_TD_RC=E_RC,
            S_RD_OC=S_RD,
            E_RD_OC=E_RD,
            S_RD_RC=S_RD,
            E_RD_RC=E_RC,
        )
    assert len(record) == 3
    assert record[0].message.args[0] == "overflow encountered in multiply"
    assert record[1].message.args[0] == "Non-finite inner product detected."
    assert record[2].message.args[0] == "Non-finite M detected."
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, numpy.inf)
    # NaN M.
    with pytest.warns(Warning) as record:
        M = computation.M(
            S_TD_OC=S_TD,
            E_TD_OC=E_TD,
            S_TD_RC=S_TD,
            E_TD_RC=E_RC,
            S_RD_OC=S_RD,
            E_RD_OC=E_RD,
            S_RD_RC=S_RD,
            E_RD_RC=E_RC,
        )
    assert record[0].message.args[0] == "overflow encountered in multiply"
    assert record[1].message.args[0] == "Non-finite inner product detected."
    assert record[2].message.args[0] == "overflow encountered in multiply"
    assert record[3].message.args[0] == "Non-finite inner product detected."
    assert record[4].message.args[0] == "invalid value encountered in scalar divide"
    assert record[5].message.args[0] == "Non-finite M detected."
    assert record[6].message.args[0] == "Non-positive M detected."
    assert len(record) == 7
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, numpy.nan)

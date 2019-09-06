import os
import sys

import numpy
import pytest

import pvfit_m


# TODO Test all classes, including more tests for SR->QE and QE->SR conversions.


def test_constants():
    """Know when constants change."""
    assert pvfit_m.api.q_C == 1.6021766208e-19
    assert pvfit_m.api.c_m_per_s == 299792458.0
    assert pvfit_m.api.h_J_s == 6.62607004e-34


def test_DataFunction():
    """Test DataFunction class."""
    # Invaid data cases.
    x = numpy.array([])
    y = numpy.array([0, 0.5, 1.])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "x must have at least one element." in str(excinfo.value)

    x = numpy.array([[100, 200, 300], [100, 200, 300]])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "x cannot have dimension greater than one." in str(excinfo.value)

    x = numpy.array([100, 200, 100])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "x values must be unique." in str(excinfo.value)

    x = numpy.array([100, 200, numpy.inf])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "array must not contain infs or NaNs" in str(excinfo.value)

    x = numpy.array([100, 200, numpy.nan])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "array must not contain infs or NaNs" in str(excinfo.value)

    x = numpy.array([100, 200, 300])
    y = numpy.array([0, numpy.inf, 1.])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "array must not contain infs or NaNs" in str(excinfo.value)

    y = numpy.array([numpy.nan, 0.5, 1.])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "array must not contain infs or NaNs" in str(excinfo.value)

    y = numpy.array([0, 0.5])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "last dimension of y must equal size of x." in str(excinfo.value)

    y = numpy.array([0, 0.5, 1., 1.5])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunction(x=x, y=y)
    assert "last dimension of y must equal size of x." in str(excinfo.value)

    x = numpy.array([100, 200, 300])
    # 1D case
    y = numpy.array([0, 0.5, 1.])
    f = pvfit_m.api.DataFunction(x=x, y=y)
    f_duplicate = pvfit_m.api.DataFunction(x=x, y=y)
    assert f == f_duplicate
    # 2D case
    y = numpy.array([[0, 0.5, 1.],
                     [1, 1.5, 2.]])
    f = pvfit_m.api.DataFunction(x=x, y=y)
    f_duplicate = pvfit_m.api.DataFunction(x=x, y=y)
    assert f == f_duplicate
    # 3D case
    y = numpy.array([[[0, 0.5, 1.],
                      [1, 1.5, 2.]],
                     [[1, 1.5, 2.],
                      [0, 0.5, 1.]]])
    f = pvfit_m.api.DataFunction(x=x, y=y)
    f_duplicate = pvfit_m.api.DataFunction(x=x, y=y)
    assert f == f_duplicate

    # Check assignment with sorting.
    # 1D case
    x = numpy.array([300, 200., 100])
    y = numpy.array([1., 0.5, 0])
    f = pvfit_m.api.DataFunction(x=x, y=y)
    numpy.testing.assert_array_equal(f.x, numpy.flip(x))
    numpy.testing.assert_array_equal(f.y, numpy.flip(y))
    # 2D case
    x = numpy.array([300, 200., 100])
    y = numpy.array([[1., 0.5, 0],
                     [2., 1.5, 1]])
    f = pvfit_m.api.DataFunction(x=x, y=y)
    numpy.testing.assert_array_equal(f.x, numpy.flip(x))
    numpy.testing.assert_array_equal(f.y, numpy.fliplr(y))


def test_DataFunctionPositiveXNonnegativeY():
    """Test DataFunctionPositiveXNonnegativeY class."""
    # Invaid data cases.
    # x
    x = numpy.array([0, 100, 200])
    y = numpy.array([0, 0.5, 1.])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunctionPositiveXNonnegativeY(x=x, y=y)
    assert "x values must all be positive." in str(excinfo.value)
    # y
    x = numpy.array([100, 200])
    y = numpy.array([-0.5, 1.])
    with pytest.raises(ValueError) as excinfo:
        pvfit_m.api.DataFunctionPositiveXNonnegativeY(x=x, y=y)
    assert "y values must all be non-negative." in str(excinfo.value)


def test_QuantumEfficiency():
    """Test QuantumEfficiency class."""
    # Assignment.
    lambda_nm = numpy.array([100, 200, 300])
    QE = numpy.array([0, 0.5, 1.])
    QE_fraction = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, QE=QE)
    numpy.testing.assert_array_equal(QE_fraction.lambda_nm, lambda_nm)
    numpy.testing.assert_array_equal(QE_fraction.QE, QE)
    # Should not alter underlying QE data representation provided by user.
    QE_percent = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, QE=numpy.array([0, 50, 100]), is_percent=True)
    assert QE_fraction != QE_percent
    numpy.testing.assert_equal(QE_fraction.x, QE_percent.x)
    numpy.testing.assert_equal(numpy.equal(QE_fraction.y, QE_percent.y), numpy.array([True, False, False]))
    numpy.testing.assert_equal(100*QE_fraction.y, QE_percent.y)
    numpy.testing.assert_equal(QE_fraction.lambda_nm, QE_percent.lambda_nm)
    numpy.testing.assert_equal(QE_fraction.QE, QE_percent.QE)
    numpy.testing.assert_equal(QE_fraction.QE_percent, QE_percent.QE_percent)
    # Test conversions from QuantumEfficiency to SpectralResponivity.
    # Indifference to fraction vs. percent representation.
    QE_fraction = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, QE=numpy.array([0, 0.5, 1.]))
    QE_percent = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, QE=numpy.array([0, 50, 100]), is_percent=True)
    assert QE_fraction.S_A_per_W == QE_percent.S_A_per_W
    # Round trip identity function.
    QE_fraction_round_trip = QE_fraction.S_A_per_W.QE
    numpy.testing.assert_equal(QE_fraction_round_trip.lambda_nm, QE_fraction.lambda_nm)
    numpy.testing.assert_almost_equal(QE_fraction_round_trip.QE, QE_fraction.QE)
    qe_percent_round_trip = QE_percent.S_A_per_W.QE
    numpy.testing.assert_equal(qe_percent_round_trip.lambda_nm, QE_percent.lambda_nm)
    numpy.testing.assert_almost_equal(qe_percent_round_trip.QE, QE_percent.QE)


def test_SpectralIrradiance():
    """Test SpectralIrradiance class."""
    # Assignment.
    lambda_nm = numpy.array([100., 200., 300.])
    E_W_per_m2_nm = numpy.array([0, 0.5, 1.])
    E = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=E_W_per_m2_nm)
    numpy.testing.assert_array_equal(E.lambda_nm, lambda_nm)
    numpy.testing.assert_array_equal(E.E_W_per_m2_nm, E_W_per_m2_nm)


def test_SpectralResponivity():
    """Test SpectralResponivity class."""
    # Assignment.
    lambda_nm = numpy.array([100., 200., 300.])
    S_A_per_W = numpy.array([0, 0.5, 1.])
    S = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=S_A_per_W)
    numpy.testing.assert_array_equal(S.lambda_nm, lambda_nm)
    numpy.testing.assert_array_equal(S.S_A_per_W, S_A_per_W)
    # Test conversions from SpectralResponivity to QuantumEfficiency.
    # Round trip identity function.
    S_round_trip = S.QE.S_A_per_W
    numpy.testing.assert_equal(S_round_trip.lambda_nm, S.lambda_nm)
    numpy.testing.assert_almost_equal(S_round_trip.S_A_per_W, S.S_A_per_W)


def test_inner_product():
    """Test computation of inner product of two DataFunctions."""
    # Scalar-like computation, two constant lines.
    x1 = numpy.array([100, 200, 300])
    f1 = pvfit_m.api.DataFunction(x=x1, y=numpy.ones_like(x1))
    x2 = numpy.array([50, 150, 250])
    f2 = pvfit_m.api.DataFunction(x=x2, y=numpy.full_like(x2, 2))
    inner_product_expected = 2. * (250 - 100)
    inner_product = pvfit_m.api.inner_product(f1=f1, f2=f2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, pvfit_m.api.inner_product(f1=f2, f2=f1))

    # Scalar-like computation, two non-constant lines.
    x1 = numpy.array([100, 200, 300])
    f1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([0, 1, 2]))
    x2 = numpy.array([50, 150, 250, 350])
    f2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([3, 2, 1, 0]))
    inner_product_expected = -(300**3 - 100**3) / 30000 + 45 / 2000 * (300**2 - 100**2) - 35 / 10 * (300 - 100)
    inner_product = pvfit_m.api.inner_product(f1=f1, f2=f2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, pvfit_m.api.inner_product(f1=f2, f2=f1))

    # Compatible vectorized computation, time-series like.
    f1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([[0, 1, 2],
                                                       [0, 1, 2]]))
    f2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([[3, 2, 1, 0],
                                                       [3, 2, 1, 0]]))
    inner_product = pvfit_m.api.inner_product(f1=f1, f2=f2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, (2,))
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, pvfit_m.api.inner_product(f1=f2, f2=f1))

    # Compatible vectorized computation, table like.
    f1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([[[0, 1, 2],
                                                        [0, 1, 2]],
                                                       [[0, 1, 2],
                                                        [0, 1, 2]]]))
    f2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([[[3, 2, 1, 0],
                                                        [3, 2, 1, 0]],
                                                       [[3, 2, 1, 0],
                                                        [3, 2, 1, 0]]]))
    inner_product = pvfit_m.api.inner_product(f1=f1, f2=f2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, (2, 2))
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, pvfit_m.api.inner_product(f1=f2, f2=f1))

    # Incompatible vectorized computation because of shape mismatch in multi-curves.
    f1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([[[0, 1, 2],
                                                        [0, 1, 2]],
                                                       [[0, 1, 2],
                                                        [0, 1, 2]],
                                                       [[0, 1, 2],
                                                        [0, 1, 2]]]))
    f2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([[[3, 2, 1, 0],
                                                        [3, 2, 1, 0]],
                                                       [[3, 2, 1, 0],
                                                        [3, 2, 1, 0]]]))
    with pytest.raises(ValueError):
        # Cannot broadcast in computation.
        inner_product = pvfit_m.api.inner_product(f1=f1, f2=f2)
    # Non-overlapping domains.
    # No broadcast case.
    x1 = numpy.array([200, 300])
    f1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([1, 2]))
    x2 = numpy.array([50, 150])
    f2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([3, 2]))
    inner_product_expected = 0.
    with pytest.warns(Warning) as record:
        inner_product = pvfit_m.api.inner_product(f1=f1, f2=f2)
    assert len(record) == 1
    assert record[0].message.args[0] == "DataFunction domains do not overlap."
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_equal(inner_product, inner_product_expected)
    # Broadcast case.
    x1 = numpy.array([200, 300])
    f1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([1, 2]))
    x2 = numpy.array([50, 150])
    f2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([[3, 2], [5, 7]]))
    inner_product_expected = numpy.zeros((2,))
    with pytest.warns(Warning) as record:
        inner_product = pvfit_m.api.inner_product(f1=f1, f2=f2)
    assert len(record) == 1
    assert record[0].message.args[0] == "DataFunction domains do not overlap."
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, (2, ))
    numpy.testing.assert_equal(inner_product, inner_product_expected)
    # Infinite computation.
    f = pvfit_m.api.DataFunction(x=numpy.array([0, 1]), y=numpy.array([sys.float_info.max, sys.float_info.max]))
    with pytest.warns(Warning) as record:
        inner_product = pvfit_m.api.inner_product(f1=f, f2=f)
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
    S_TD = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 2))
    E_TD = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    S_RD = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 0.5))
    E_RD = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    E_RC = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    M_expected = numpy.ones(shape)
    M = pvfit_m.api.M(S_TD_OC=S_TD, E_TD_OC=E_TD, S_TD_RC=S_TD, E_TD_RC=E_RC, S_RD_OC=S_RD, E_RD_OC=E_RD,
                      S_RD_RC=S_RD, E_RD_RC=E_RC)
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Vectorized computation, time-series like.
    shape = (2,)
    S_TD = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 2))
    E_TD = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.ones((2, len(lambda_nm))))
    S_RD = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 0.5))
    E_RD = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    E_RC = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    M_expected = numpy.ones(shape)
    M = pvfit_m.api.M(S_TD_OC=S_TD, E_TD_OC=E_TD, S_TD_RC=S_TD, E_TD_RC=E_RC, S_RD_OC=S_RD, E_RD_OC=E_RD,
                      S_RD_RC=S_RD, E_RD_RC=E_RC)
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Vectorized computation, table like.
    shape = (2, 2)
    S_TD = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=2*numpy.ones((2, 2, len(lambda_nm))))
    E_TD = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    S_RD = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 0.5))
    E_RD = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    E_RC = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    M_expected = numpy.ones(shape)
    M = pvfit_m.api.M(S_TD_OC=S_TD, E_TD_OC=E_TD, S_TD_RC=S_TD, E_TD_RC=E_RC, S_RD_OC=S_RD, E_RD_OC=E_RD,
                      S_RD_RC=S_RD, E_RD_RC=E_RC)
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Scaling invariance.
    shape = ()
    S_TD_1 = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 2))
    E_TD = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    S_RD = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 0.5))
    E_RD = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    E_RC = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    # Scale S_TD_1 by 1/2.
    S_TD_2 = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, 1))
    M_expected = numpy.ones(shape)
    M_1 = pvfit_m.api.M(S_TD_OC=S_TD_1, E_TD_OC=E_TD, S_TD_RC=S_TD_1, E_TD_RC=E_RC, S_RD_OC=S_RD, E_RD_OC=E_RD,
                        S_RD_RC=S_RD, E_RD_RC=E_RC)
    M_2 = pvfit_m.api.M(S_TD_OC=S_TD_2, E_TD_OC=E_TD, S_TD_RC=S_TD_2, E_TD_RC=E_RC, S_RD_OC=S_RD, E_RD_OC=E_RD,
                        S_RD_RC=S_RD, E_RD_RC=E_RC)
    numpy.testing.assert_equal(M_1, M_2)

    # Infinite Integrals.
    # Infinite M.
    S_TD = pvfit_m.api.SpectralResponsivity(
        lambda_nm=lambda_nm, S_A_per_W=numpy.full_like(lambda_nm, sys.float_info.max))
    with pytest.warns(Warning) as record:
        M = pvfit_m.api.M(S_TD_OC=S_TD, E_TD_OC=E_TD, S_TD_RC=S_TD_1, E_TD_RC=E_RC, S_RD_OC=S_RD, E_RD_OC=E_RD,
                          S_RD_RC=S_RD, E_RD_RC=E_RC)
    assert len(record) == 3
    assert record[0].message.args[0] == "overflow encountered in multiply"
    assert record[1].message.args[0] == "Non-finite inner product detected."
    assert record[2].message.args[0] == "Non-finite M detected."
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, numpy.inf)
    # NaN M.
    with pytest.warns(Warning) as record:
        M = pvfit_m.api.M(S_TD_OC=S_TD, E_TD_OC=E_TD, S_TD_RC=S_TD, E_TD_RC=E_RC, S_RD_OC=S_RD, E_RD_OC=E_RD,
                          S_RD_RC=S_RD, E_RD_RC=E_RC)
    assert record[0].message.args[0] == "overflow encountered in multiply"
    assert record[1].message.args[0] == "Non-finite inner product detected."
    assert record[2].message.args[0] == "overflow encountered in multiply"
    assert record[3].message.args[0] == "Non-finite inner product detected."
    assert record[4].message.args[0] == "invalid value encountered in double_scalars"
    assert record[5].message.args[0] == "Non-finite M detected."
    if os.name != 'nt':
        # Not running on Windows.
        assert record[6].message.args[0] == "Non-positive M detected."
        assert len(record) == 7
    else:
        # An extra warning on Windows!
        assert record[6].message.args[0] == "invalid value encountered in greater"
        assert record[7].message.args[0] == "Non-positive M detected."
        assert len(record) == 8
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, numpy.nan)

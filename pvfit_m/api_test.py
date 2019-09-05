import numpy
import pytest

import pvfit_m


# TODO Test all classes, including more tests for SR->QE and QE->SR conversions.


def test_constants():
    """Know when the constants change."""
    assert pvfit_m.api.q_C == 1.6021766208e-19
    assert pvfit_m.api.c_m_per_s == 299792458.0
    assert pvfit_m.api.h_J_s == 6.62607004e-34


def test_DataFunction():
    """Test DataFunction class."""
    x = numpy.array([100, 200, 300])
    # 1D case
    y = numpy.array([0, 0.5, 1.])
    dc = pvfit_m.api.DataFunction(x=x, y=y)
    dc_duplicate = pvfit_m.api.DataFunction(x=x, y=y)
    assert dc == dc_duplicate
    # 2D case
    y = numpy.array([[0, 0.5, 1.],
                     [1, 1.5, 2.]])
    dc = pvfit_m.api.DataFunction(x=x, y=y)
    dc_duplicate = pvfit_m.api.DataFunction(x=x, y=y)
    assert dc == dc_duplicate
    # 3D case
    y = numpy.array([[[0, 0.5, 1.],
                      [1, 1.5, 2.]],
                     [[1, 1.5, 2.],
                      [0, 0.5, 1.]]])
    dc = pvfit_m.api.DataFunction(x=x, y=y)
    dc_duplicate = pvfit_m.api.DataFunction(x=x, y=y)
    assert dc == dc_duplicate


def test_QuantumEfficiency():
    """Test QE class."""
    lambda_nm = numpy.array([100, 200, 300])
    qe = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, qe=numpy.array([0, 0.5, 1.]))
    qe_percent = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, qe=numpy.array([0, 50, 100]), is_percent=True)
    # Should not alter underlying QE data representation provided by user.
    assert qe != qe_percent
    numpy.testing.assert_equal(qe.x, qe_percent.x)
    numpy.testing.assert_equal(numpy.equal(qe.y, qe_percent.y), numpy.array([True, False, False]))
    numpy.testing.assert_equal(100*qe.y, qe_percent.y)
    numpy.testing.assert_equal(qe.lambda_nm, qe_percent.lambda_nm)
    numpy.testing.assert_equal(qe.qe, qe_percent.qe)
    numpy.testing.assert_equal(qe.qe_percent, qe_percent.qe_percent)


def test_inner_product():
    """Test computation of inner product of two DataFunctions."""
    # Scalar-like computation, two constant lines.
    x1 = numpy.array([100, 200, 300])
    df1 = pvfit_m.api.DataFunction(x=x1, y=numpy.ones_like(x1))
    x2 = numpy.array([50, 150, 250])
    df2 = pvfit_m.api.DataFunction(x=x2, y=numpy.full_like(x2, 2))
    inner_product_expected = 2. * (250 - 100)
    inner_product = pvfit_m.api.inner_product(df1=df1, df2=df2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, pvfit_m.api.inner_product(df1=df2, df2=df1))

    # Scalar-like computation, two non-constant lines.
    x1 = numpy.array([100, 200, 300])
    df1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([0, 1, 2]))
    x2 = numpy.array([50, 150, 250, 350])
    df2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([3, 2, 1, 0]))
    inner_product_expected = -(300**3 - 100**3) / 30000 + 45 / 2000 * (300**2 - 100**2) - 35 / 10 * (300 - 100)
    inner_product = pvfit_m.api.inner_product(df1=df1, df2=df2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, ())
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, pvfit_m.api.inner_product(df1=df2, df2=df1))

    # Compatible vectorized computation, time-series like.
    df1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([[0, 1, 2],
                                                     [0, 1, 2]]))
    df2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([[3, 2, 1, 0],
                                                     [3, 2, 1, 0]]))
    inner_product = pvfit_m.api.inner_product(df1=df1, df2=df2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, (2,))
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, pvfit_m.api.inner_product(df1=df2, df2=df1))

    # Compatible vectorized computation, table like.
    df1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([[[0, 1, 2],
                                                      [0, 1, 2]],
                                                     [[0, 1, 2],
                                                      [0, 1, 2]]]))
    df2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([[[3, 2, 1, 0],
                                                      [3, 2, 1, 0]],
                                                     [[3, 2, 1, 0],
                                                      [3, 2, 1, 0]]]))
    inner_product = pvfit_m.api.inner_product(df1=df1, df2=df2)
    assert isinstance(inner_product, numpy.ndarray)
    numpy.testing.assert_equal(inner_product.shape, (2, 2))
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    numpy.testing.assert_equal(inner_product, pvfit_m.api.inner_product(df1=df2, df2=df1))

    # Incompatible vectorized computation because of shape mismatch in multi-curves.
    df1 = pvfit_m.api.DataFunction(x=x1, y=numpy.array([[[0, 1, 2],
                                                      [0, 1, 2]],
                                                     [[0, 1, 2],
                                                      [0, 1, 2]],
                                                     [[0, 1, 2],
                                                      [0, 1, 2]]]))
    df2 = pvfit_m.api.DataFunction(x=x2, y=numpy.array([[[3, 2, 1, 0],
                                                      [3, 2, 1, 0]],
                                                     [[3, 2, 1, 0],
                                                      [3, 2, 1, 0]]]))
    with pytest.raises(ValueError):
        # Cannot broadcast in computation.
        inner_product = pvfit_m.api.inner_product(df1=df1, df2=df2)


def test_m():
    """Test computation of M."""
    lambda_nm = numpy.array([100, 200, 300], dtype=float)
    # Scalar-like computation.
    shape = ()
    sr_td = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 2))
    si_td = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    sr_rd = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 0.5))
    si_rd = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    si_0 = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    M_expected = numpy.ones(shape)
    M = pvfit_m.api.m(sr_td=sr_td, si_td=si_td, sr_rd=sr_rd, si_rd=si_rd, si_0=si_0)
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Vectorized computation, time-series like.
    shape = (2,)
    sr_td = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 2))
    si_td = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.ones((2, len(lambda_nm))))
    sr_rd = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 0.5))
    si_rd = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    si_0 = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    M_expected = numpy.ones(shape)
    M = pvfit_m.api.m(sr_td=sr_td, si_td=si_td, sr_rd=sr_rd, si_rd=si_rd, si_0=si_0)
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Vectorized computation, table like.
    shape = (2, 2)
    sr_td = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=2*numpy.ones((2, 2, len(lambda_nm))))
    si_td = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    sr_rd = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 0.5))
    si_rd = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    si_0 = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    M_expected = numpy.ones(shape)
    M = pvfit_m.api.m(sr_td=sr_td, si_td=si_td, sr_rd=sr_rd, si_rd=si_rd, si_0=si_0)
    assert isinstance(M, numpy.ndarray)
    numpy.testing.assert_equal(M.shape, shape)
    numpy.testing.assert_almost_equal(M, M_expected)

    # Scaling invariance.
    shape = ()
    sr_td_1 = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 2))
    si_td = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    sr_rd = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 0.5))
    si_rd = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    si_0 = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    # Scale sr_td_1 by 1/2.
    sr_td_2 = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 1))
    M_expected = numpy.ones(shape)
    m_1 = pvfit_m.api.m(sr_td=sr_td_1, si_td=si_td, sr_rd=sr_rd, si_rd=si_rd, si_0=si_0)
    m_2 = pvfit_m.api.m(sr_td=sr_td_2, si_td=si_td, sr_rd=sr_rd, si_rd=si_rd, si_0=si_0)
    numpy.testing.assert_equal(m_1, m_2)


def test_qe_to_sr():
    """Test conversions from QE to SR."""
    lambda_nm = numpy.array([100, 200, 300])
    # Indifference to fraction vs. percent representation.
    qe = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, qe=numpy.array([0, 0.5, 1.]))
    qe_percent = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, qe=numpy.array([0, 50, 100]), is_percent=True)
    assert qe.sr_A_per_W == qe_percent.sr_A_per_W
    # Round trip identity function.
    qe_fraction_round_trip = qe.sr_A_per_W.qe
    numpy.testing.assert_equal(qe_fraction_round_trip.lambda_nm, qe.lambda_nm)
    numpy.testing.assert_almost_equal(qe_fraction_round_trip.qe, qe.qe)
    qe_percent_round_trip = qe_percent.sr_A_per_W.qe
    numpy.testing.assert_equal(qe_percent_round_trip.lambda_nm, qe_percent.lambda_nm)
    numpy.testing.assert_almost_equal(qe_percent_round_trip.qe, qe_percent.qe)


def test_sr_to_qe():
    """Test conversions from SR to QE."""
    lambda_nm = numpy.array([100, 200, 300])
    sr = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.array([0, 0.5, 1.]))
    # Round trip identity function.
    sr_round_trip = sr.qe.sr_A_per_W
    numpy.testing.assert_equal(sr_round_trip.lambda_nm, sr.lambda_nm)
    numpy.testing.assert_almost_equal(sr_round_trip.sr_A_per_W, sr.sr_A_per_W)

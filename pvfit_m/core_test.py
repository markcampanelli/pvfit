import numpy

import pvfit_m


# TODO Test all classes.
# TODO More tests for SR to QE conversion.
# TODO More tests for QE to SR conversion.


def test_constants():
    """Know when the constants change."""
    assert pvfit_m.core.q_C == 1.6021766208e-19
    assert pvfit_m.core.c_m_per_s == 299792458.0
    assert pvfit_m.core.h_J_s == 6.62607004e-34


def test_DataCurve():
    """Test DataCurve class."""
    x = numpy.array([100, 200, 300])
    y = numpy.array([0, 0.5, 1.])
    # Indifference to fraction vs. percent representation.
    dc = pvfit_m.core.DataCurve(x=x, y=y)
    dc_duplicate = pvfit_m.core.DataCurve(x=x, y=y)
    assert dc == dc_duplicate


def test_QuantumEfficiency():
    """Test QE class."""
    lambda_ = numpy.array([100, 200, 300])
    qe_fraction = pvfit_m.core.QuantumEfficiency(lambda_=lambda_, qe=numpy.array([0, 0.5, 1.]))
    qe_percent = pvfit_m.core.QuantumEfficiency(lambda_=lambda_, qe=numpy.array([0, 50, 100]), is_percent=True)
    # Should not alter underlying QE data representation provided by user.
    assert qe_fraction != qe_percent
    numpy.testing.assert_equal(qe_fraction.x, qe_percent.x)
    numpy.testing.assert_equal(numpy.equal(qe_fraction.y, qe_percent.y), numpy.array([True, False, False]))
    numpy.testing.assert_equal(100*qe_fraction.y, qe_percent.y)
    numpy.testing.assert_equal(qe_fraction.lambda_, qe_percent.lambda_)
    numpy.testing.assert_equal(qe_fraction.qe, qe_percent.qe)
    numpy.testing.assert_equal(qe_fraction.qe_as_percent, qe_percent.qe_as_percent)


def test_compute_m():
    """Test computation of M."""
    lambda_ = numpy.array([100, 200, 300], dtype=float)
    sr_td = pvfit_m.core.SpectralResponsivity(lambda_=lambda_, sr=numpy.full_like(lambda_, 2))
    ir_td = pvfit_m.core.SpectralIrradiance(lambda_=lambda_, ir=numpy.full_like(lambda_, 1))
    sr_rd = pvfit_m.core.SpectralResponsivity(lambda_=lambda_, sr=numpy.full_like(lambda_, 0.5))
    ir_rd = pvfit_m.core.SpectralIrradiance(lambda_=lambda_, ir=numpy.full_like(lambda_, 1))
    ir_0 = pvfit_m.core.SpectralIrradiance(lambda_=lambda_, ir=numpy.full_like(lambda_, 1))
    assert pvfit_m.core.compute_m(sr_td=sr_td, ir_td=ir_td, sr_rd=sr_rd, ir_rd=ir_rd, ir_0=ir_0) == 1.


def test_convert_qe_to_sr():
    """Test conversions from QE to SR."""
    lambda_ = numpy.array([100, 200, 300])
    # Indifference to fraction vs. percent representation.
    qe_fraction = pvfit_m.core.QuantumEfficiency(lambda_=lambda_, qe=numpy.array([0, 0.5, 1.]))
    qe_percent = pvfit_m.core.QuantumEfficiency(lambda_=lambda_, qe=numpy.array([0, 50, 100]), is_percent=True)
    assert pvfit_m.core.convert_qe_to_sr(qe=qe_fraction) == pvfit_m.core.convert_qe_to_sr(qe=qe_percent)
    # Round trip identity function.
    qe_fraction_round_trip = pvfit_m.core.convert_sr_to_qe(sr=pvfit_m.core.convert_qe_to_sr(qe=qe_fraction))
    numpy.testing.assert_equal(qe_fraction_round_trip.lambda_, qe_fraction.lambda_)
    numpy.testing.assert_almost_equal(qe_fraction_round_trip.qe, qe_fraction.qe)
    qe_percent_round_trip = pvfit_m.core.convert_sr_to_qe(sr=pvfit_m.core.convert_qe_to_sr(qe=qe_percent))
    numpy.testing.assert_equal(qe_percent_round_trip.lambda_, qe_percent.lambda_)
    numpy.testing.assert_almost_equal(qe_percent_round_trip.qe, qe_percent.qe)


def test_convert_sr_to_qe():
    """Test conversions from SR to QE."""
    lambda_ = numpy.array([100, 200, 300])
    sr = pvfit_m.core.SpectralResponsivity(lambda_=lambda_, sr=numpy.array([0, 0.5, 1.]))
    # Round trip identity function.
    sr_round_trip = pvfit_m.core.convert_qe_to_sr(qe=pvfit_m.core.convert_sr_to_qe(sr=sr))
    numpy.testing.assert_equal(sr_round_trip.lambda_, sr.lambda_)
    numpy.testing.assert_almost_equal(sr_round_trip.sr, sr.sr)


def test_inner_product():
    """Test computation of inner product of two DataCurves."""
    x1 = numpy.array([100, 200, 300])
    dc1 = pvfit_m.core.DataCurve(x=x1, y=numpy.ones_like(x1))
    x2 = numpy.array([50, 150, 250])
    dc2 = pvfit_m.core.DataCurve(x=x2, y=numpy.full_like(x2, 2))
    inner_product_expected = 2. * (250 - 100)
    inner_product = pvfit_m.core.inner_product(dc1=dc1, dc2=dc2)
    assert isinstance(inner_product, float)
    assert inner_product == inner_product_expected
    # Commutativity.
    assert inner_product == pvfit_m.core.inner_product(dc1=dc2, dc2=dc1)

    x1 = numpy.array([100, 200, 300])
    dc1 = pvfit_m.core.DataCurve(x=x1, y=numpy.array([0, 1, 2]))
    x2 = numpy.array([50, 150, 250, 350])
    dc2 = pvfit_m.core.DataCurve(x=x2, y=numpy.array([3, 2, 1, 0]))
    inner_product_expected = -(300**3 - 100**3) / 30000 + 45 / 2000 * (300**2 - 100**2) - 35 / 10 * (300 - 100)
    inner_product = pvfit_m.core.inner_product(dc1=dc1, dc2=dc2)
    assert isinstance(inner_product, float)
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    assert inner_product == pvfit_m.core.inner_product(dc1=dc2, dc2=dc1)

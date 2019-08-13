import numpy

import pvfit_m


# TODO Test all classes.
# TODO More tests for SR to QE conversion.
# TODO More tests for QE to SR conversion.


def test_constants():
    """Know when the constants change."""
    assert pvfit_m.api.q_C == 1.6021766208e-19
    assert pvfit_m.api.c_m_per_s == 299792458.0
    assert pvfit_m.api.h_J_s == 6.62607004e-34


def test_DataCurve():
    """Test DataCurve class."""
    x = numpy.array([100, 200, 300])
    y = numpy.array([0, 0.5, 1.])
    # Indifference to fraction vs. percent representation.
    dc = pvfit_m.api.DataCurve(x=x, y=y)
    dc_duplicate = pvfit_m.api.DataCurve(x=x, y=y)
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


def test_compute_m():
    """Test computation of M."""
    lambda_nm = numpy.array([100, 200, 300], dtype=float)
    sr_td = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 2))
    si_td = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    sr_rd = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.full_like(lambda_nm, 0.5))
    si_rd = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    si_0 = pvfit_m.api.SpectralIrradiance(lambda_nm=lambda_nm, si_W_per_m2_nm=numpy.full_like(lambda_nm, 1))
    assert pvfit_m.api.compute_m(sr_td=sr_td, si_td=si_td, sr_rd=sr_rd, si_rd=si_rd, si_0=si_0) == 1.


def test_convert_qe_to_sr():
    """Test conversions from QE to SR."""
    lambda_nm = numpy.array([100, 200, 300])
    # Indifference to fraction vs. percent representation.
    qe = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, qe=numpy.array([0, 0.5, 1.]))
    qe_percent = pvfit_m.api.QuantumEfficiency(lambda_nm=lambda_nm, qe=numpy.array([0, 50, 100]), is_percent=True)
    assert pvfit_m.api.convert_qe_to_sr(qe=qe) == pvfit_m.api.convert_qe_to_sr(qe=qe_percent)
    # Round trip identity function.
    qe_fraction_round_trip = pvfit_m.api.convert_sr_to_qe(sr=pvfit_m.api.convert_qe_to_sr(qe=qe))
    numpy.testing.assert_equal(qe_fraction_round_trip.lambda_nm, qe.lambda_nm)
    numpy.testing.assert_almost_equal(qe_fraction_round_trip.qe, qe.qe)
    qe_percent_round_trip = pvfit_m.api.convert_sr_to_qe(sr=pvfit_m.api.convert_qe_to_sr(qe=qe_percent))
    numpy.testing.assert_equal(qe_percent_round_trip.lambda_nm, qe_percent.lambda_nm)
    numpy.testing.assert_almost_equal(qe_percent_round_trip.qe, qe_percent.qe)


def test_convert_sr_to_qe():
    """Test conversions from SR to QE."""
    lambda_nm = numpy.array([100, 200, 300])
    sr = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=numpy.array([0, 0.5, 1.]))
    # Round trip identity function.
    sr_round_trip = pvfit_m.api.convert_qe_to_sr(qe=pvfit_m.api.convert_sr_to_qe(sr=sr))
    numpy.testing.assert_equal(sr_round_trip.lambda_nm, sr.lambda_nm)
    numpy.testing.assert_almost_equal(sr_round_trip.sr_A_per_W, sr.sr_A_per_W)


def test_inner_product():
    """Test computation of inner product of two DataCurves."""
    x1 = numpy.array([100, 200, 300])
    dc1 = pvfit_m.api.DataCurve(x=x1, y=numpy.ones_like(x1))
    x2 = numpy.array([50, 150, 250])
    dc2 = pvfit_m.api.DataCurve(x=x2, y=numpy.full_like(x2, 2))
    inner_product_expected = 2. * (250 - 100)
    inner_product = pvfit_m.api.inner_product(dc1=dc1, dc2=dc2)
    assert isinstance(inner_product, float)
    assert inner_product == inner_product_expected
    # Commutativity.
    assert inner_product == pvfit_m.api.inner_product(dc1=dc2, dc2=dc1)

    x1 = numpy.array([100, 200, 300])
    dc1 = pvfit_m.api.DataCurve(x=x1, y=numpy.array([0, 1, 2]))
    x2 = numpy.array([50, 150, 250, 350])
    dc2 = pvfit_m.api.DataCurve(x=x2, y=numpy.array([3, 2, 1, 0]))
    inner_product_expected = -(300**3 - 100**3) / 30000 + 45 / 2000 * (300**2 - 100**2) - 35 / 10 * (300 - 100)
    inner_product = pvfit_m.api.inner_product(dc1=dc1, dc2=dc2)
    assert isinstance(inner_product, float)
    numpy.testing.assert_almost_equal(inner_product, inner_product_expected)
    # Commutativity.
    assert inner_product == pvfit_m.api.inner_product(dc1=dc2, dc2=dc1)

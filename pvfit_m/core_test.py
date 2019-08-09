import numpy

import pvfit_m


# TODO Test all the classes.


def test_constants():
    """Know when the constants change."""
    assert pvfit_m.core.q_C == 1.6021766208e-19
    assert pvfit_m.core.c_m_per_s == 299792458.0
    assert pvfit_m.core.h_J_s == 6.62607004e-34


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
    QE_fraction = pvfit_m.core.QuantumEfficiency(lambda_=lambda_, qe=numpy.array([0, 0.5, 1.]))
    QE_percent = pvfit_m.core.QuantumEfficiency(lambda_=lambda_, qe=numpy.array([0, 50, 100]), is_percent=True)
    assert pvfit_m.core.convert_qe_to_sr(qe=QE_fraction) == pvfit_m.core.convert_qe_to_sr(qe=QE_percent)


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

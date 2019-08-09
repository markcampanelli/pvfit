import numpy
import pytest

import pvfit_m


# TODO Test all the classes.

def test_version():
    assert pvfit_m.__version__ is not None


def test_constants():
    """Know when the constants change."""
    assert pvfit_m.core.q_C == 1.6021766208e-19
    assert pvfit_m.core.c_m_per_s == 299792458.0
    assert pvfit_m.core.h_J_s == 6.62607004e-34


def test_convert_qe_to_sr():
    """Test conversions from QE to SR."""
    lambda_ = numpy.array([100., 200., 300.])
    QE_fraction = pvfit_m.core.QuantumEfficiency(lambda_=lambda_, qe=numpy.array([0., 0.5, 1.]))
    QE_percent = pvfit_m.core.QuantumEfficiency(lambda_=lambda_, qe=numpy.array([0., 50., 100.]), is_percent=True)
    assert pvfit_m.core.convert_qe_to_sr(qe=QE_fraction) == pvfit_m.core.convert_qe_to_sr(qe=QE_percent)


def test_compute_m_nan():
    """Test computation of NaN under bad conditions."""
    lambda_ = numpy.array([100., 200., 300.])
    sr_td = pvfit_m.core.SpectralResponsivity(lambda_=lambda_, sr=numpy.random.random(lambda_.shape))
    ir_td = pvfit_m.core.SpectralIrradiance(lambda_=lambda_, ir=numpy.random.random(lambda_.shape))
    sr_rd = pvfit_m.core.SpectralResponsivity(lambda_=lambda_, sr=numpy.random.random(lambda_.shape))
    ir_rd = pvfit_m.core.SpectralIrradiance(lambda_=lambda_, ir=numpy.random.random(lambda_.shape))
    ir_0 = pvfit_m.core.SpectralIrradiance(lambda_=lambda_, ir=numpy.random.random(lambda_.shape))
    with pytest.raises(NotImplementedError):
        pvfit_m.core.compute_m(sr_td=sr_td, ir_td=ir_td, sr_rd=sr_rd, ir_rd=ir_rd, ir_0=ir_0)

"""
PVfit testing: Spectral correction types.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy
import pytest

from pvfit.measurement.spectral_correction.types import (
    DataFunction,
    DataFunctionNonnegativeXNonnegativeY,
    QuantumEfficiency,
    SpectralIrradiance,
    SpectralResponsivity,
)

# TODO Test all classes, including more tests for SR->QE and QE->SR conversions.


def test_DataFunction():
    """Test DataFunction class."""
    # Invaid data cases.
    x = numpy.array([])
    y = numpy.array([0, 0.5, 1.0])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "x must have at least one element." in str(excinfo.value)

    x = numpy.array([[100, 200, 300], [100, 200, 300]])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "x cannot have dimension greater than one." in str(excinfo.value)

    x = numpy.array([100, 200, 100])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "x values must be unique." in str(excinfo.value)

    x = numpy.array([100, 200, numpy.inf])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "array must not contain infs or NaNs" in str(excinfo.value)

    x = numpy.array([100, 200, numpy.nan])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "array must not contain infs or NaNs" in str(excinfo.value)

    x = numpy.array([100, 200, 300])
    y = numpy.array([0, numpy.inf, 1.0])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "array must not contain infs or NaNs" in str(excinfo.value)

    y = numpy.array([numpy.nan, 0.5, 1.0])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "array must not contain infs or NaNs" in str(excinfo.value)

    y = numpy.array([0, 0.5])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "last dimension of y must equal size of x." in str(excinfo.value)

    y = numpy.array([0, 0.5, 1.0, 1.5])
    with pytest.raises(ValueError) as excinfo:
        DataFunction(x=x, y=y)
    assert "last dimension of y must equal size of x." in str(excinfo.value)

    x = numpy.array([100, 200, 300])
    # 1D case
    y = numpy.array([0, 0.5, 1.0])
    f = DataFunction(x=x, y=y)
    f_duplicate = DataFunction(x=x, y=y)
    assert f == f_duplicate
    # 2D case
    y = numpy.array([[0, 0.5, 1.0], [1, 1.5, 2.0]])
    f = DataFunction(x=x, y=y)
    f_duplicate = DataFunction(x=x, y=y)
    assert f == f_duplicate
    # 3D case
    y = numpy.array([[[0, 0.5, 1.0], [1, 1.5, 2.0]], [[1, 1.5, 2.0], [0, 0.5, 1.0]]])
    f = DataFunction(x=x, y=y)
    f_duplicate = DataFunction(x=x, y=y)
    assert f == f_duplicate

    # Check assignment with sorting.
    # 1D case
    x = numpy.array([300, 200.0, 100])
    y = numpy.array([1.0, 0.5, 0])
    f = DataFunction(x=x, y=y)
    numpy.testing.assert_array_equal(f.x, numpy.flip(x))
    numpy.testing.assert_array_equal(f.y, numpy.flip(y))
    # 2D case
    x = numpy.array([300, 200.0, 100])
    y = numpy.array([[1.0, 0.5, 0], [2.0, 1.5, 1]])
    f = DataFunction(x=x, y=y)
    numpy.testing.assert_array_equal(f.x, numpy.flip(x))
    numpy.testing.assert_array_equal(f.y, numpy.fliplr(y))


def test_DataFunctionNonnegativeXNonnegativeY():
    """Test DataFunctionNonnegativeXNonnegativeY class."""
    # Invaid data cases.
    # x
    x = numpy.array([-1, 100, 200])
    y = numpy.array([0, 0.5, 1.0])
    with pytest.raises(ValueError) as excinfo:
        DataFunctionNonnegativeXNonnegativeY(x=x, y=y)
    assert "x values must all be non-negative." in str(excinfo.value)
    # y
    x = numpy.array([100, 200])
    y = numpy.array([-0.5, 1.0])
    with pytest.raises(ValueError) as excinfo:
        DataFunctionNonnegativeXNonnegativeY(x=x, y=y)
    assert "y values must all be non-negative." in str(excinfo.value)


def test_QuantumEfficiency():
    """Test QuantumEfficiency class."""
    # Assignment.
    lambda_nm = numpy.array([100, 200, 300])
    QE = numpy.array([0, 0.5, 1.0])
    QE_fraction = QuantumEfficiency(lambda_nm=lambda_nm, QE=QE)
    numpy.testing.assert_array_equal(QE_fraction.lambda_nm, lambda_nm)
    numpy.testing.assert_array_equal(QE_fraction.QE, QE)
    # Should not alter underlying QE data representation provided by user.
    QE_percent = QuantumEfficiency(
        lambda_nm=lambda_nm, QE=numpy.array([0, 50, 100]), is_percent=True
    )
    assert QE_fraction != QE_percent
    numpy.testing.assert_equal(QE_fraction.x, QE_percent.x)
    numpy.testing.assert_equal(
        numpy.equal(QE_fraction.y, QE_percent.y), numpy.array([True, False, False])
    )
    numpy.testing.assert_equal(100 * QE_fraction.y, QE_percent.y)
    numpy.testing.assert_equal(QE_fraction.lambda_nm, QE_percent.lambda_nm)
    numpy.testing.assert_equal(QE_fraction.QE, QE_percent.QE)
    numpy.testing.assert_equal(QE_fraction.QE_percent, QE_percent.QE_percent)
    # Test conversions from QuantumEfficiency to SpectralResponivity.
    # Indifference to fraction vs. percent representation.
    QE_fraction = QuantumEfficiency(lambda_nm=lambda_nm, QE=numpy.array([0, 0.5, 1.0]))
    QE_percent = QuantumEfficiency(
        lambda_nm=lambda_nm, QE=numpy.array([0, 50, 100]), is_percent=True
    )
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
    lambda_nm = numpy.array([100.0, 200.0, 300.0])
    E_W_per_m2_nm = numpy.array([0, 0.5, 1.0])
    E = SpectralIrradiance(lambda_nm=lambda_nm, E_W_per_m2_nm=E_W_per_m2_nm)
    numpy.testing.assert_array_equal(E.lambda_nm, lambda_nm)
    numpy.testing.assert_array_equal(E.E_W_per_m2_nm, E_W_per_m2_nm)


def test_SpectralResponivity():
    """Test SpectralResponivity class."""
    # Assignment.
    lambda_nm = numpy.array([100.0, 200.0, 300.0])
    S_A_per_W = numpy.array([0, 0.5, 1.0])
    S = SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=S_A_per_W)
    numpy.testing.assert_array_equal(S.lambda_nm, lambda_nm)
    numpy.testing.assert_array_equal(S.S_A_per_W, S_A_per_W)
    # Test conversions from SpectralResponivity to QuantumEfficiency.
    # Round trip identity function.
    S_round_trip = S.QE.S_A_per_W
    numpy.testing.assert_equal(S_round_trip.lambda_nm, S.lambda_nm)
    numpy.testing.assert_almost_equal(S_round_trip.S_A_per_W, S.S_A_per_W)

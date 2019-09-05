"""Package tests."""

import pvfit_m


def test_package_imports():
    assert pvfit_m.api is not None
    assert pvfit_m.data is not None


def test_package_version():
    assert pvfit_m.__version__ is not None

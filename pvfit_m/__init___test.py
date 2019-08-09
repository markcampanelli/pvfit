import pvfit_m


def test_imports():
    assert pvfit_m.core is not None


def test_version():
    assert pvfit_m.__version__ is not None

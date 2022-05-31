import copy
import logging

import numpy
import pytest

import pvfit.common.utils


@pytest.fixture(params=[
        {  # Test empty dictionary.
            'dictionary': {},
            'dictionary_expected': {}
        },
        {  # Test mixed Python dictionary types.
            'dictionary': {'a': 0, 'b': 0., 'c': 'blah', 'd': numpy.array(0), 'e': numpy.array([0., 1.])},
            'dictionary_expected': {'a': numpy.intc(0), 'b': numpy.float64(0.), 'c': 'blah', 'd': numpy.array(0),
                                    'e': numpy.array([0., 1.])}
        },
    ])
def ensure_numpy_scalars_fixture(request):
    return request.param


def test_ensure_numpy_scalars(ensure_numpy_scalars_fixture):
    dictionary_expected = ensure_numpy_scalars_fixture['dictionary_expected']
    dictionary = pvfit.common.utils.ensure_numpy_scalars(
        dictionary=copy.deepcopy(ensure_numpy_scalars_fixture['dictionary']))

    for key, value in dictionary_expected.items():
        assert isinstance(value, type(dictionary[key]))
        if isinstance(value, numpy.ndarray) or isinstance(value, numpy.float64) or isinstance(value, numpy.intc):
            assert value.dtype == dictionary[key].dtype
            assert value.shape == dictionary[key].shape
            numpy.testing.assert_array_equal(value, dictionary[key])


def test_get_version(caplog) -> str:
    caplog.set_level(logging.INFO)
    version1 = pvfit.common.utils.get_version()
    assert version1
    assert len(caplog.records) == 0

    version2 = pvfit.common.utils.get_version(info_log=True)
    assert version1 == version2
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"
    assert f"pvfit version {version2}" in caplog.text

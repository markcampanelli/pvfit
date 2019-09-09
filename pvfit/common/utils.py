"""Common utilities for PVfit."""

import numpy


def ensure_numpy_scalars(*, dictionary: dict) -> dict:
    """
    Convert Python float and int dictionary entries to numpy interface.

    TODO: Complete docstring.

    CAUTION: This updates the input dictionary in place and returns a reference for convienence.
    """

    # In order to allow consistent downstream processing, make sure no Python scalars are returned.
    # Note that many external operations cast rank-0 numpy.ndarray to numpy.float64.
    dictionary.update({key: numpy.float64(value) for key, value in dictionary.items() if isinstance(value, float)})
    dictionary.update({key: numpy.intc(value) for key, value in dictionary.items() if isinstance(value, int)})

    return dictionary

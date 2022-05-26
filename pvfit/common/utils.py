"""Common utilities for PVfit."""
from importlib.metadata import version, PackageNotFoundError
import logging

import numpy

logger = logging.getLogger(__name__)


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


def get_version(*, log: bool = True) -> str:
    """Return pvfit version with optional info log, raising with exception log if not found."""

    try:
        # This relies on setuptools_scm magic, and raises when pvfit package is not installed.
        version_ = version('pvfit')
    except PackageNotFoundError:
        logger.exception("pvfit version not found. Is the pvfit package properly installed?")

    if log:
        logger.info(f"pvfit version {version_}")

    return version_

"""
PVfit: Types.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from dataclasses import dataclass
import typing

import numpy
import numpy.typing

# Vectors are 1D.
type FloatVector = numpy.typing.NDArray[numpy.floating[typing.Any]]
type IntVector = numpy.typing.NDArray[numpy.integer[typing.Any]]

# Arrays are ND, inc. 0D and 1D.
type FloatArray = numpy.typing.NDArray[numpy.floating[typing.Any]]
type IntArray = numpy.typing.NDArray[numpy.integer[typing.Any]]

# Broadcastables indicate that they have broadcast compatibility in the usage context.
# numpy scalar types deliberately excluded, due to inconsistent typing interoperability.
type FloatBroadcastable = typing.Union[float, FloatArray]
type IntBroadcastable = typing.Union[int, IntArray]


@dataclass
class NewtonOptions:
    """Optional options exposed for newton solver."""

    maxiter: int = 50
    disp: bool = False


@dataclass
class OdrOptions:
    """
    Options (with defaults) exposed for orthogonal distance regression (ODR) solver.
    """

    maxit: int = 10000

"""
PVfit: Types.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import TypedDict, Union

import numpy
import numpy.typing

# Vectors are 1D.
FloatVector = numpy.typing.NDArray[numpy.float_]
IntVector = numpy.typing.NDArray[numpy.int_]

# Arrays are ND, inc. 0D and 1D.
FloatArray = numpy.typing.NDArray[numpy.float_]
IntArray = numpy.typing.NDArray[numpy.int_]

# Broadcastables indicate that they have broadcast compatibility in the usage context.
# numpy scalar types deliberately excluded, due to inconsistent typing interoperability.
FloatBroadcastable = Union[float, FloatArray]
IntBroadcastable = Union[int, IntArray]


class NewtonOptions(TypedDict, total=False):
    """Optional options exposed for newton solver."""

    maxiter: int
    disp: bool


class OdrOptions(TypedDict, total=False):
    """Optional options exposed for orthogonal distance regression (ODR) solver."""

    maxit: int

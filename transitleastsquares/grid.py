from __future__ import division, print_function
import numba
import transitleastsquares.tls_constants as tls_constants
import numpy
from numpy import pi, sqrt, arccos, degrees, floor, ceil
import warnings


@numba.jit(fastmath=True, parallel=False, nopython=True)
def T14(
    R_s, M_s, P, upper_limit=tls_constants.FRACTIONAL_TRANSIT_DURATION_MAX, small=False
):
    """Input:  Stellar radius and mass; planetary period
               Units: Solar radius and mass; days
       Output: Maximum planetary transit duration T_14max
               Unit: Fraction of period P"""

    P = P * tls_constants.SECONDS_PER_DAY
    R_s = tls_constants.R_sun * R_s
    M_s = tls_constants.M_sun * M_s

    if small:  # small planet assumption
        T14max = R_s * ((4 * P) / (pi * tls_constants.G * M_s)) ** (1 / 3)
    else:  # planet size 2 R_jup
        T14max = (R_s + 2 * tls_constants.R_jup) * (
            (4 * P) / (pi * tls_constants.G * M_s)
        ) ** (1 / 3)

    result = T14max / P
    if result > upper_limit:
        result = upper_limit
    return result

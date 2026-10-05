"""Period-dependent pre-binning of dense light curves (idea L4).

The search stage of TLS only needs a time resolution that is small compared
with the shortest trial duration. For each trial period P the light curve is
binned by k consecutive cadences such that k * cadence <= BIN_FRACTION *
d_min(P), where d_min(P) is the shortest duration searched at P. Copies with
k = 1, 2, 4, 8, ... are searched for their period ranges. The final T0 fit and
all statistics use the unbinned data, as before.

Each binned point carries the inverse-variance weight of its points, so for
a model that is constant within a bin the unbinned chi2 equals the binned
chi2 plus the within-bin scatter sum(w (y - y_bin)^2), which is a constant per
copy; it is added to the binned chi2 so that all periods share one chi2
scale.
"""

import os

import numpy

from transitleastsquares import tls_constants
from transitleastsquares.grid import T14


def bin_fraction():
    """Maximum bin length as a fraction of the shortest trial duration
    (tls_constants.BIN_FRACTION; environment override TLS_BIN_FRACTION)."""
    return float(os.environ.get("TLS_BIN_FRACTION", tls_constants.BIN_FRACTION))


def cadence(t):
    """Typical sampling interval (median of the positive time differences)."""
    dt = numpy.diff(numpy.sort(t))
    dt = dt[dt > 0]
    return float(numpy.median(dt)) if len(dt) else 0.0


def bin_factors(periods, cad, R_star_min, M_short, fraction, k_max=None):
    """Bin factor (power of 2, >= 1) for every trial period."""
    k_max = tls_constants.BIN_FACTOR_MAX if k_max is None else k_max
    periods = numpy.asarray(periods, dtype=float)
    if fraction <= 0 or cad <= 0:
        return numpy.ones(len(periods), dtype=numpy.int64)
    d_min = numpy.array(
        [T14(R_s=R_star_min, M_s=M_short, P=P, small=True) * P for P in periods]
    )
    k_float = fraction * d_min / cad
    k = numpy.ones(len(periods), dtype=numpy.int64)
    k_ok = k_float >= 2
    k[k_ok] = 2 ** numpy.floor(numpy.log2(k_float[k_ok])).astype(numpy.int64)
    return numpy.minimum(k, k_max)


def bin_lightcurve(t, y, dy, k, cad):
    """Bin by k consecutive points; a bin also ends at gaps (> 1.5 cadences),
    so bins never span a gap. Returns (t_b, y_b, dy_b, offset) with
    inverse-variance weighted means, dy_b = 1 / sqrt(sum w) and the
    within-bin scatter offset = sum w (y - y_b)^2."""
    order = numpy.argsort(t, kind="mergesort")
    t, y, dy = t[order], y[order], dy[order]
    w = 1 / dy**2
    n = len(t)
    new = numpy.zeros(n, dtype=bool)
    new[0] = True
    new[1:] = numpy.diff(t) > 1.5 * cad
    # bin index: count within runs between gaps
    run = numpy.cumsum(new) - 1
    start = numpy.flatnonzero(new)
    pos = numpy.arange(n) - start[run]
    first = new | (pos % k == 0)
    b = numpy.cumsum(first) - 1
    nb = b[-1] + 1
    W = numpy.bincount(b, w, nb)
    yb = numpy.bincount(b, w * y, nb) / W
    tb = numpy.bincount(b, w * t, nb) / W
    offset = float(numpy.sum(w * (y - yb[b]) ** 2))
    return tb, yb, 1 / numpy.sqrt(W), offset

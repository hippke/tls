"""Pre-binning (binning.py, PERFORMANCE_LOG step 35)."""

import numpy

from transitleastsquares import binning
from transitleastsquares import tls_constants as C
from transitleastsquares.grid import duration_limit_masses


def test_bin_lightcurve_chi2_identity():
    """For a model constant within bins: chi2 unbinned = chi2 binned + offset;
    bins never span a gap; weights add up."""
    rng = numpy.random.default_rng(0)
    cad = 25 / 86400
    t = numpy.concatenate([numpy.arange(0, 1, cad), numpy.arange(1.5, 2, cad)])
    y = 1 + rng.normal(0, 1e-3, len(t))
    dy = rng.uniform(0.8, 1.2, len(t))
    tb, yb, dyb, offset = binning.bin_lightcurve(t, y, dy, 4, cad)
    assert len(tb) == numpy.ceil(numpy.sum(t < 1) / 4) + numpy.ceil(
        numpy.sum(t > 1) / 4
    )
    assert not numpy.any((tb > 1 + cad) & (tb < 1.5 - cad))  # no bin in the gap
    numpy.testing.assert_allclose(numpy.sum(1 / dyb**2), numpy.sum(1 / dy**2))
    for m in [1.0, 0.999]:  # constant models
        full = numpy.sum((y - m) ** 2 / dy**2)
        binned = numpy.sum((yb - m) ** 2 / dyb**2) + offset
        numpy.testing.assert_allclose(binned, full, rtol=1e-12)


def test_bin_factors():
    m_short, _ = duration_limit_masses(
        C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX
    )
    periods = numpy.array([0.5, 3.0, 30.0, 300.0])
    k = binning.bin_factors(periods, 25 / 86400, C.R_STAR_MIN, m_short, 0.03)
    assert numpy.all(numpy.diff(k) >= 0)  # longer periods: longer d_min
    assert numpy.all(numpy.log2(k) == numpy.round(numpy.log2(k)))  # powers of 2
    assert k[0] == 1 and k[-1] >= 4
    off = binning.bin_factors(periods, 25 / 86400, C.R_STAR_MIN, m_short, 0.0)
    assert numpy.all(off == 1)

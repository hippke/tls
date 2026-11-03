"""transitleastsquares.gp: rotation fits on binned data for short cadences
(fit_bin_width). The search always uses the unbinned data; K2-like cadences
are never binned."""

import numpy as np
import pytest

from gp_cases import synthetic
from transitleastsquares import gp
from transitleastsquares.gp.baselines import rotation_period_guess
from transitleastsquares.gp.fitting import bin_series

CAD_2MIN = 2.0 / 1440


def test_bin_series():
    t = np.array([0.0, 0.1, 0.2, 1.05, 1.1, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 7.0])
    e = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    tb, yb, eb, n = bin_series(t, y, e, 1.0)
    np.testing.assert_allclose(tb, [0.1, 1.075, 3.0])
    np.testing.assert_allclose(yb, [2.0, 5.0, 7.0])
    np.testing.assert_allclose(eb, [np.sqrt(3) / 3, np.sqrt(8) / 2, 3.0])
    np.testing.assert_array_equal(n, [3, 2, 1])


def test_k2_cadence_not_binned():
    """At the K2 cadence (29.4 min) the default bin width (10 min) changes
    nothing: bit-identical to bin_width=None."""
    t, f, e = synthetic(days=12.0)
    a, la = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, use_threads=1)
    b, lb = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, use_threads=1, bin_width=None)
    assert a == b and la == lb


@pytest.fixture(scope="module")
def short_cadence():
    t, f, e = synthetic(days=8.0, cadence=CAD_2MIN)
    unbinned = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, use_threads=1, bin_width=None)
    binned = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, use_threads=1)
    return (t, f, e), unbinned, binned


def test_binned_fit_matches_unbinned(short_cadence):
    """2 min cadence binned to 10 min. The rotation likelihood has several
    nearly equal optima (the parameters may land in another one), so compare
    what matters: the unbinned likelihood at the binned optimum is at least
    that of the unbinned fit minus a few units (out of ~4600 points), the
    rotation period is right, and the per-point jitter is converted back."""
    (t, f, e), (bu, llu), (bb, llb) = short_cadence
    assert llb > llu - 5.0
    assert abs(bb.period - 2.3) / 2.3 < 0.05 and abs(bu.period - 2.3) / 2.3 < 0.05
    assert abs(bb.jitter - bu.jitter) < 0.2 * bu.jitter


def test_binned_fit_returns_unbinned_likelihood(short_cadence):
    (t, f, e), _, (bb, llb) = short_cadence
    np.testing.assert_allclose(bb.log_likelihood(t, f, e), llb, rtol=1e-12)


def test_search_with_binned_fit_finds_transit():
    """End to end at 2 min cadence: the fit is binned, the search is not."""
    t, f, e = synthetic(days=8.0, cadence=CAD_2MIN)
    r = gp.search(t, f, e, period_min=1.0, period_max=4.0, rotation_period=2.3, use_threads=1)
    assert abs(r.period - 3.1) / 3.1 < 0.01 and len(r.time) == len(t)


def test_rotation_period_guess_binned():
    """Short cadences are binned for the Lomb-Scargle guess; the peak is the
    same as for the unbinned periodogram."""
    t, f, e = synthetic(days=8.0, cadence=CAD_2MIN)
    p = rotation_period_guess(t, f)
    assert abs(p - 2.3) / 2.3 < 0.1

"""Backend "events" (fold of single transit events): the single-event sums
and the fold against direct computations, and detections close to the
default backend."""

import os
import warnings

import numpy
import pytest

from transitleastsquares import transitleastsquares
from transitleastsquares.backends import events as ev
from transitleastsquares.gp.events import compact_offsets, template_table


def synthetic(n=6000, cadence=0.0204, P=3.71, depth=1.5e-3, seed=5):
    rng = numpy.random.default_rng(seed)
    t = numpy.arange(n) * cadence
    t = t[rng.random(n) > 0.05]
    dy = numpy.full(len(t), 1e-3) * rng.uniform(0.9, 1.1, len(t))
    y = 1 + rng.normal(0, 1, len(t)) * dy
    ph = (t - 1.3 + 0.5 * P) % P - 0.5 * P
    y[numpy.abs(ph) < 0.06] -= depth
    return t, y, dy


def test_single_events_direct():
    t, y, dy = synthetic(1500)
    r, w = y - 1, 1 / dy**2
    durs = numpy.array([0.05, 0.13, 0.3])
    delta = 0.0204 / 4
    mstep = numpy.array([1, 2, 6])
    tc = t[0] + delta * numpy.arange(int((t[-1] - t[0]) / delta) + 1)
    off = compact_offsets(len(tc), mstep)
    table = template_table()
    nsubs = numpy.array([7, 7, 1])
    ab = ev.single_events_white(t, r, w, tc, durs, nsubs, table, 0.0204, mstep, off)
    rng = numpy.random.default_rng(1)
    for row in rng.integers(0, off[-1], 40):
        j = numpy.searchsorted(off, row, side="right") - 1
        tcm = tc[(row - off[j]) * mstep[j]]
        s = numpy.array([ev._template(x, tcm, durs[j], table, 0.0204, nsubs[j]) for x in t])
        numpy.testing.assert_allclose(ab[row], [numpy.sum(s * r * w), numpy.sum(s * s * w)], rtol=1e-10, atol=1e-12)


def test_fold_brute_force():
    rng = numpy.random.default_rng(2)
    M, delta = 400, 0.01
    mstep = numpy.array([1, 3])
    off = compact_offsets(M, mstep)
    ab = numpy.column_stack([rng.normal(0, 1, off[-1]), rng.uniform(0.5, 2, off[-1])])
    periods = numpy.array([0.731, 1.37, 1.9])
    jlo, jhi = numpy.zeros(3, numpy.int64), numpy.ones(3, numpy.int64)
    gain, jb, A, B = ev.fold_events(ab, off, M, periods, delta, mstep, jlo, jhi, 2, 0.0)
    for p, P in enumerate(periods):
        best = 0.0
        for j in range(2):
            dj = mstep[j] * delta
            Mj = off[j + 1] - off[j]
            for k0 in range(max(1, int(P / dj))):
                ks = [k0 + int(round(nn * P / dj)) for nn in range(int(Mj * dj / P) + 2)]
                ks = [k for k in ks if k < Mj]
                if len(ks) < 2:
                    continue
                a, b = ab[off[j] + numpy.array(ks)].sum(0)
                if a > 0:
                    best = max(best, a * a / b)
        assert gain[p] == pytest.approx(best, rel=1e-12)


def run(t, y, dy, backend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transitleastsquares(t, y, dy, verbose=False).power(
            backend=backend, use_threads=2, show_progress_bar=False, verbose=False,
            period_min=0.5, period_max=20)


def test_detection_close_to_default():
    t, y, dy = synthetic()
    a, b = run(t, y, dy, "fused-pl"), run(t, y, dy, "events")
    assert abs(b.period - a.period) / a.period < 1e-3
    assert abs(b.T0 - a.T0) < 0.01
    assert abs(b.SDE - a.SDE) < 0.1 * a.SDE
    assert abs(b.depth - a.depth) < 2e-4


def test_no_fit_below_depth_minimum():
    t, y, dy = synthetic(depth=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = transitleastsquares(t, numpy.ones(len(t)) + 1e-9 * numpy.sin(t), dy, verbose=False).power(
            backend="events", use_threads=1, show_progress_bar=False, verbose=False, transit_depth_min=1e-3)
    assert numpy.isnan(r.period)


def test_events_uses_user_n_transits_min():
    """The events backend applies the user's n_transits_min (a single transit
    is found with n_transits_min=1)."""
    rng = numpy.random.default_rng(3)
    t = numpy.arange(0, 40, 0.01)
    y = 1 + rng.normal(0, 2e-4, len(t))
    y[numpy.abs(t - 23.0) < 0.1] -= 3e-3  # one transit
    from transitleastsquares import transitleastsquares

    r = transitleastsquares(t, y, numpy.full(len(t), 2e-4)).power(
        backend="events", n_transits_min=1, use_threads=1, show_progress_bar=False, verbose=False,
        period_min=5, period_max=39)
    assert numpy.min(numpy.abs(numpy.asarray(r.transit_times) - 23.0)) < 0.1


def test_lazy_templates_equal():
    """Templates on demand (a backend that does not search with them) equal the full cache."""
    from transitleastsquares import transit

    kw = dict(per=12.9, rp=0.1, a=23.1, inc=89.0, ecc=0.0, w=90.0, u=[0.4804, 0.1867], limb_dark="quadratic", verbose=False)
    durations = numpy.geomspace(0.001, 0.05, 12)
    ov, arr = transit.get_cache(durations, 200000, **kw)
    ov_l, arr_l = transit.get_cache(durations, 200000, lazy=True, **kw)
    assert isinstance(arr_l, transit._LazyTemplates) and len(arr_l) == len(arr)
    numpy.testing.assert_array_equal(ov_l["duration"], ov["duration"])
    for row in (0, 5, 11):
        numpy.testing.assert_array_equal(arr_l[row], arr[row])
    assert not isinstance(transit.get_cache(durations, 2000, lazy=True, **kw)[1], transit._LazyTemplates)

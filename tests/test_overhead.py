"""Per-call overhead changes (round 10): exact equivalence with the previous
implementations, scheduling and the shared process pool."""

import numpy
import pytest

from transitleastsquares import transitleastsquares
from transitleastsquares.backends import get_backend
from transitleastsquares.backends.pool import schedule
from transitleastsquares.helpers import _pad_to_length, running_median
from transitleastsquares.stats import (
    _between,
    _pink_noise,
    count_stats,
    intransit_stats,
)

Q = dict(show_progress_bar=False, verbose=False)


def median_reference(data, kernel):
    idx = numpy.arange(kernel) + numpy.arange(len(data) - kernel + 1)[:, None]
    med = numpy.median(data[idx.astype(numpy.int64)], axis=1)
    return _pad_to_length(med, len(data))


@pytest.mark.parametrize("kind", ["normal", "ties", "integers", "inf"])
def test_running_median_identical(kind):
    rng = numpy.random.default_rng(len(kind))
    for _ in range(40):
        n = int(rng.integers(5, 2000))
        k = int(rng.choice([1, 3, 5, 31, 91, 151]))
        if k > n:
            continue
        x = rng.normal(size=n)
        if kind == "ties":
            x = numpy.round(x, 1)
        elif kind == "integers":
            x = rng.integers(0, 3, n).astype(float)
        elif kind == "inf":
            x[rng.integers(0, n, 3)] = numpy.inf
            x[rng.integers(0, n, 3)] = -numpy.inf
        numpy.testing.assert_array_equal(running_median(x, k), median_reference(x, k))


def test_running_median_fallbacks():
    x = numpy.random.default_rng(0).normal(size=300)
    x[17] = numpy.nan  # NaN: reference path (NaN windows)
    numpy.testing.assert_array_equal(running_median(x, 31), median_reference(x, 31))
    y = numpy.random.default_rng(1).normal(size=300)
    numpy.testing.assert_array_equal(running_median(y, 30), median_reference(y, 30))


def pink_reference(data, width):
    total = 0.0
    datapoints = len(data) - width + 1
    for i in range(datapoints):
        mean = 0.0
        for j in range(i, i + width):
            mean += data[j]
        mean /= width
        var = 0.0
        for j in range(i, i + width):
            var += (data[j] - mean) ** 2
        total += numpy.sqrt(var / width) / width**0.5
    return total / datapoints


def test_pink_noise_bit_identical():
    rng = numpy.random.default_rng(3)
    for n, w in [(1, 1), (3, 2), (4, 1), (7, 4), (50, 7), (333, 40), (1000, 1000)]:
        x = 1 + rng.normal(0, 1e-3, n)
        assert _pink_noise(x, w) == pink_reference(x, w)


def test_between_matches_mask():
    rng = numpy.random.default_rng(4)
    t = numpy.sort(numpy.round(rng.uniform(0, 10, 500), 2))
    for _ in range(200):
        lo, hi = numpy.sort(rng.uniform(-1, 11, 2))
        if rng.random() < 0.3:
            lo = t[rng.integers(len(t))]  # boundaries on data points (strict)
        sel = _between(t, lo, hi, True)
        numpy.testing.assert_array_equal(numpy.arange(len(t))[sel],
                                         numpy.flatnonzero((t > lo) & (t < hi)))


def test_transit_stats_unsorted_time():
    rng = numpy.random.default_rng(5)
    t = numpy.sort(rng.uniform(0, 30, 3000))
    y = 1 + rng.normal(0, 1e-3, len(t))
    perm = rng.permutation(len(t))
    tt = [1.0 + 2.5 * k for k in range(12)]
    a = intransit_stats(t, y, tt, 0.2)
    b = intransit_stats(t[perm], y[perm], tt, 0.2)
    numpy.testing.assert_array_equal(a[6], b[6])  # counts
    assert count_stats(t, y, tt, 0.2) == count_stats(t[perm], y[perm], tt, 0.2)


def test_schedule_covers_every_period_once():
    rng = numpy.random.default_rng(6)
    tasks = [(rng.permutation(1000) * 0.01 + 1, 3.0), (numpy.arange(7) + 0.5, 50.0),
             (numpy.arange(3000) * 0.001 + 20, 1.0)]
    for threads in (1, 2, 8):
        chunks = schedule(tasks, threads)
        for i, (periods, _) in enumerate(tasks):
            got = numpy.concatenate([c for k, c in chunks if k == i])
            numpy.testing.assert_array_equal(numpy.sort(got), numpy.sort(periods))
        sizes = [len(c) * tasks[k][1] for k, c in chunks]
        assert sizes[-1] <= max(sizes)  # shrinking tail
    assert schedule([], 4) == []


def lc():
    rng = numpy.random.default_rng(7)
    t = numpy.arange(0, 30, 0.01)
    y = 1 + rng.normal(0, 3e-4, len(t))
    ph = (t - 1.1) % 3.3
    y[(ph < 0.06) | (ph > 3.3 - 0.06)] -= 2e-3
    return t, y


def test_search_many_equals_separate_searches():
    from transitleastsquares.backends import SearchProblem
    from transitleastsquares.transit import get_cache

    t, y = lc()
    durations = numpy.array([0.01, 0.015, 0.02])
    ov, arr = get_cache(durations, 60, 10, 0.1, 20, 89.9, 0, 90, [0.4, 0.2],
                        "quadratic", verbose=False)
    pr = [SearchProblem(t=t[::s], y=y[::s], dy=numpy.ones(len(t[::s])), lc_arr=arr,
                        lc_cache_overview=ov, transit_depth_min=1e-5, R_star_min=0.13,
                        R_star_max=3.5, M_star_min=0.1, M_star_max=1.0,
                        T0_search_margin=0.01) for s in (1, 2)]
    jobs = [(pr[0], numpy.linspace(3.0, 3.6, 40)), (pr[1], numpy.linspace(5, 6, 25))]
    be = get_backend("fused-pl")
    many = be.search_many(jobs, use_threads=3, defer_join=True)
    be.finish()
    for (p, periods), r in zip(jobs, many):
        single = be.search(p, periods, use_threads=1)
        numpy.testing.assert_array_equal(single.chi2, r.chi2)
        numpy.testing.assert_array_equal(single.rows, r.rows)
        numpy.testing.assert_array_equal(single.periods, r.periods)


def test_dense_cadence_binned_copies_threads_identical():
    """Pre-binning tasks share one pool: 1 and 4 workers give identical spectra."""
    rng = numpy.random.default_rng(8)
    t = numpy.arange(0, 20, 25 / 86400)  # 25-s cadence: two binned copies
    y = 1 + rng.normal(0, 1e-3, len(t))
    y[((t - 0.4) % 6.1) < 0.1] -= 2e-3
    kw = dict(period_min=4.0, period_max=9.0, backend="fused-pl", **Q)
    model = transitleastsquares(t, y)
    from transitleastsquares.validate import validate_args

    validate_args(model, dict(kw))
    periods, durations = model._grids()
    ov, arr = model._templates(durations)
    assert len(model._search_plan(get_backend("fused-pl"), periods, ov, arr)) == 2
    a = transitleastsquares(t, y).power(use_threads=1, **kw)
    b = transitleastsquares(t, y).power(use_threads=4, **kw)
    numpy.testing.assert_array_equal(a.chi2, b.chi2)
    numpy.testing.assert_array_equal(a.power, b.power)
    assert a.T0 == b.T0 and a.SDE == b.SDE


# --- persistent worker pool (round 11) ---------------------------------------


def _run(t, y, **kw):
    return transitleastsquares(t, y).power(
        period_min=3, period_max=3.6, use_threads=3, backend="fused-pl", **Q, **kw
    )


def test_persistent_pool_identical_and_reused(monkeypatch):
    from transitleastsquares.backends import pool as P

    t, y = lc()
    monkeypatch.setenv("TLS_PERSISTENT_POOL", "0")
    fresh = _run(t, y)
    monkeypatch.delenv("TLS_PERSISTENT_POOL")
    a = _run(t, y)
    first = P._Persistent.pool
    b = _run(t, y)
    assert first is not None and P._Persistent.pool is first  # reused
    for r in (a, b):
        numpy.testing.assert_array_equal(fresh.chi2, r.chi2)
        numpy.testing.assert_array_equal(fresh.power, r.power)
        assert fresh.T0 == r.T0 and fresh.SDE == r.SDE
    P.shutdown_workers()
    assert P._Persistent.pool is None


def test_persistent_pool_sees_current_constants(monkeypatch):
    """Workers started earlier must use the parent's constants at call time."""
    from transitleastsquares import tls_constants

    t, y = lc()
    _run(t, y)  # start the persistent workers
    monkeypatch.setattr(tls_constants, "SIGNAL_DEPTH", 0.4)
    persistent = _run(t, y)
    monkeypatch.setenv("TLS_PERSISTENT_POOL", "0")
    fresh = _run(t, y)
    numpy.testing.assert_array_equal(fresh.chi2, persistent.chi2)


def test_persistent_pool_idle_timeout(monkeypatch):
    import time

    from transitleastsquares import tls_constants
    from transitleastsquares.backends import pool as P

    monkeypatch.setattr(tls_constants, "WORKER_IDLE_TIMEOUT", 0.2)
    t, y = lc()
    _run(t, y)
    assert P._Persistent.pool is not None
    time.sleep(1.0)
    assert P._Persistent.pool is None


def test_persistent_pool_spawn(monkeypatch):
    import glob
    import tempfile

    from transitleastsquares.backends import pool as P

    t, y = lc()
    monkeypatch.setenv("TLS_PERSISTENT_POOL", "0")
    ref = _run(t, y)
    monkeypatch.delenv("TLS_PERSISTENT_POOL")
    monkeypatch.setenv("TLS_MP_START_METHOD", "spawn")
    a = _run(t, y)
    b = _run(t, y)
    numpy.testing.assert_array_equal(ref.chi2, a.chi2)
    numpy.testing.assert_array_equal(ref.chi2, b.chi2)
    P.shutdown_workers()
    assert not glob.glob(tempfile.gettempdir() + "/tls_search_*.pkl")

"""Three-gap walk fold (idea L12, PERFORMANCE_LOG step 51): for time stamps on
a regular grid the phase order comes from a walk over the grid slots instead
of the counting scatter. It must give exactly the stable (mergesort) order,
hence bit-identical search results."""

import os

import numpy
import pytest
from test_core_oracle import cache

from transitleastsquares import tls_constants as C
from transitleastsquares.backends import SearchProblem
from transitleastsquares.core import foldfast
from transitleastsquares.core_fused import (
    FusedProblem,
    _three_gap_steps,
    fold_walk_into,
    regular_grid,
    regular_grid_crossover,
)


def grids(rng):
    g = numpy.arange(6000)
    d = 2 / 1440
    yield "regular", g * d + 1325.0
    yield "gappy", (g * d + 1325.0)[rng.random(g.size) > 0.35]
    yield "segments", numpy.concatenate([g[:2500], g[3500:]]) * d
    yield "jitter", g * d + rng.normal(0, 3e-6, g.size)  # ~0.3 s
    yield "drift", g * d * (1 + 1e-5 * numpy.sin(g / 900.0))  # slow cadence drift
    yield "unsorted", rng.permutation(g * d + 7.0)
    yield "integer", g.astype(float)  # P = m * delta: exactly tied grid phases
    yield "negative", g * d - 3.3


def test_three_gap_steps_match_scan():
    """a, b = argmin / argmax of frac(j alpha), 1 <= j < G (exact rational
    arithmetic on the float alpha). Near-rational alpha may be misjudged by
    the float descent; the walk's repair covers that, so only the bijection
    property (1 <= a, b < G, a + b >= G) is required there."""
    from fractions import Fraction

    rng = numpy.random.default_rng(0)
    for G in (2, 3, 10, 97, 1000):
        for alpha in rng.random(25):
            fa = Fraction(float(alpha))
            f = [(j * fa) % 1 for j in range(1, G)]
            a, b = _three_gap_steps(G, alpha)
            assert (a, b) == (f.index(min(f)) + 1, f.index(max(f)) + 1)
        for alpha in (0.5, 1 / 3, 0.1, 1e-4, 1 - 1e-12, 2 / 7):
            a, b = _three_gap_steps(G, alpha)
            assert 1 <= a < max(G, 2) and 1 <= b < max(G, 2) and a + b >= G


@pytest.mark.parametrize("name", [n for n, _ in grids(numpy.random.default_rng(1))])
def test_walk_order_is_stable_order(name):
    rng = numpy.random.default_rng(1)
    t = dict(grids(rng))[name]
    n = len(t)
    walk = regular_grid(t)
    assert walk is not None, name
    gp, slot = walk[0], walk[1]
    assert numpy.sum(gp >= 0) == n
    numpy.testing.assert_array_equal(gp[slot], numpy.arange(n))
    span = numpy.ptp(t)
    d = walk[2]
    periods = list(rng.uniform(16 * d, span / 2, 40))
    periods += [m * d for m in (17, 97, 360, 720, 1440, 1441)] + [1.0, 2.0, 0.5]
    ph = numpy.empty(n)
    buf = numpy.empty(n + 1, dtype=numpy.int64)
    order = numpy.empty(n, dtype=numpy.int64)
    used = 0
    for p in periods:
        ph[:] = foldfast(t, p)
        work = fold_walk_into(p, ph, walk, buf, order, 10**12)
        if work < 0:
            continue
        used += 1
        numpy.testing.assert_array_equal(order, numpy.argsort(ph, kind="mergesort"))
    assert used >= len(periods) - 2


def test_regular_grid_rejects_irregular_data():
    rng = numpy.random.default_rng(2)
    assert regular_grid(numpy.sort(rng.uniform(0, 100, 5000))) is None
    assert regular_grid(numpy.repeat(numpy.arange(3000) * 0.03, 2)) is None
    sparse = numpy.sort(rng.choice(30000, 3000, replace=False)) * 0.01
    assert regular_grid(sparse) is None  # walk would visit 10x more slots
    assert regular_grid(numpy.arange(10) * 0.1) is None  # too short


def problem(t, weighted, seed=3):
    rng = numpy.random.default_rng(seed)
    y = 1 + rng.normal(0, 1e-3, len(t))
    y[(t % 3.1) < 0.08] -= 2e-3
    dy = numpy.ones(len(t))
    if weighted:
        dy += rng.uniform(-0.05, 0.05, len(t))
    ov, lc = cache(t, y, "default")
    return SearchProblem(
        t, y, dy, lc, ov, 1e-5,
        C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX, 0.01,
    )


@pytest.mark.parametrize("name", ["regular", "gappy", "jitter", "unsorted", "integer"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("backend", ["fused", "fused-pl"])
def test_kernel_results_bit_identical(name, weighted, backend):
    """Walk (forced at every period, including the budget fallback) versus the
    bucket path: identical permutation and bit-identical kernel results."""
    from transitleastsquares.backends import get_backend

    rng = numpy.random.default_rng(1)
    t = dict(grids(rng))[name][:3000]
    be = get_backend(backend)
    fp = be.prepare(problem(t, weighted))
    fp.set_walk()
    assert fp.walk_grid is not None
    span = numpy.ptp(t)
    multiples = fp.walk_grid[2] * numpy.array([360, 720])  # tied grid phases
    periods = numpy.concatenate((numpy.geomspace(0.3, span / 3.2, 25), multiples))
    fp.walk = fp.walk_grid[:5] + (0.0,)  # walk at every period
    n = len(t)
    with_walk = []
    for p in periods:
        with_walk.append(fp.search(p))
        perm = fp.workspace()[1][n + 1 : 2 * n + 1]
        numpy.testing.assert_array_equal(
            perm, numpy.argsort(foldfast(t, p), kind="mergesort")
        )
    fp.set_walk(False)
    without = [fp.search(p) for p in periods]
    assert with_walk == without


def test_crossover_and_switch(monkeypatch):
    rng = numpy.random.default_rng(1)
    t = dict(grids(rng))["regular"]
    walk = regular_grid(t)
    p_star = regular_grid_crossover(t, walk, numpy.ptp(t))
    assert p_star < 1.0  # exact grid: the walk needs no repair anywhere
    # restricted to the searched range: P* within it
    # (P = m * delta exactly: degenerate three-gap steps, the walk declines)
    rng_p = (2.03, 2.97)
    assert 2.03 <= regular_grid_crossover(t, walk, numpy.ptp(t), period_range=rng_p)
    fp = FusedProblem(problem(t, False))
    assert fp.walk[5] == numpy.inf  # not configured yet: the first search does it
    fp.search(1.3)
    assert fp.walk[5] == fp.walk_grid[5] < 1.0
    fp.set_walk(periods=[2.03, 2.57])
    assert 2.03 <= fp.walk[5] <= 2.57
    monkeypatch.setenv("TLS_WALK", "0")
    fp = FusedProblem(problem(t, False))
    fp.set_walk()
    assert fp.walk_grid is None and fp.walk[5] == numpy.inf
    # irregular data: never walks
    monkeypatch.delenv("TLS_WALK")
    t2 = numpy.sort(rng.uniform(0, 20, 3000))
    fp = FusedProblem(problem(t2, False))
    fp.set_walk()
    assert fp.walk_grid is None and fp.walk[5] == numpy.inf


def test_backends_plan_the_walk(monkeypatch):
    """ProcessPoolBackend.search configures the walk for the searched periods
    in the parent (FusedBackend.plan), before any worker starts."""
    from transitleastsquares.backends import get_backend

    rng = numpy.random.default_rng(1)
    t = dict(grids(rng))["regular"][:3000]
    be = get_backend("fused-pl")
    seen = []
    original = be.plan

    def plan(state, periods):
        original(state, periods)
        seen.append((state.walk[5], numpy.min(periods)))

    monkeypatch.setattr(be, "plan", plan)
    be.search(problem(t, False), numpy.array([1.53, 1.71, 2.03]))
    assert len(seen) == 1 and 1.53 <= seen[0][0] <= 2.03


def test_power_spectra_bit_identical(monkeypatch):
    from transitleastsquares import transitleastsquares

    rng = numpy.random.default_rng(1)
    t = dict(grids(rng))["gappy"]
    y = 1 + rng.normal(0, 1e-3, len(t))
    y[(t % 2.3) < 0.07] -= 2.5e-3
    results = []
    for flag in ("1", "0"):
        monkeypatch.setenv("TLS_WALK", flag)
        results.append(
            transitleastsquares(t, y, verbose=False).power(
                use_threads=1, show_progress_bar=False, verbose=False,
                backend=os.environ.get("TLS_BACKEND", "fused-pl"),
            )
        )
    a, b = results
    numpy.testing.assert_array_equal(a.chi2, b.chi2)
    numpy.testing.assert_array_equal(a.power, b.power)
    assert a.SDE == b.SDE and a.period == b.period and a.T0 == b.T0

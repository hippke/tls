"""The cache-blocked period fold equals the strided fold bit for bit."""

import numpy as np
import pytest

from transitleastsquares.gp.events import (
    compact_offsets,
    fold_groups,
    multiple_events_blocked,
    multiple_events_strided,
)


def _inputs(seed, M=4000, J=12):
    rng = np.random.default_rng(seed)
    delta = 0.01
    mstep = np.maximum(1, (np.arange(J) // 3) + 1).astype(np.int64)
    off = compact_offsets(M, mstep)
    total = off[-1]
    a = rng.normal(0, 1, total)
    b = rng.uniform(0.1, 2.0, total)
    gaps = rng.uniform(size=total) < 0.2  # rows without data: a = b = 0
    a[gaps] = 0.0
    b[gaps] = 0.0
    ab = np.column_stack([a, b])
    periods = np.sort(rng.uniform(0.3, 15.0, 400))
    jlo = rng.integers(0, J // 2, len(periods)).astype(np.int64)
    jhi = (jlo + rng.integers(0, J // 2, len(periods))).astype(np.int64)
    return ab, off, M, periods, delta, mstep, jlo, jhi


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("ntr_min", [1, 2, 3, 5])
@pytest.mark.parametrize("chunk,budget", [(16384, 1 << 20), (97, 1 << 12), (1, 64)])
def test_blocked_equals_strided(seed, ntr_min, chunk, budget):
    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs(seed)
    ref = multiple_events_strided(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, 1.0, False)
    gstart = fold_groups(periods, delta, budget=budget)
    new = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, gstart, chunk)
    for i in range(5):  # dchi2, best T0 index, best duration, sums A and B
        np.testing.assert_array_equal(new[i], ref[i])


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("depth_min", [0.0, 0.05, 0.5])
def test_blocked_equals_events_fold(seed, depth_min):
    from transitleastsquares.backends.events import fold_events

    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs(seed)
    gain, jb, A, B = fold_events(ab, off, M, periods, delta, mstep, jlo, jhi, 3, depth_min)
    new = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, 3,
                                  fold_groups(periods, delta), 97, depth_min)
    fit = gain > 0
    np.testing.assert_array_equal(new[0], gain)
    np.testing.assert_array_equal(new[2][fit], jb[fit])
    np.testing.assert_array_equal(new[3][fit], A[fit])
    np.testing.assert_array_equal(new[4][fit], B[fit])


def _inputs_coarse(seed, M=6000, J=12, empty=False):
    """As _inputs, with epoch steps of 4 and more (as on real light curves, where every duration's epoch grid
    is coarser than delta), and optionally periods without durations (jhi < jlo, as left by the FFA)."""
    ab, off, M, periods, delta, _, jlo, jhi = _inputs(seed, M, J)
    mstep = ((np.arange(J) // 3) + 4).astype(np.int64)
    off = compact_offsets(M, mstep)
    rng = np.random.default_rng(seed + 100)
    ab = np.column_stack([rng.normal(0, 1, off[-1]), rng.uniform(0.1, 2.0, off[-1])])
    ab[rng.uniform(size=off[-1]) < 0.2] = 0.0
    if empty:
        jhi = jhi.copy()
        jhi[::7] = jlo[::7] - 1
    return ab, off, M, periods, delta, mstep, jlo, jhi


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("ntr_min", [1, 3])
@pytest.mark.parametrize("chunk,budget", [(16384, 512 << 10), (97, 1 << 12), (1, 64)])
@pytest.mark.parametrize("empty", [False, True])
def test_blocked_used_groups_equal_strided(seed, ntr_min, chunk, budget, empty):
    """Groups sized at each period's coarsest used epoch grid (fold_groups with mstep, jlo, jhi): bit-identical
    to the strided fold and to the groups sized at the finest grid."""
    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs_coarse(seed, empty=empty)
    ref = multiple_events_strided(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, 1.0, False)
    for gstart in (fold_groups(periods, delta, mstep, jlo, jhi, budget=budget), fold_groups(periods, delta, budget=budget)):
        new = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, gstart, chunk)
        for i in range(5):
            np.testing.assert_array_equal(new[i], ref[i])


def test_fold_groups_used_budget():
    """The per-epoch sums of a group, counted at the grid used, fit the budget (or the group is one period);
    the groups are larger than those counted at the finest grid; at most gmax periods."""
    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs_coarse(3, empty=True)
    budget, gmax = 1 << 13, 16
    used = fold_groups(periods, delta, mstep, jlo, jhi, budget=budget, gmax=gmax)
    fine = fold_groups(periods, delta, budget=budget, gmax=gmax)
    assert used[0] == 0 and used[-1] == len(periods) and np.all(np.diff(used) > 0)
    assert len(used) < len(fine)
    for g in range(len(used) - 1):
        p0, p1 = used[g], used[g + 1]
        assert p1 - p0 <= gmax
        need = sum(16 * (int(periods[p] / (delta * mstep[jlo[p]:jhi[p] + 1].min())) + 1) if jhi[p] >= jlo[p] else 16
                   for p in range(p0, p1))
        assert need <= budget or p1 - p0 == 1


def test_blocked_coarse_used_groups():
    """multiple_events_blocked_coarse with coarse = 1 and the used-grid groups equals multiple_events_blocked."""
    from transitleastsquares.gp.events import multiple_events_blocked_coarse

    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs_coarse(4, empty=True)
    gstart = fold_groups(periods, delta, mstep, jlo, jhi)
    ref = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997, 0.05)
    new = multiple_events_blocked_coarse(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997, 0.05, None, 1)
    for i in range(5):
        np.testing.assert_array_equal(new[i], ref[i])

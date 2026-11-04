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

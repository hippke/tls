"""FFA + polish fold (gp.fold_ffa): the shift plan, the FFA against its plan,
and the fold against the brute-force fold (multiple_events_blocked): every
reported value is an exact brute-force path value, never above the brute
force, and equal to it for a clear periodic signal."""

import numpy as np
import pytest

from transitleastsquares.gp import fold_ffa as ff
from transitleastsquares.gp.events import compact_offsets, fold_groups, multiple_events_blocked
from transitleastsquares.grid import period_grid

MAX_SHIFT_ERROR = 3.0  # epoch steps (measured: see test_plan_error)


@pytest.mark.parametrize("m", [2, 3, 4, 5, 7, 16, 33, 100, 257, 1000, 2500])
def test_plan_error(m):
    eff = ff.effective_shifts(m)
    ideal = np.arange(m)[None, :] * np.arange(m)[:, None] / max(m - 1, 1)
    assert eff.shape == (m, m)
    assert np.all(eff[:, 0] == 0)  # row 0 is never shifted: the FFA phase is the epoch
    assert np.abs(eff - ideal).max() <= MAX_SHIFT_ERROR


@pytest.mark.parametrize("m,p", [(2, 5), (7, 3), (33, 17), (100, 41)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ffa_executes_its_plan(m, p, dtype):
    """The FFA of rows of single events (the last row partly beyond Mj)
    equals the explicit sums along its effective shifts."""
    rng = np.random.default_rng(m)
    Mj = m * p - p // 2
    ab = np.column_stack([rng.normal(size=Mj), rng.uniform(0, 2, Mj) * (rng.uniform(size=Mj) > 0.2)])
    x = np.zeros((m * p, 3))
    x[:Mj, :2] = ab
    x[:Mj, 2] = ab[:, 1] > 0
    x = x.reshape(m, p, 3)
    plan, tree = ff.flat_plan([m]), ff.flat_tree([m])
    out, tmp = np.empty((m, 3 * p), dtype), np.empty((m, 3 * p), dtype)
    ff._ffa(ab, 0, Mj, m, p, out, tmp, np.arange(m), *tree, *plan)
    eff = ff.effective_shifts(m).astype(int)
    i = np.arange(p)
    tol = 1e-12 if dtype == np.float64 else 1e-4
    for s in range(m):
        ref = sum(x[n, (i + eff[s, n]) % p] for n in range(m))
        np.testing.assert_allclose(out[s].reshape(p, 3), ref, rtol=tol, atol=tol)
    need = np.unique([0, m // 3, m - 1])  # the root merge only for these drifts
    out2 = np.full_like(out, np.nan)
    ff._ffa(ab, 0, Mj, m, p, out2, np.empty_like(out), need, *tree, *plan)
    np.testing.assert_array_equal(out2[need], out[need])


def _inputs(seed, M=20000, J=10, NP=None, signal=None):
    """Random single events (span M delta = 200 d) on the TLS period grid
    (NP: a log grid of NP periods instead)."""
    rng = np.random.default_rng(seed)
    delta = 0.01
    mstep = np.maximum(1, (np.arange(J) // 2) + 1).astype(np.int64)
    off = compact_offsets(M, mstep)
    total = off[-1]
    a = rng.normal(0, 1, total)
    b = rng.uniform(0.5, 1.5, total)
    gaps = rng.uniform(size=total) < 0.1  # rows without data
    a[gaps] = 0.0
    b[gaps] = 0.0
    if signal is not None:  # periodic signal in duration j0 and its neighbours
        P, T0, amp, j0 = signal
        for j in range(J):
            dj = mstep[j] * delta
            k = np.arange(off[j + 1] - off[j])
            ph = ((k * dj - T0) / P + 0.5) % 1 - 0.5
            hit = np.abs(ph * P) < 0.5 * dj
            a[off[j] + np.nonzero(hit)[0]] += amp * b[off[j] + np.nonzero(hit)[0]] / (1 + abs(j - j0))
    ab = np.column_stack([a, b])
    if NP is None:
        periods = np.sort(period_grid(1.0, 1.0, M * delta, 0.3, 60.0, oversampling_factor=3))
    else:
        periods = np.exp(np.linspace(np.log(0.3), np.log(60.0), NP))
    NP = len(periods)
    jlo = np.clip(rng.integers(0, 3, NP), 0, J - 1).astype(np.int64)
    jhi = np.clip(jlo + 4 + rng.integers(0, 3, NP), 0, J - 1).astype(np.int64)
    return ab, off, M, periods, delta, mstep, jlo, jhi


def _check_exact(ab, off, periods, delta, mstep, res, ntr_min, depth_min):
    """Every reported value is the brute-force path of its epoch and duration."""
    dchi2, k0, jb, A, B, _ = res
    for q in np.nonzero(dchi2 > 0)[0]:
        j = jb[q]
        assert k0[q] % mstep[j] == 0
        g, a, b = ff.exact_path(ab, off[j], off[j + 1] - off[j], periods[q], mstep[j] * delta,
                                k0[q] // mstep[j], ntr_min, depth_min)
        assert (g, a, b) == (dchi2[q], A[q], B[q])


@pytest.mark.parametrize("screen,coarse", [(False, 1), (True, 1), (False, 2), (True, 2)])
@pytest.mark.parametrize("ntr_min,depth_min", [(2, 0.0), (3, 0.05)])
def test_fold_ffa_bounded_and_exact(screen, coarse, ntr_min, depth_min):
    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs(0)
    gstart = fold_groups(periods, delta)
    ref = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, gstart, 997, depth_min)
    stats = {}
    new = ff.fold_ffa(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, gstart, 997, depth_min,
                      screen=screen, coarse=coarse, stats=stats)
    assert stats["blocks"] > 50  # the FFA is used
    assert stats["cells_ffa"] + stats["cells_brute"] + stats["cells_polish"] < 0.6 * stats["cells_full_brute"]  # ~2x
    assert np.all(new[0] <= ref[0])
    _check_exact(ab, off, periods, delta, mstep, new, ntr_min, depth_min)
    if not screen and coarse == 1:  # screen/coarse assume correlated neighbours (real data), not this noise
        # on pure noise the FFA often finds the brute-force maximum, but not always
        assert np.mean(new[0] == ref[0]) > 0.3
        assert np.median(new[0] / np.where(ref[0] > 0, ref[0], 1)) > 0.97


@pytest.mark.parametrize("screen,coarse", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("P", [0.9137, 2.3456, 7.01])
def test_fold_ffa_finds_signal(screen, coarse, P):
    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs(1, signal=(P, 0.37, 3.0, 3))
    gstart = fold_groups(periods, delta)
    ref = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997)
    new = ff.fold_ffa(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997, screen=screen, coarse=coarse)
    i = np.argmax(ref[0])
    assert abs(periods[i] / P - 1) < 0.01
    assert np.argmax(new[0]) == i
    assert new[0][i] == ref[0][i]  # at the signal's period the brute-force maximum


@pytest.mark.parametrize("screen", [False, True])
def test_coarse_kernel_with_step_1_is_the_blocked_kernel(screen):
    from transitleastsquares.gp.events import multiple_events_blocked_coarse

    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs(3, NP=800)
    gstart = fold_groups(periods, delta)
    sh = jhi if screen else None
    ref = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997, 0.05, sh)
    new = multiple_events_blocked_coarse(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997, 0.05, sh, 1)
    for i in range(5):
        np.testing.assert_array_equal(new[i], ref[i])


def test_fold_ffa_without_blocks_is_the_brute_force(monkeypatch):
    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs(2, NP=500)
    monkeypatch.setattr(ff, "MIN_ROWS", 10**9)
    gstart = fold_groups(periods, delta)
    ref = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997)
    new = ff.fold_ffa(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997)
    for i in range(5):
        np.testing.assert_array_equal(new[i], ref[i])


# ------------------------------------------------------------ integration
def test_gp_search_fold_ffa():
    """gp.search: same best signal, every period's dchi2 <= the full fold."""
    from gp_cases import ROT_SYN, synthetic
    from transitleastsquares import gp

    t, f, e = synthetic(days=120.0, seed=5)
    kw = dict(baseline=ROT_SYN, period_min=0.5, period_max=20.0)
    ref = gp.search(t, f, e, **kw)
    for fold in ("ffa", "ffa-screen", "ffa-screen-coarse"):
        new = gp.search(t, f, e, fold=fold, **kw)
        assert new.period == ref.period
        assert new.T0 == ref.T0 and new.duration == ref.duration
        assert np.all(new.delta_chi2 <= ref.delta_chi2)
        assert np.mean(new.delta_chi2 == ref.delta_chi2) > 0.3


def test_events_ffa_backend():
    """TLS backend "events-ffa": same best period as "events", chi2 never below it."""
    from transitleastsquares import transitleastsquares
    from transitleastsquares.transit_model import light_curve

    rng = np.random.default_rng(3)
    t = np.arange(0.0, 200.0, 0.0204)
    f = 1.0 + rng.normal(0, 1e-3, len(t))
    f *= light_curve(t, 0.4, 1.7321, 0.03, 6.0, 90.0, 0.0, 90.0, [0.4, 0.2], "quadratic")
    kw = dict(use_threads=4, show_progress_bar=False, verbose=False, period_min=0.5, period_max=40.0)
    ref = transitleastsquares(t, f, np.full(len(t), 1e-3)).power(backend="events", **kw)
    new = transitleastsquares(t, f, np.full(len(t), 1e-3)).power(backend="events-ffa", **kw)
    assert abs(ref.period / 1.7321 - 1) < 1e-3
    assert new.period == ref.period and new.SDE > 0.97 * ref.SDE
    assert np.all(np.asarray(new.chi2) >= np.asarray(ref.chi2))


def test_gp_search_bin_fraction():
    """bin_fraction (approximate, opt-in): weighted bins keep the signal at a
    short cadence; same best period, values close to the unbinned search."""
    from gp_cases import ROT_SYN, synthetic
    from transitleastsquares.gp.core import _search

    t, f, e = synthetic(days=20.0, seed=7, cadence=25 / 86400)
    model = ROT_SYN._model(t, e)
    ref = _search(t, f, e, model, period_min=1.0, period_max=8.0)
    new = _search(t, f, e, model, period_min=1.0, period_max=8.0, bin_fraction=1 / 30)
    assert new["period"] == ref["period"]
    assert abs(new["snr"] / ref["snr"] - 1) < 0.02
    ok = ref["dchi2"] > 0
    assert np.median(np.sqrt(new["dchi2"][ok] / ref["dchi2"][ok])) > 0.99

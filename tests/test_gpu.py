"""GPU kernels (transitleastsquares.gpu) against the CPU kernels: equal up to
FP32 rounding (skipped without a CUDA device)."""

import numpy as np
import pytest

from transitleastsquares import gpu

pytestmark = pytest.mark.skipif(not gpu.available(), reason="no CUDA device")


def _white_inputs(days=60.0, cad=0.0204, seed=4):
    rng = np.random.default_rng(seed)
    t = np.sort(np.arange(0, days, cad)[rng.uniform(size=int(np.ceil(days / cad))) > 0.05])
    r = rng.normal(0, 1e-3, len(t))
    w = rng.uniform(0.7, 1.3, len(t))
    return t, r, w, cad


def test_single_events_white():
    from transitleastsquares.backends.events import single_events_white
    from transitleastsquares.gp.events import compact_offsets, template_table
    from transitleastsquares.gpu import driver

    t, r, w, cad = _white_inputs()
    delta = cad / 4
    durs = np.array([0.02, 0.05, 0.1, 0.2, 0.4])
    mstep = np.maximum(1, np.floor(0.1 * durs / delta)).astype(np.int64)
    nsubs = np.array([7, 7, 7, 1, 1], np.int64)
    tc = t[0] + delta * np.arange(int((t[-1] - t[0]) / delta) + 1)
    off = compact_offsets(len(tc), mstep)
    table = template_table()
    ref = single_events_white(t, r, w, tc, durs, nsubs, table, cad, mstep, off)
    new = driver.single_events_white(t, r, w, t[0], delta, durs, nsubs, table, cad, mstep, off).copy_to_host()
    ok = ref[:, 1] > 0
    np.testing.assert_allclose(new[ok, 1], ref[ok, 1], rtol=2e-5)
    assert np.max(np.abs(new[ok, 0] - ref[ok, 0]) / np.sqrt(ref[ok, 1])) < 1e-4 * np.max(np.abs(r) * np.sqrt(w))


@pytest.mark.parametrize("name", ["k2_rot", "k2_spline", "syn_rot"])
def test_single_events_gp(name):
    from gp_cases import CASES
    from transitleastsquares.gp.events import compact_offsets, single_events_split, tail_operators, template_table
    from transitleastsquares.gp.statespace import kalman_whiten, smoothed_mean, transition_matrices
    from transitleastsquares.gpu import driver

    load, baseline = CASES[name]
    t, f, e = load()
    y, yerr = (f - 1) * 1e3, e * 1e3
    model = baseline._model(t, e)
    kinds, starts, pars, P0 = model.blocks()
    R = yerr**2 + model.jitter**2
    A = transition_matrices(t, kinds, starts, pars, model.dim)
    Kg, S, ey = kalman_whiten(t, y, R, kinds, starts, pars, P0, 1)
    eyd = ey[:model.n_diffuse].copy()
    ey[:model.n_diffuse] = 0.0
    W, _ = tail_operators(A, Kg, S, model.H, ey, 1)
    z = (y - smoothed_mean(t, y, R, kinds, starts, pars, P0, 1)) / R
    cad = np.median(np.diff(t))
    delta = cad / 4
    durs = np.array([0.03, 0.06, 0.12, 0.25])
    mstep = np.maximum(1, np.floor(0.1 * durs / delta)).astype(np.int64)
    tc = t.min() + delta * np.arange(int((t.max() - t.min()) / delta) + 1)
    off = compact_offsets(len(tc), mstep)
    table = template_table()
    ref = single_events_split(t, z, eyd, A, Kg, S, model.H, W, tc, durs, table, cad, 7, mstep, off)
    new = driver.single_events_gp(t, z, eyd, A, Kg, S, model.H, W, t.min(), delta, durs, 7, table, cad,
                                  mstep, off).copy_to_host()
    ok = ref[:, 1] > 1e-6 * ref[:, 1].max()
    rel_b = np.abs(new[ok, 1] - ref[ok, 1]) / ref[ok, 1]
    err_a = np.abs(new[ok, 0] - ref[ok, 0]) / np.sqrt(ref[ok, 1])
    print(name, "b: rel. error q99.9", np.quantile(rel_b, 0.999), "max", rel_b.max(),
          "(at b/b_max", ref[ok, 1][np.argmax(rel_b)] / ref[ok, 1].max(), ") | a/sqrt(b): q99.9",
          np.quantile(err_a, 0.999), "max", err_a.max())
    assert np.quantile(rel_b, 0.999) < 1e-5 and rel_b.max() < 1e-3  # SNR errors <= ~5e-4
    assert np.quantile(err_a, 0.999) < 1e-4 and err_a.max() < 1e-3


def _fold_inputs():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_gp_fold_ffa import _inputs

    ab, off, M, periods, delta, mstep, jlo, jhi = _inputs(0, signal=(1.2345, 0.37, 1.0, 3))
    return ab.astype(np.float32).astype(float), off, M, periods, delta, mstep, jlo, jhi


@pytest.mark.parametrize("fold", ["blocked", "ffa", "blocked-screen-coarse", "ffa-screen-coarse", "blocked-coarse"])
def test_fold_gpu(fold):
    from numba import cuda
    from transitleastsquares.gp.events import fold_groups, multiple_events_blocked
    from transitleastsquares.gp.fold_ffa import fold_ffa, fold_options
    from transitleastsquares.gpu import driver

    ab, off, M, periods, delta, mstep, jlo, jhi = _fold_inputs()
    gstart = fold_groups(periods, delta)
    opt = fold_options(fold)
    if fold != "blocked":
        ref = fold_ffa(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997, 0.05, **opt)
    else:
        ref = multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, 2, gstart, 997, 0.05)
    d_ab = cuda.to_device(ab.astype(np.float32))
    new = driver.fold(d_ab, off, M, periods, delta, mstep, jlo, jhi, 2, 0.05, **opt)
    close = np.isclose(new[0], ref[0], rtol=1e-4, atol=1e-4)
    if opt["screen"]:  # FP32 vs FP64 may pick another scout in a near-tie, then refine another neighbour
        assert close.mean() > 0.995
    else:
        assert close.all()
    same = (new[1] == ref[1]) & (new[2] == ref[2])
    print(fold, "same best epoch and duration:", same.mean(), "values within 1e-4:", close.mean())
    assert same.mean() > 0.99
    assert np.argmax(new[0]) == np.argmax(ref[0])
    if fold != "blocked":  # every value an exact (FP32) brute-force value: never above the brute fold
        brute = driver.fold(d_ab, off, M, periods, delta, mstep, jlo, jhi, 2, 0.05, ffa=False)
        assert np.all(new[0] <= brute[0] * (1 + 1e-6) + 1e-6)


def test_events_gpu_backend(monkeypatch):
    """TLS backend "events-gpu" = "events" up to FP32 rounding (and with the
    FFA fold = "events-ffa")."""
    from transitleastsquares import transitleastsquares
    from transitleastsquares.transit_model import light_curve

    rng = np.random.default_rng(3)
    t = np.arange(0.0, 200.0, 0.0204)
    f = 1.0 + rng.normal(0, 1e-3, len(t))
    f *= light_curve(t, 0.4, 1.7321, 0.03, 6.0, 90.0, 0.0, 90.0, [0.4, 0.2], "quadratic")
    kw = dict(use_threads=1, show_progress_bar=False, verbose=False, period_min=0.5, period_max=40.0)
    for cpu, fold in (("events", "blocked"), ("events-ffa", "ffa")):
        monkeypatch.setenv("TLS_EVENTS_FOLD", fold)
        ref = transitleastsquares(t, f, np.full(len(t), 1e-3)).power(backend=cpu, **kw)
        new = transitleastsquares(t, f, np.full(len(t), 1e-3)).power(backend="events-gpu", **kw)
        assert new.period == ref.period and new.T0 == ref.T0 and new.duration == ref.duration
        assert abs(new.SDE / ref.SDE - 1) < 1e-3
        np.testing.assert_allclose(np.asarray(new.chi2), np.asarray(ref.chi2), rtol=1e-5)


@pytest.mark.parametrize("fold", ["blocked", "ffa"])
def test_gp_search_gpu(fold):
    """gp.search(device="gpu") = the CPU search up to FP32 rounding."""
    from gp_cases import ROT_SYN, synthetic
    from transitleastsquares import gp

    t, f, e = synthetic(days=120.0, seed=5)
    kw = dict(baseline=ROT_SYN, period_min=0.5, period_max=20.0, fold=fold)
    ref = gp.search(t, f, e, **kw)
    new = gp.search(t, f, e, device="gpu", **kw)
    assert new.period == ref.period and new.T0 == ref.T0 and new.duration == ref.duration
    assert abs(new.SDE / ref.SDE - 1) < 1e-3
    np.testing.assert_allclose(new.delta_chi2, ref.delta_chi2, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("name", ["k2_rot", "k2_spline", "syn_rot"])
def test_tail_operators_gpu(name):
    """The FP32 segment-parallel tail recursion and numerator weights on the
    GPU equal the CPU's (FP64)."""
    from gp_cases import CASES
    from transitleastsquares.gp.events import numerator_weights, tail_operators
    from transitleastsquares.gp.statespace import kalman_whiten, transition_matrices
    from transitleastsquares.gpu import driver

    load, baseline = CASES[name]
    t, f, e = load()
    y, yerr = (f - 1) * 1e3, e * 1e3
    model = baseline._model(t, e)
    kinds, starts, pars, P0 = model.blocks()
    A = transition_matrices(t, kinds, starts, pars, model.dim)
    Kg, S, ey = kalman_whiten(t, y, yerr**2 + model.jitter**2, kinds, starts, pars, P0, 1)
    ey[:model.n_diffuse] = 0.0
    W, w = tail_operators(A, Kg, S, model.H, ey, 1)
    Wd, wd, zd, _ = driver.tail_operators_gpu(A, Kg, S, model.H, ey)
    Wg, wg, zg = Wd.copy_to_host(), wd.copy_to_host(), zd.copy_to_host()
    scale = np.abs(W).max(axis=(1, 2))[:, None, None] + 1e-30
    z = numerator_weights(ey, S, Kg, w)
    print(name, "W rel", np.max(np.abs(Wg - W) / scale), "z rel", np.max(np.abs(zg - z)) / np.abs(z).max())
    assert np.max(np.abs(Wg - W) / scale) < 1e-4
    assert np.max(np.abs(zg - z)) < 1e-4 * np.abs(z).max()


@pytest.mark.parametrize("name", ["k2_rot", "syn_rot"])
def test_trend_gpu(name):
    """Baseline.trend(device="gpu") (f = y - R C^-1 y, no smoother) equals the
    smoother's trend."""
    from gp_cases import CASES

    load, baseline = CASES[name]
    t, f, e = load()
    use = np.ones(len(t), bool)
    use[len(t) // 3: len(t) // 3 + 20] = False  # a masked stretch (as the transits of the best signal)
    ref = baseline.trend(t, f, e, use)
    new = baseline.trend(t, f, e, use, device="gpu")
    amp = np.ptp(ref)
    print(name, "max |trend_gpu - trend| / amplitude", np.max(np.abs(new - ref)) / amp)
    assert np.max(np.abs(new - ref)) < 1e-5 * amp

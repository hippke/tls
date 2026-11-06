"""Single events through C^-1 y = N^-1 (y - f) (gp.events.single_events_split)
equal the Kalman single events: b bit for bit, a up to rounding."""

import numpy as np
import pytest

from gp_cases import CASES, SEARCH
from transitleastsquares.gp.core import _search
from transitleastsquares.gp.events import (
    compact_offsets,
    numerator_weights,
    single_events_compact,
    single_events_split,
    tail_operators,
    template_table,
)
from transitleastsquares.gp.statespace import kalman_whiten, smoothed_mean, transition_matrices


@pytest.mark.parametrize("name", sorted(CASES))
def test_single_events_split(name):
    load, baseline = CASES[name]
    t, f, e = load()
    y, yerr = (f - 1) * 1e3, e * 1e3
    model = baseline._model(t, e)
    kinds, starts, pars, P0 = model.blocks()
    R = yerr**2 + model.jitter**2
    A = transition_matrices(t, kinds, starts, pars, model.dim)
    K, S, ey = kalman_whiten(t, y, R, kinds, starts, pars, P0, 1)
    eyd = ey[:model.n_diffuse].copy()
    ey[:model.n_diffuse] = 0.0
    W, w = tail_operators(A, K, S, model.H, ey, 1)
    cad = np.median(np.diff(t))
    delta = cad / 4
    tc = t.min() + delta * np.arange(int((t.max() - t.min()) / delta) + 1)
    durs = np.array([0.03, 0.06, 0.12, 0.25])
    mstep = np.maximum(1, np.floor(0.1 * durs / delta)).astype(np.int64)
    off = compact_offsets(len(tc), mstep)
    table = template_table()
    ref = single_events_compact(t, ey, A, K, S, model.H, W, w, tc, durs, table, cad, 7, mstep, off)
    z = (y - smoothed_mean(t, y, R, kinds, starts, pars, P0, 1)) / R
    new = single_events_split(t, z, eyd, A, K, S, model.H, W, tc, durs, table, cad, 7, mstep, off)
    np.testing.assert_array_equal(new[:, 1], ref[:, 1])
    # events whose template touches the diffuse start (the first n_diffuse
    # samples): the spline's 3 diffuse states are ill-conditioned there, a
    # through the smoother differs by up to a few % (rotation: exact)
    j = np.searchsorted(off, np.arange(off[-1]), side="right") - 1
    tcm = tc[(np.arange(off[-1]) - off[j]) * mstep[j]]
    edge = tcm - 0.5 * durs[j] - 0.5 * cad < t[model.n_diffuse]
    ok = (ref[:, 1] > 0) & (~edge if "spline" in name else True)
    err = np.abs(new[ok, 0] - ref[ok, 0]) / np.sqrt(ref[ok, 1])
    assert err.max() < (5e-6 if "spline" in name else 1e-9)  # spline: ill-conditioned (diffuse prior)
    # z from the filter and the tail recursion (numerator_weights): exact everywhere
    zt = numerator_weights(ey, S, K, w)
    new2 = single_events_split(t, zt, np.zeros(0), A, K, S, model.H, W, tc, durs, table, cad, 7, mstep, off)
    np.testing.assert_array_equal(new2[:, 1], ref[:, 1])
    ok = ref[:, 1] > 0
    assert np.max(np.abs(new2[ok, 0] - ref[ok, 0]) / np.sqrt(ref[ok, 1])) < 1e-12


@pytest.mark.parametrize("name", sorted(CASES))
def test_search_single_split(name):
    load, baseline = CASES[name]
    t, f, e = load()
    model = baseline._model(t, e)
    ref = _search(t, f, e, model, **SEARCH)
    new = _search(t, f, e, model, single="split", **SEARCH)
    assert new["period"] == ref["period"] and new["duration"] == ref["duration"]
    np.testing.assert_allclose(new["dchi2"], ref["dchi2"], rtol=1e-9, atol=1e-9)


def _kepler_like(n=6000, seed=3):
    """Real time stamps: a slow drift of the step and rounding per point, dropouts of a few cadences, long gaps."""
    rng = np.random.default_rng(seed)
    dt = 0.0204 * (1 + 4e-5 * np.sin(np.arange(n) / 800.0)) * (1 + 1e-7 * rng.normal(size=n))
    dt[rng.integers(0, n, 150)] *= rng.integers(2, 6, 150)
    dt[n // 3] += 5.0
    dt[2 * n // 3] += 1.3
    t = np.cumsum(dt)
    e = np.full(n, 0.2)
    y = 2.0 * np.sin(2 * np.pi * t / 3.1) + rng.normal(0, 0.2, n)
    return t, y, e


def test_step_table_pools_repeated_steps():
    """Steps within the tolerance share an entry wherever they are: a gap's multiples do not add entries."""
    from gp_cases import ROT_SYN
    from transitleastsquares.gp.statespace import _block_sizes, kalman_loglik, kalman_loglik_table, step_table, step_transitions

    t, y, e = _kepler_like()
    idx, steps = step_table(t)
    dt = np.diff(t)
    assert len(np.unique(dt)) > 5000 and len(steps) < 100
    assert steps[0] == 0.0 and idx[0] == 0
    assert np.max(np.abs(steps[idx[1:]] / dt - 1)) < 1.5e-5
    idx0, steps0 = step_table(t, 0.0)
    np.testing.assert_array_equal(steps0[idx0[1:]], dt)
    model = ROT_SYN._model(t, e / 1e3)
    kinds, starts, pars, P0 = model.blocks()
    R = e**2 + model.jitter**2
    At, Qt = step_transitions(steps, kinds, pars)
    for k in range(len(kinds)):
        np.testing.assert_array_equal(At[0, k], np.eye(2) if _block_sizes(kinds)[k] == 2 else np.diag([1.0, 0.0]))
    np.testing.assert_array_equal(Qt[0], np.zeros_like(Qt[0]))
    ll = kalman_loglik(t, y, R, kinds, starts, pars, P0, model.n_diffuse)
    llt = kalman_loglik_table(idx, At, Qt, y, R, kinds, starts, P0, model.n_diffuse)
    print("table - exact log likelihood", llt - ll)
    assert abs(llt - ll) < 0.02
    At0, Qt0 = step_transitions(steps0, kinds, pars)
    np.testing.assert_allclose(kalman_loglik_table(idx0, At0, Qt0, y, R, kinds, starts, P0, model.n_diffuse), ll, rtol=1e-12)


@pytest.mark.parametrize("name", ["rot", "spline"])
def test_mean_adjoint_equals_smoother(name):
    """y - R C^-1 y equals the RTS smoother, also across masked points and at a masked start."""
    from gp_cases import ROT_SYN, SPLINE_SYN
    from transitleastsquares.gp.statespace import gp_mean, mean_adjoint, smoothed_mean

    t, y, e = _kepler_like()
    model = (ROT_SYN if name == "rot" else SPLINE_SYN)._model(t, e / 1e3)
    kinds, starts, pars, P0 = model.blocks()
    use = np.ones(len(t), bool)
    use[:40] = False
    use[1000:1100] = False
    use[::97] = False
    yerr = np.where(use, e, 1e9)
    R = yerr**2 + model.jitter**2
    ym = y * use
    ref = smoothed_mean(t, ym, R, kinds, starts, pars, P0, 1)
    np.testing.assert_allclose(mean_adjoint(t, ym, R, kinds, starts, pars, P0), ref, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(gp_mean(model, t, ym, yerr), mean_adjoint(t, ym, R, kinds, starts, pars, P0))
    np.testing.assert_allclose(gp_mean(model, t, ym, yerr, 2), ref, rtol=0, atol=1e-5)


def test_t14_durations():
    """One call for all periods gives T14(...) * P per period."""
    from transitleastsquares.grid import T14, T14_durations

    periods = np.geomspace(0.5, 200.0, 50)
    for small in (True, False):
        ref = np.array([T14(1.1, 0.9, P, small=small) * P for P in periods])
        np.testing.assert_allclose(T14_durations(1.1, 0.9, periods, small), ref, rtol=1e-14)

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


def test_riccati_pieces():
    """transition_table, riccati_head, riccati_keys (bins of 1 point: exact)
    and mean_recursion reproduce kalman_whiten."""
    from gp_cases import CASES
    from transitleastsquares.gp.statespace import (
        _block_sizes, _gains_segment, kalman_whiten, mean_recursion, riccati_head, riccati_keys, transition_table,
    )

    load, baseline = CASES["k2_rot"]
    t, f, e = load()
    y, yerr = (f - 1) * 1e3, e * 1e3
    model = baseline._model(t, e)
    kinds, starts, pars, P0 = model.blocks()
    R = yerr**2 + model.jitter**2
    Kg, S, ey = kalman_whiten(t, y, R, kinds, starts, pars, P0, 1)
    idx, At, Qt = transition_table(t, kinds, pars)
    print("table entries", len(At), "of", len(t))
    assert len(At) < len(t) / 2  # steps within the tolerance share an entry
    Atx = transition_table(t, kinds, pars, 0.0)
    n, d, n0 = len(t), P0.shape[0], 100
    K2, iS2, ey2 = np.empty((n, d)), np.empty(n), np.empty(n)
    x, P = riccati_head(t, y, R, kinds, starts, pars, P0, n0, K2, iS2, ey2)
    np.testing.assert_allclose(K2[:n0], Kg[:n0], rtol=1e-12)
    np.testing.assert_allclose(ey2[:n0], ey[:n0], rtol=1e-12)
    edges = np.array([n0 + 50, n0 + 400, n // 2], np.int64)
    gap = 2.0 * np.median(np.diff(t))
    Pk = riccati_keys(t, R, kinds, starts, pars, P, n0, edges, 1, gap)
    Kx, Sx, ex = np.empty((n, d)), np.empty(n), np.empty(n)
    for s, e_ in enumerate(edges):
        xs, Ps = np.zeros(d), P0.copy()
        _gains_segment(t, y, R, kinds, starts, pars, xs, Ps, 0, int(e_), Kx, Sx, ex)
        np.testing.assert_allclose(Pk[s], Ps, rtol=1e-6, atol=1e-9 * np.abs(Ps).max())
    iS = 1.0 / np.sqrt(S)
    mean_recursion(Atx[1], Atx[0], y, starts, _block_sizes(kinds), Kg, iS, x.copy(), n0, ey2)
    np.testing.assert_allclose(ey2, ey, rtol=1e-6, atol=1e-8)
    mean_recursion(At, idx, y, starts, _block_sizes(kinds), Kg, iS, x.copy(), n0, ey2)
    print("ey with the shared steps", np.abs(ey2 - ey).max())
    assert np.abs(ey2 - ey).max() < 1e-3


def test_t14_durations():
    """One call for all periods gives T14(...) * P per period."""
    from transitleastsquares.grid import T14, T14_durations

    periods = np.geomspace(0.5, 200.0, 50)
    for small in (True, False):
        ref = np.array([T14(1.1, 0.9, P, small=small) * P for P in periods])
        np.testing.assert_allclose(T14_durations(1.1, 0.9, periods, small), ref, rtol=1e-14)

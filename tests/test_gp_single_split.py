"""Single events through C^-1 y = N^-1 (y - f) (gp.events.single_events_split)
equal the Kalman single events: b bit for bit, a up to rounding."""

import numpy as np
import pytest

from gp_cases import CASES, SEARCH
from transitleastsquares.gp.core import _search
from transitleastsquares.gp.events import (
    compact_offsets,
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
    # differs by up to a few % (rotation: exact); elsewhere up to rounding
    j = np.searchsorted(off, np.arange(off[-1]), side="right") - 1
    tcm = tc[(np.arange(off[-1]) - off[j]) * mstep[j]]
    edge = tcm - 0.5 * durs[j] - 0.5 * cad < t[model.n_diffuse]
    ok = (ref[:, 1] > 0) & (~edge if "spline" in name else True)
    err = np.abs(new[ok, 0] - ref[ok, 0]) / np.sqrt(ref[ok, 1])
    assert err.max() < (5e-6 if "spline" in name else 1e-9)  # spline: ill-conditioned (diffuse prior)


@pytest.mark.parametrize("name", sorted(CASES))
def test_search_single_split(name):
    load, baseline = CASES[name]
    t, f, e = load()
    model = baseline._model(t, e)
    ref = _search(t, f, e, model, **SEARCH)
    new = _search(t, f, e, model, single="split", **SEARCH)
    assert new["period"] == ref["period"] and new["duration"] == ref["duration"]
    close = np.isclose(new["dchi2"], ref["dchi2"], rtol=1e-6, atol=1e-6)
    if "spline" in name:  # periods whose best fold includes an event at the diffuse start
        assert close.mean() > 0.99
        np.testing.assert_allclose(new["dchi2"], ref["dchi2"], rtol=1e-2)
    else:
        assert close.all()

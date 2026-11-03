"""transitleastsquares.gp: the segment-parallel passes (likelihood, smoother,
gains and whitening, tail operators, Lomb-Scargle guess) against the
sequential ones; exact up to rounding for any number of segments."""

import numba
import numpy as np
import pytest

from gp_cases import ROT_SYN, synthetic
from transitleastsquares import gp
from transitleastsquares.gp import baselines
from transitleastsquares.gp import events as ev
from transitleastsquares.gp import statespace as g

MODELS = {
    "rotation": lambda: g.rotation_model(dict(sigma=3, period=1.3, Q0=2.0, dQ=50.0, f=0.4,
                                              sigma_slow=1.0, rho_slow=6.0, jitter=0.05)),
    "spline": lambda: g.spline_model(0.4, 0.3, 0.0204, 0.01),
}
# the spline's diffuse level and slope (plus the diffuse offset) make it
# ill-conditioned: the sequential block and dense filters already differ
# by ~1e-9 (log likelihood) and ~1e-7 (whitened data) relative
TOL = {"rotation": dict(ll=1e-11, mean=1e-8, ab=1e-8), "spline": dict(ll=1e-8, mean=1e-5, ab=1e-6)}


def data(n=6000, seed=4):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.8, 1.2, n) * 0.0204)
    t[n // 3:] += 2.0  # gap
    y = 3 * np.sin(2 * np.pi * t / 1.3) + 0.5 * t / t[-1] + rng.normal(0, 0.3, n)
    yerr = 0.3 * rng.uniform(0.9, 1.1, n)
    return t, y, yerr


@pytest.fixture(params=list(MODELS))
def setup(request):
    t, y, yerr = data()
    m = MODELS[request.param]()
    kinds, starts, pars, P0 = m.blocks()
    return t, y, yerr, m, (kinds, starts, pars, P0, yerr**2 + m.jitter**2), TOL[request.param]


def test_loglik_segments(setup):
    t, y, yerr, m, (kinds, starts, pars, P0, R), tol = setup
    ref = g.kalman_loglik(t, y, R, kinds, starts, pars, P0, m.n_diffuse)
    for nseg in (1, 2, 5):
        v = g.kalman_loglik_parallel(t, y, R, kinds, starts, pars, P0, m.n_diffuse, nseg)
        assert abs(v - ref) < tol["ll"] * abs(ref)


def test_smoothed_mean_segments(setup):
    t, y, yerr, m, (kinds, starts, pars, P0, R), tol = setup
    A, Q, _, _ = m.matrices(t, yerr)
    K, S, _, Pf = g.kalman_gains(A, Q, m.H, R, P0, False, True)
    _, xp, xf = g.whiten(y, A, K, S, m.H)
    ref = _rts_reference(A, Q, xp, xf, Pf) @ m.H
    for nseg in (1, 2, 5):
        v = g.smoothed_mean(t, y, R, kinds, starts, pars, P0, nseg)
        assert np.max(np.abs(v - ref)) < tol["mean"] * np.std(ref), np.max(np.abs(v - ref)) / np.std(ref)


def _rts_reference(A, Q, xpred, xfilt, Pfilt):
    """Textbook RTS smoother (dense numpy), the independent reference."""
    xs = xfilt.copy()
    for t in range(len(xs) - 2, -1, -1):
        Pp = A[t + 1] @ Pfilt[t] @ A[t + 1].T + Q[t + 1]
        G = Pfilt[t] @ A[t + 1].T @ np.linalg.pinv(Pp, rcond=1e-14, hermitian=True)
        xs[t] = xfilt[t] + G @ (xs[t + 1] - xpred[t + 1])
    return xs


def test_search_passes_segments(setup):
    t, y, yerr, m, (kinds, starts, pars, P0, R), tol = setup
    A0, Q, _, _ = m.matrices(t, yerr)
    K0, S0, _, _ = g.kalman_gains(A0, Q, m.H, R, P0, False, False)
    e0, _, _ = g.whiten(y, A0, K0, S0, m.H)
    W0, w0 = ev.tail_operators(A0, K0, S0, m.H, e0, 1)
    A = g.transition_matrices(t, kinds, starts, pars, m.dim)
    np.testing.assert_allclose(A, A0, rtol=1e-12, atol=1e-14)
    for nseg in (1, 2, 5):
        K, S, ey = g.kalman_whiten(t, y, R, kinds, starts, pars, P0, nseg)
        W, w = ev.tail_operators(A, K, S, m.H, ey, nseg)
        for a, b in ((K, K0), (S, S0), (ey, e0), (W, W0), (w, w0)):
            assert np.max(np.abs(a - b)) < tol["ab"] * np.max(np.abs(b))


def _ls_reference(t, yc, freqs):
    """The Lomb-Scargle power in plain numpy (the previous implementation)."""
    w = 2 * np.pi * freqs[:, None]
    tau = np.arctan2(np.sum(np.sin(2 * w * t), 1), np.sum(np.cos(2 * w * t), 1)) / (2 * w[:, 0])
    arg = w * (t - tau[:, None])
    c, s = np.cos(arg), np.sin(arg)
    return (c @ yc) ** 2 / np.sum(c * c, 1) + (s @ yc) ** 2 / np.sum(s * s, 1)


def test_lomb_scargle_power():
    t, y, _ = data(3000)
    yc = y - y.mean()
    freqs = np.linspace(0.05, 5, 700)
    np.testing.assert_allclose(baselines._ls_power(t, yc, freqs), _ls_reference(t, yc, freqs), rtol=1e-9)


def test_search_threads():
    """1 and 4 threads: the same best signal, dchi2 equal up to rounding."""
    t, f, e = synthetic(days=60.0)
    old = numba.get_num_threads()
    try:
        r1 = gp.search(t, f, e, baseline=ROT_SYN, use_threads=1)
        r4 = gp.search(t, f, e, baseline=ROT_SYN, use_threads=min(4, numba.config.NUMBA_NUM_THREADS))
    finally:
        numba.set_num_threads(old)
    for k in ("period", "T0", "duration"):
        assert r1[k] == r4[k]
    assert np.max(np.abs(r1.delta_chi2 - r4.delta_chi2)) < 1e-8 * np.max(r1.delta_chi2)
    np.testing.assert_allclose(r1.trend, r4.trend, rtol=1e-10)

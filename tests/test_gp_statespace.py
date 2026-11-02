"""transitleastsquares.gp: Kalman algebra vs. dense Gaussian algebra
(independent reference, analytic kernels).

Small problems (N ~ 300) with a gap and irregular sampling, for the rotation
GP and the smoothing spline: log likelihood, single events a = s'C^-1 y and
b = s'C^-1 s, and the periodic statistic including cross terms.
"""

import numpy as np
import pytest

from transitleastsquares.gp import events as ev
from transitleastsquares.gp import statespace as g


def k_sho(tau, sigma, period, Q):
    w0 = 2 * np.pi / period
    eta = np.sqrt(1 - 1 / (4 * Q**2))
    x = eta * w0 * np.abs(tau)
    return sigma**2 * np.exp(-w0 * np.abs(tau) / (2 * Q)) * (np.cos(x) + np.sin(x) / (2 * eta * Q))


def k_iwp(t, q):
    s = t - t[0]
    mn = np.minimum.outer(s, s)
    mx = np.maximum.outer(s, s)
    return g.DIFFUSE * (1 + np.outer(s, s)) + q * mn**2 * (3 * mx - mn) / 6


def dense_cov(model, t, yerr):
    C = np.diag(yerr**2 + model.jitter**2)
    tau = np.subtract.outer(t, t)
    for c in model.components:
        if isinstance(c, g.Offset):
            C += g.DIFFUSE
        elif isinstance(c, g.SHO):
            C += k_sho(tau, c.sigma, 2 * np.pi / c.w0, c.Q)
        elif isinstance(c, g.Spline):
            C += k_iwp(t, c.q)
    return C


def problem():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 12, 330))
    t = np.delete(t, np.where((t > 5) & (t < 6)))
    yerr = 0.4 * rng.uniform(0.8, 1.2, len(t))
    y = 8 * np.sin(2 * np.pi * t / 1.9) + 2 * np.sin(2 * np.pi * t / 7.3) + rng.normal(0, 0.4, len(t))
    y -= 8 * ((t - 1.234) % 4.1 < 0.1)
    return t, y, yerr


MODELS = {
    "rotation": lambda t: g.rotation_model(dict(sigma=8, period=1.9, Q0=2.0, dQ=5.0, f=0.4,
                                                sigma_slow=2.0, rho_slow=6.0, jitter=0.2)),
    "spline": lambda t: g.spline_model(0.4, 0.4, np.median(np.diff(t)), jitter=0.1),
}
# tolerances: the spline's diffuse prior makes the dense matrix ill-conditioned
TOL = {"rotation": dict(ll=1e-8, ab=1e-6, per=1e-3), "spline": dict(ll=1e-6, ab=5e-4, per=1e-3)}


@pytest.fixture(scope="module", params=list(MODELS))
def setup(request):
    t, y, yerr = problem()
    model = MODELS[request.param](t)
    C = dense_cov(model, t, yerr)
    A, Q, P0, R = model.matrices(t, yerr)
    K, S, _, _ = g.kalman_gains(A, Q, model.H, R, P0)
    e, _, _ = g.whiten(y, A, K, S, model.H)
    return request.param, t, y, yerr, model, C, (A, K, S, e)


def test_log_likelihood(setup):
    name, t, y, yerr, model, C, (A, K, S, e) = setup
    _, logdet = np.linalg.slogdet(C)
    ll_dense = -0.5 * (y @ np.linalg.solve(C, y) + logdet + len(t) * np.log(2 * np.pi))
    ll_kal = -0.5 * np.sum(np.log(2 * np.pi * S) + e**2)
    assert abs(ll_kal - ll_dense) < TOL[name]["ll"] * abs(ll_dense)


def test_single_events(setup):
    name, t, y, yerr, model, C, (A, K, S, e) = setup
    table = ev.template_table()
    texp = np.median(np.diff(t))
    durs = np.array([0.05, 0.2, 0.6])
    tc = np.array([0.0, 3.3, 5.5, 8.71, 11.98])  # edges and the gap included
    a, b = ev.single_event_sums(t, e, A, K, S, model.H, tc, durs, table, texp, 7)
    for m in range(len(tc)):
        for j in range(len(durs)):
            s = np.array([ev._template(x, tc[m], durs[j], table, texp, 7) for x in t])
            bd = s @ np.linalg.solve(C, s)
            if bd == 0:
                assert b[m, j] == 0
                continue
            ad = s @ np.linalg.solve(C, y)
            assert abs(a[m, j] - ad) / np.sqrt(bd) < TOL[name]["ab"]
            assert abs(b[m, j] - bd) / bd < TOL[name]["ab"]


def test_periodic_cross_terms(setup):
    """Summed single events vs. the dense statistic with all cross terms,
    for a period much longer than the filter memory."""
    name, t, y, yerr, model, C, (A, K, S, e) = setup
    table = ev.template_table()
    texp = np.median(np.diff(t))
    P, D = 4.1, 0.1
    tcs = 1.234 + P * np.arange(3)
    a, b = ev.single_event_sums(t, e, A, K, S, model.H, tcs, np.array([D]), table, texp, 7)
    s = sum(np.array([ev._template(x, c, D, table, texp, 7) for x in t]) for c in tcs)
    d2_dense = (s @ np.linalg.solve(C, y)) ** 2 / (s @ np.linalg.solve(C, s))
    assert abs(a.sum() ** 2 / b.sum() - d2_dense) / d2_dense < TOL[name]["per"]

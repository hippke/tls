"""transitleastsquares.gp.nelder_mead (trimmed copy) must equal scipy.optimize.minimize(method=
"Nelder-Mead") bit for bit: x, fun and the number of evaluations. SciPy is
a test-only dependency; without it these tests are skipped."""

import numpy as np
import pytest

from gp_cases import synthetic
from transitleastsquares.gp import statespace as g
from transitleastsquares.gp.fitting import ROT_KEYS, ROT_START_SHIFTS
from transitleastsquares.gp.nelder_mead import nelder_mead

scipy_optimize = pytest.importorskip("scipy.optimize")


def same(r_scipy, r_ours):
    return np.array_equal(r_scipy.x, r_ours.x) and r_scipy.fun == r_ours.fun and r_scipy.nfev == r_ours.nfev


@pytest.mark.parametrize("options", [
    {},
    {"maxiter": 50},
    {"maxfev": 77},
    {"initial_simplex": np.vstack([np.zeros(4)] + [0.5 * np.eye(4)[i] for i in range(4)])},
    {"xatol": 1e-8, "fatol": 1e-8},
])
def test_rosenbrock(options):
    x0 = np.array([1.3, 0.7, 0.8, 1.9])
    r1 = scipy_optimize.minimize(scipy_optimize.rosen, x0, method="Nelder-Mead", options=options)
    r2 = nelder_mead(scipy_optimize.rosen, x0, **options)
    assert same(r1, r2)


def _rotation_objective():
    t, f, e = synthetic()
    y, yerr = (f - 1) * 1e3, e * 1e3  # internal units (ppt), as in the fits

    def nll(lx):
        with np.errstate(over="ignore"):
            p = dict(zip(ROT_KEYS, np.exp(lx)))
        if not (0.69 < p["period"] < 6.9 and 0.5 < p["rho_slow"] < 100):
            return 1e25
        v = -g.log_likelihood(g.rotation_model(p), t, y, yerr)
        return v if np.isfinite(v) else 1e25

    base = np.log([np.std(y), 2.3, 1.0, 1.0, 0.5, 0.3 * np.std(y), 5.0, np.median(yerr) * 0.5])
    return nll, base


@pytest.mark.parametrize("start", [0, 1])
def test_rotation_objective(start):
    """The real fit objective (multimodal): same path, same optimum."""
    nll, base = _rotation_objective()
    x0 = base + ROT_START_SHIFTS[start]
    opts = dict(maxiter=4000, xatol=1e-4, fatol=1e-4)
    assert same(scipy_optimize.minimize(nll, x0, method="Nelder-Mead", options=opts), nelder_mead(nll, x0, **opts))


def test_spline_objective():
    t, f, e = synthetic()
    y, yerr = (f - 1) * 1e3, e * 1e3
    sig, cad = np.median(yerr), np.median(np.diff(t))

    def nll(lx):
        h, jit = np.exp(lx)
        if not 0.05 <= h <= 5.0:
            return 1e25
        return -g.log_likelihood(g.spline_model(h, sig, cad, jit), t, y, yerr)

    opts = dict(maxiter=2000, xatol=1e-5, fatol=1e-5)
    x0 = np.log([0.3, sig])
    assert same(scipy_optimize.minimize(nll, x0, method="Nelder-Mead", options=opts), nelder_mead(nll, x0, **opts))

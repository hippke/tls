"""transitleastsquares.gp: the gradient fit (fit_method="gradient").

The adjoint log likelihood must equal the sequential filter with the step
table bit for bit (also with checkpoints), and its gradient central finite
differences; the bounded quasi-Newton must find known optima inside and on
the bounds. On K2-25 (the points that the default fit keeps after flare
clipping) the gradient fit must reach the reference likelihood of the
default fit with clipping (test_gp_fit.REFERENCE, -4595.4897 in ppt units)
and beat the default's single cold start, which ends far from the optimum.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from gp_cases import k2_25, synthetic
from transitleastsquares import gp
from transitleastsquares.gp import gradient
from transitleastsquares.gp.baselines import fit_with_flare_clipping
from transitleastsquares.gp.quasi_newton import minimize_bounded
from transitleastsquares.gp.statespace import kalman_loglik_table, rotation_model, step_transitions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# log hyperparameters (ppt): the synthetic data's own model, and one far from it
LX = {"true": np.log([5.0, 2.3, 3.0, 20.0, 0.6, 2.0, 8.0, 0.3]),
      "off": np.log([2.0, 3.1, 0.2, 400.0, 30.0, 0.5, 1.5, 0.05])}


def _data():
    t, f, e = synthetic()
    return t, (f - 1) * 1e3, e * 1e3


def _table_loglik(lik, lx):
    m = rotation_model(dict(zip(gradient.ROT_KEYS, np.exp(lx))))
    kinds, starts, pars, P0 = m.blocks()
    At, Qt = step_transitions(lik.steps, kinds, pars)
    return kalman_loglik_table(lik.idx, At, Qt, lik.y, lik.e2 + m.jitter**2, kinds, starts, P0, m.n_diffuse)


@pytest.mark.parametrize("point", list(LX))
def test_loglik_and_finite_differences(point):
    t, y, e = _data()
    lik = gradient.RotationGradient(t, y, e)
    lx = LX[point]
    ll, g = lik(lx)
    assert ll == _table_loglik(lik, lx)
    h = 1e-5
    fd = np.empty(len(lx))
    for j in range(len(lx)):
        a, b = lx.copy(), lx.copy()
        a[j] += h
        b[j] -= h
        fd[j] = (_table_loglik(lik, a) - _table_loglik(lik, b)) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())


def test_checkpoints(monkeypatch):
    """Segments with checkpoints: the same log likelihood bit for bit, the same gradient up to rounding."""
    t, y, e = _data()
    ll, g = gradient.RotationGradient(t, y, e)(LX["off"])
    monkeypatch.setattr(gradient, "STORE_MAX", 0)
    monkeypatch.setattr(gradient, "SEGMENT", 97)
    lik = gradient.RotationGradient(t, y, e)
    assert lik.seg == 97 and len(y) > 5 * 97
    ll2, g2 = lik(LX["off"])
    assert ll2 == ll
    np.testing.assert_allclose(g2, g, rtol=1e-10, atol=1e-10 * np.abs(g).max())


def test_minimize_bounded():
    def quad(x):  # minimum at (3, -1, 0.5); the box holds x0 <= 2 and x1 >= -0.5
        c, w = np.array([3.0, -1.0, 0.5]), np.array([1.0, 10.0, 0.1])
        return np.sum(w * (x - c) ** 2), 2 * w * (x - c)

    r = minimize_bounded(quad, np.zeros(3), [-5, -0.5, -5], [2, 5, 5], gtol=1e-10, ftol=0)
    np.testing.assert_allclose(r.x, [2.0, -0.5, 0.5], atol=1e-8)

    def rosen(x):  # infinite beyond x0 = 1.5: the line search must step back
        if x[0] > 1.5:
            return np.inf, np.zeros(2)
        f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        return f, np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])

    r = minimize_bounded(rosen, [-1.2, 1.0], [-2, -2], [2, 3], gtol=1e-8, ftol=0)
    np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-5)
    assert r.nfev < 200


def test_fit_k2_25():
    t, f, e = k2_25()
    _, _, use = fit_with_flare_clipping("rotation", t, f, e, period_guess=1.876)
    to_ppt = (use.sum() - 1) * np.log(1e3)
    _, ll_nm = gp.RotationBaseline.fit(t, f, e, period_guess=1.876, use=use)
    b1, ll1 = gp.RotationBaseline.fit(t, f, e, period_guess=1.876, use=use, method="gradient")
    b2, ll2 = gp.RotationBaseline.fit(t, f, e, period_guess=1.876, use=use, method="gradient", n_starts=2)
    assert ll1 - to_ppt >= -4595.4897 - 0.05
    assert ll1 > ll_nm + 10  # the default single start ends far from the optimum here
    assert ll2 >= ll1
    assert abs(b2.period - 1.878) / 1.878 < 0.005
    np.testing.assert_allclose(b2.log_likelihood(t[use], f[use], e[use]), ll2, rtol=1e-12)


def test_fit_starts_and_threads():
    """Several gradient starts in threads = the same starts in one thread, never worse than one start."""
    t, f, e = synthetic()
    b1, ll1 = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, method="gradient")
    b_seq, ll_seq = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, method="gradient", n_starts=3, use_threads=1)
    b_par, ll_par = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, method="gradient", n_starts=3, use_threads=3)
    assert b_par == b_seq and ll_par == ll_seq and ll_par >= ll1
    with pytest.raises(ValueError):
        gp.RotationBaseline.fit(t, f, e, period_guess=2.3, method="gradient", n_starts=4)
    with pytest.raises(ValueError):
        gp.RotationBaseline.fit(t, f, e, period_guess=2.3, method="bfgs")


def test_search_fit_method():
    t, f, e = synthetic(days=12.0)
    r = gp.search(t, f, e, rotation_period=2.3, fit_method="gradient", period_min=1.0, period_max=4.0)
    assert abs(r.period - 3.1) < 0.02
    with pytest.raises(ValueError):
        gp.search(t, f, e, baseline="spline", fit_method="gradient", period_min=1.0, period_max=4.0)


SCRIPT = r"""
import sys
class Block:
    def find_spec(self, name, path, target=None):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked")
sys.meta_path.insert(0, Block())
from transitleastsquares import gp
from gp_cases import synthetic
t, f, e = synthetic(days=12.0)
r = gp.search(t, f, e, rotation_period=2.3, fit_method="gradient", period_min=1.0, period_max=4.0)
assert not any(m == "scipy" or m.startswith("scipy.") for m in sys.modules)
print("OK", round(r["period"], 4))
"""


@pytest.mark.slow
def test_gradient_fit_without_scipy(tmp_path):
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba"),
               PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, "tests"), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", SCRIPT], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.startswith("OK")

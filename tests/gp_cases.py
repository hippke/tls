"""Fixed problems for the tests of transitleastsquares.gp (golden results,
fits, interface). The searches use fixed baselines, so they test the search
and the Kalman algebra without the optimizer. The synthetic data have gaps,
dropped points, cadence jitter and heteroscedastic errors.

Regenerate the golden files (only for an intended, understood change):
python tests/gp_cases.py --write
"""

import os
import sys

import numpy as np

from transitleastsquares import gp
from transitleastsquares.gp.events import _template, template_table
from transitleastsquares.gp.statespace import rotation_model

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden", "gp")

# K2-25 hyperparameters (maximum likelihood, rounded); amplitudes in relative flux
ROT_K2 = gp.RotationBaseline(sigma=4.6493e-3, period=1.87868, Q0=7.446, dQ=612.08, f=1.3769,
                             sigma_slow=35.383e-3, rho_slow=100.0, jitter=0.80118e-3)
SPLINE_K2 = gp.SplineBaseline(h=0.05, jitter=0.94908e-3, sigma_ref=0.37433e-3, cadence=0.0204317)
ROT_SYN = gp.RotationBaseline(sigma=5.0e-3, period=2.3, Q0=3.0, dQ=20.0, f=0.6,
                              sigma_slow=2.0e-3, rho_slow=8.0, jitter=0.3e-3)
SPLINE_SYN = gp.SplineBaseline(h=0.2, jitter=0.2e-3, sigma_ref=0.45e-3, cadence=0.0204)


def k2_25(flares=False):
    """K2-25 (EVEREST, K2 C4); without the 70 flare points by default."""
    t, f, e, flare = np.loadtxt(os.path.join(HERE, "data", "k2_25_everest_c4.csv"), delimiter=",", unpack=True)
    keep = np.ones(len(t), bool) if flares else flare == 0
    return t[keep], f[keep], e[keep]


def synthetic(days=30.0, seed=11, cadence=0.0204):
    """Rotation-like variability drawn with the state-space model itself,
    gaps, dropped points, cadence jitter, varying errors, and a transit
    (P = 3.1 d, duration 0.11 d, depth 2 ppt). Default cadence: K2 (29.4 min)."""
    rng = np.random.default_rng(seed)
    cad = cadence
    t = np.arange(0, days, cad) + rng.uniform(-0.02, 0.02, int(np.ceil(days / cad))) * cad
    keep = rng.uniform(size=len(t)) > 0.05  # 5 % dropped
    keep &= ~((t > 0.3 * days) & (t < 0.3 * days + 1.0))  # 1 d gap
    keep &= ~((t > 0.7 * days) & (t < 0.7 * days + 0.3))  # 0.3 d gap
    t = np.sort(t[keep])
    yerr = 0.45 * rng.uniform(0.7, 1.4, len(t))  # ppt
    model = rotation_model(dict(sigma=5.0, period=2.3, Q0=3.0, dQ=20.0, f=0.6, sigma_slow=2.0,
                                rho_slow=8.0, jitter=0.3))
    A, Q, P0, R = model.matrices(t, yerr)
    x = np.zeros(model.dim)
    y = np.empty(len(t))
    for i in range(len(t)):
        if i == 0:
            x = np.linalg.cholesky(P0 + 1e-12 * np.eye(model.dim)) @ rng.normal(size=model.dim)
            x[0] = 0.0  # offset state (diffuse in the model): start at 0
        else:
            x = A[i] @ x + np.linalg.cholesky(Q[i] + 1e-12 * np.eye(model.dim)) @ rng.normal(size=model.dim)
        y[i] = model.H @ x + rng.normal(0, np.sqrt(R[i]))
    table = template_table()
    s = np.zeros(len(t))
    for c in 0.77 + 3.1 * np.arange(int(days / 3.1) + 1):
        s += np.array([_template(x_, c, 0.11, table, cad, 7) for x_ in t])
    return t, 1 + y / 1e3 + 0.002 * s, yerr / 1e3


CASES = {
    "k2_rot": (k2_25, ROT_K2),
    "k2_spline": (k2_25, SPLINE_K2),
    "syn_rot": (synthetic, ROT_SYN),
    "syn_spline": (synthetic, SPLINE_SYN),
}
SEARCH = dict(period_min=1.0, period_max=8.0)

# golden file key -> GPSearchResults key
ARRAY_KEYS = {"periods": "periods", "dchi2": "delta_chi2", "power": "power", "log_bf_per_period": "log_bf"}
SCALAR_KEYS = {k: k for k in ("period", "T0", "duration", "depth", "depth_err", "snr", "SDE", "chi2_0",
                              "log_bf_global")}


def run(name, **extra):
    loader, baseline = CASES[name]
    t, f, e = loader()
    extra.setdefault("bayes_factor", True)
    return gp.search(t, f, e, baseline=baseline, **SEARCH, **extra)


def loglik_ppt(name):
    """Log likelihood in the ppt units of the golden files."""
    loader, baseline = CASES[name]
    t, f, e = loader()
    return baseline.log_likelihood(t, f, e) - (len(t) - baseline._model(t, e).n_diffuse) * np.log(1e3)


def write():
    for name in CASES:
        for suffix, kw in (("", {}), ("_cad4", dict(t0_step_duration=None))):
            r = run(name, **kw)
            out = {gk: np.asarray(r[rk]) for gk, rk in {**ARRAY_KEYS, **SCALAR_KEYS}.items()}
            out["loglik"] = np.asarray(loglik_ppt(name))
            np.savez(os.path.join(GOLDEN, f"{name}{suffix}.npz"), **out)
            print(name + suffix, {k: round(float(r[k]), 6) for k in ("period", "SDE", "depth")})


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()

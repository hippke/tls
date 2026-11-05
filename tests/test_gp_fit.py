"""transitleastsquares.gp: baseline fits.

The fits with flare clipping on K2-25 must reach at least the reference
likelihood (logL in ppt units -4595.4897 for the rotation GP with the
default single Nelder-Mead pass at tolerance 1e-3; -6457.6451 for the
spline) with the same flares clipped: optimizer changes must not cost fit
quality without a recovery check (the rotation default trades the optimum
for 2x speed; 400 K2 injections found the same planets). Parallel fit starts
must equal sequential ones, and the search result must not depend on the
number of threads.
"""

import numpy as np
import pytest

from gp_cases import k2_25, run, synthetic
from transitleastsquares import gp
from transitleastsquares.gp.baselines import fit_with_flare_clipping

# kind -> (logL in ppt units, flares clipped); the flux-unit likelihood adds
# (n - n_diffuse) ln 1000 with n_diffuse = 1 (rotation) and 3 (spline)
REFERENCE = {"rotation": (-4595.4897, 24, 1), "spline": (-6457.6451, 12, 3)}


@pytest.mark.parametrize("kind", list(REFERENCE))
def test_fit_quality(kind):
    t, f, e = k2_25()  # the 70 flares of the file removed; clipping finds 24 more
    kw = dict(period_guess=1.876) if kind == "rotation" else {}
    b, ll, use = fit_with_flare_clipping(kind, t, f, e, **kw)
    ll_ref, nclip_ref, nd = REFERENCE[kind]
    assert int((~use).sum()) == nclip_ref
    ll_ppt = ll - (use.sum() - nd) * np.log(1e3)
    assert ll_ppt >= ll_ref - 0.05, f"{kind}: logL {ll_ppt:.4f} < reference {ll_ref:.4f}"
    np.testing.assert_allclose(b.log_likelihood(t[use], f[use], e[use]), ll, rtol=1e-12)
    if kind == "rotation":
        assert abs(b.period - 1.879) / 1.879 < 0.005


def test_parallel_starts_deterministic():
    """4 starts in 4 threads (GIL-free likelihood) = the same 4 starts in one
    thread, bit for bit, and never worse than the single start."""
    t, f, e = synthetic()
    b1, ll1 = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, n_starts=1)
    b_seq, ll_seq = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, n_starts=4, use_threads=1)
    b_par, ll_par = gp.RotationBaseline.fit(t, f, e, period_guess=2.3, n_starts=4, use_threads=4)
    assert b_par == b_seq and ll_par == ll_seq
    assert ll_par >= ll1


def test_search_threads():
    """use_threads changes the speed, not the result, and is restored."""
    import numba

    before = numba.get_num_threads()
    r1 = run("syn_rot", bayes_factor=False, use_threads=1)
    assert numba.get_num_threads() == before
    r4 = run("syn_rot", bayes_factor=False, use_threads=4)
    np.testing.assert_array_equal(r1.delta_chi2, r4.delta_chi2)
    assert r1.period == r4.period and r1.T0 == r4.T0


def test_warm_start_parameter_at_zero(monkeypatch):
    """A parameter that underflowed to 0 in the previous clipping round: the next round must still converge."""
    from transitleastsquares.gp import fitting

    t, f, e = synthetic()
    b, _, use = fit_with_flare_clipping("rotation", t, f, e, use_threads=1)
    nit = []
    nm = fitting.nelder_mead

    def count(*a, **k):
        r = nm(*a, **k)
        nit.append(r.nit)
        return r

    monkeypatch.setattr(fitting, "nelder_mead", count)
    start = dict(b.as_dict(), f=0.0, sigma_slow=0.0)
    p, ll = fitting.fit_rotation(t, (f - 1) * 1e3, e * 1e3, period_guess=b.as_dict()["period"], start=start, use_threads=1)
    assert np.isfinite(ll) and all(np.isfinite(v) for v in p.values())
    assert nit[0] < 4000


def _with_flares(cadence):
    """Rotation signal, white noise and 4 flares of 6 sigma per point at 2 min cadence."""
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 20.0, cadence)
    f = 1 + rng.normal(0, 1e-3, len(t)) + 3e-3 * np.sin(2 * np.pi * t / 3.7)
    fl = np.zeros(len(t))
    for t0 in (3.1, 8.4, 12.2, 17.6):
        m = (t > t0) & (t < t0 + 0.1)
        fl[m] += 6e-3 * np.exp(-(t[m] - t0) / 0.02)
    return t, f + fl, np.full(len(t), 1e-3), fl


def test_binned_flare_clipping():
    """At a short cadence the default clips the faint part of a flare too; single points miss most of it."""
    t, f, e, fl = _with_flares(2 / 1440)
    faint = fl > 1e-3
    _, _, use0 = fit_with_flare_clipping("rotation", t, f, e, flare_bin=None, use_threads=1)
    _, _, use = fit_with_flare_clipping("rotation", t, f, e, use_threads=1)
    assert (~use[faint]).mean() == 1 and (~use0[faint]).mean() < 0.7
    assert (~use & (fl == 0)).mean() < 0.02


def test_binned_flare_clipping_long_cadence():
    """It leaves a cadence above half the bin width alone (K2: REFERENCE)."""
    t, f, e, _ = _with_flares(0.0204)
    b0, ll0, use0 = fit_with_flare_clipping("rotation", t, f, e, flare_bin=None, use_threads=1)
    b, ll, use = fit_with_flare_clipping("rotation", t, f, e, use_threads=1)
    assert b == b0 and ll == ll0 and np.array_equal(use, use0)


def test_search_flare_bin_width():
    """gp.search uses the binned clipping by default; flare_bin_width=None turns it off."""
    t, f, e, fl = _with_flares(2 / 1440)
    kw = dict(period_min=2.0, period_max=8.0, trend=False, use_threads=1)
    r, r0 = gp.search(t, f, e, **kw), gp.search(t, f, e, flare_bin_width=None, **kw)
    assert (~r.use).sum() > (~r0.use).sum()
    assert (~r.use[fl > 1e-3]).mean() == 1

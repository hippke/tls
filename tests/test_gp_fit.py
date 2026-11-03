"""transitleastsquares.gp: baseline fits.

The fits with flare clipping on K2-25 must reach at least the reference
likelihood (best known, logL in ppt units -4597.2701 for the rotation GP,
reached by 1 start as by 4; -6457.6451 for the spline) with the same flares
clipped: optimizer changes must not cost fit quality. Parallel fit starts
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
REFERENCE = {"rotation": (-4597.2701, 19, 1), "spline": (-6457.6451, 12, 3)}


@pytest.mark.parametrize("kind", list(REFERENCE))
def test_fit_quality(kind):
    t, f, e = k2_25()  # the 70 flares of the file removed; clipping finds 19 more
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

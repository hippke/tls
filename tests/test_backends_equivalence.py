"""All registered backends must give the reference results (backend "numba")."""

import warnings

import numpy
import pytest
from golden_cases import CASES

from transitleastsquares import available_backends, get_backend, transitleastsquares

NAMES = [
    "k2_3_window",
    "syn_earth_dy",
    "narrow_stellar",
    "centered_time",
    "no_fit",
    "short_lc",
    "k2_box",
]


def run(name, backend):
    loader, kw = CASES[name]
    t, y, dy = loader()
    kwargs = dict(show_progress_bar=False, verbose=False, use_threads=2)
    kwargs.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transitleastsquares(t, y, dy, verbose=False).power(
            backend=backend, **kwargs
        )


EXACT = [b for b in available_backends() if b != "numba" and get_backend(b).exact]
APPROX = [b for b in available_backends() if not get_backend(b).exact]


@pytest.mark.parametrize("backend", EXACT)
@pytest.mark.parametrize("name", NAMES)
def test_backend_equivalence(backend, name):
    ref = run(name, "numba")
    cur = run(name, backend)
    numpy.testing.assert_allclose(cur.chi2, ref.chi2, rtol=1e-10)
    numpy.testing.assert_allclose(cur.power, ref.power, rtol=0, atol=1e-6)
    for key in ["period", "T0", "duration", "depth", "SDE", "snr"]:
        a, b = numpy.asarray(cur[key], float), numpy.asarray(ref[key], float)
        numpy.testing.assert_allclose(
            a, b, rtol=1e-7, atol=1e-12, equal_nan=True, err_msg=key
        )


@pytest.mark.parametrize("backend", APPROX)
@pytest.mark.parametrize("name", NAMES)
def test_approximate_backend_close(backend, name):
    """Approximate backends: same detection (P, T0, duration), small changes of
    the statistic. Sensitivity is validated separately (injection-recovery)."""
    ref = run(name, "fused")
    cur = run(name, backend)
    numpy.testing.assert_allclose(cur.chi2, ref.chi2, rtol=5e-3)
    for key in ["period", "T0", "duration"]:
        a, b = numpy.asarray(cur[key], float), numpy.asarray(ref[key], float)
        numpy.testing.assert_allclose(a, b, rtol=1e-9, equal_nan=True, err_msg=key)
    if numpy.isfinite(ref.SDE) and ref.SDE > 0:
        assert abs(cur.SDE - ref.SDE) < 0.02 * ref.SDE + 0.05


def test_ls_depth_option_runs_and_never_fits_worse():
    """ls_depth (experimental, specialized away unless enabled): the least-
    squares depth maximizes the gain per shift, so with the exact search its
    chi2 is never higher than with TLS's box depth."""
    import numpy
    from test_core_oracle import cache

    from transitleastsquares import tls_constants as C
    from transitleastsquares.backends import SearchProblem
    from transitleastsquares.core_fused import FusedProblem

    rng = numpy.random.default_rng(11)
    t = numpy.arange(3000) * 0.01
    y = 1 + rng.normal(0, 1e-3, len(t))
    y[(t % 3.3) < 0.08] -= 2e-3
    dy = numpy.ones(len(t))
    ov, lc = cache(t, y, "default")
    problem = SearchProblem(
        t, y, dy, lc, ov, 1e-5,
        C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX, 0.01,
    )
    box, ls = FusedProblem(problem), FusedProblem(problem)
    ls.set_pl(0, ls_depth=True)
    for p in (1.1, 2.0, 3.3, 7.7):
        assert ls.search(p)[1] <= box.search(p)[1] * (1 + 1e-12)

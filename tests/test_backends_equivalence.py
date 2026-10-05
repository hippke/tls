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

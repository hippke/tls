"""All registered backends must give the reference results (backend "numba")."""

import warnings

import numpy
import pytest
from golden_cases import CASES

from transitleastsquares import available_backends, transitleastsquares

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


@pytest.mark.parametrize("backend", [b for b in available_backends() if b != "numba"])
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

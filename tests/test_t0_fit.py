"""The fast final T0 fit (rotation of one phase sort) must reproduce the
reference implementation (re-sort for every trial T0, TLS <= 1.33)."""

import numpy
import pytest

from transitleastsquares import tls_constants
from transitleastsquares.stats import final_T0_fit, final_T0_fit_sorting


def make(seed, n=3000, span=40.0, period=4.37, hetero=True, gaps=True):
    rng = numpy.random.default_rng(seed)
    t = numpy.sort(rng.uniform(0, span, n)) if gaps else numpy.linspace(0, span, n)
    t0 = rng.uniform(0, period)
    dur = 0.12
    ph = numpy.abs((t - t0 + 0.5 * period) % period - 0.5 * period)
    y = 1 + rng.normal(0, 3e-4, n)
    y[ph < dur / 2] -= 1.2e-3
    dy = rng.uniform(0.5, 2.0, n) if hetero else numpy.ones(n)
    width = int(round(n * dur / period))
    x = numpy.linspace(-1, 1, width)
    signal = 1 - tls_constants.SIGNAL_DEPTH * numpy.sqrt(1 - x**2)  # TLS-like template
    return t, y, dy, signal, period


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("margin", [0.0, 0.01, 0.05])
@pytest.mark.parametrize("hetero", [False, True])
def test_rotation_equals_sorting(seed, margin, hetero):
    t, y, dy, signal, period = make(seed, hetero=hetero, gaps=seed % 2 == 0)
    kw = dict(
        signal=signal,
        depth=1 - 1.1e-3,
        t=t,
        y=y,
        dy=dy,
        period=period,
        T0_fit_margin=margin,
        show_progress_bar=False,
        verbose=False,
    )
    assert final_T0_fit(**kw) == final_T0_fit_sorting(**kw)

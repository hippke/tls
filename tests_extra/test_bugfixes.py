"""Regression tests for the bugs fixed on branch `baseline-fixes`.
Most of these tests fail on upstream TLS 1.33.
"""

import contextlib
import io
import os
import subprocess
import sys
import warnings

import numpy
import pytest

import transitleastsquares.tls_constants as tls_constants
from transitleastsquares import cleaned_array, duration_grid, period_grid
from transitleastsquares import transitleastsquares as TLS
from transitleastsquares.stats import final_T0_fit, period_uncertainty
from transitleastsquares.transit import reference_transit

warnings.simplefilter("ignore")
QUIET = dict(show_progress_bar=False, verbose=False)


def box_lc(t, period, t0, duration, depth):
    y = numpy.ones_like(t)
    phase = (t - t0 + 0.5 * period) % period - 0.5 * period
    y[numpy.abs(phase) < duration / 2] -= depth
    return y

def test_final_T0_fit_uses_uncertainties():
    """final_T0_fit overwrote dy with the (rolled) flux -> unweighted fit.
    A deep fake dip with huge uncertainties must not attract T0."""
    t = numpy.arange(0, 50, 0.01)
    period, t0_true, dur = 5.0, 2.0, 0.1
    y = box_lc(t, period, t0_true, dur, 1e-3)
    fake = numpy.abs((t - 3.5 + 0.5 * period) % period - 0.5 * period) < dur / 2
    y[fake] -= 3e-3
    dy = numpy.ones_like(t)
    dy[fake] = 100.0
    n_in = int(round(len(t) * dur / period))
    signal = numpy.full(n_in, 1 - tls_constants.SIGNAL_DEPTH)
    T0 = final_T0_fit(
        signal=signal,
        depth=1 - 1e-3,
        t=t,
        y=y,
        dy=dy,
        period=period,
        T0_fit_margin=0.01,
        show_progress_bar=False,
        verbose=False,
    )
    assert abs(((T0 - t0_true + 2.5) % 5) - 2.5) < 0.02, T0


def test_final_T0_fit_does_not_modify_inputs():
    t = numpy.arange(0, 20, 0.02)
    y = box_lc(t, 4.0, 1.0, 0.2, 1e-3)
    dy = numpy.linspace(1, 2, len(t))
    dy0 = dy.copy()
    final_T0_fit(numpy.full(20, 0.5), 0.999, t, y, dy, 4.0, 0.01, False, False)
    numpy.testing.assert_array_equal(dy, dy0)

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

def test_snr_uses_out_of_transit_noise():
    rng = numpy.random.default_rng(4)
    t = numpy.arange(0, 90, 0.02)
    sigma, depth = 1e-4, 1e-3
    y = box_lc(t, 7.3, 2.0, 0.25, depth) + rng.normal(0, sigma, len(t))
    r = TLS(t, y).power(period_min=7, period_max=7.6, use_threads=1, **QUIET)
    n_in = numpy.sum(r.per_transit_count)
    snr_expected = depth / sigma * numpy.sqrt(n_in)
    assert abs(r.snr / snr_expected - 1) < 0.15, (r.snr, snr_expected)

def test_no_fit_is_detected():
    """With transit_depth_min above any signal, TLS must report no detection
    (1.33 compared max(chi2)==min(chi2) exactly, which fails by rounding)."""
    rng = numpy.random.default_rng(3)
    t = numpy.linspace(0, 60, 6000)
    y = 1 + rng.normal(0, 1e-5, len(t))
    r = TLS(t, y).power(transit_depth_min=1e-2, use_threads=1, **QUIET)
    assert r.SDE == 0 and numpy.isnan(r.period)

def test_cleaned_array_keeps_nonpositive_times():
    t = numpy.linspace(-5, 5, 11)
    y = numpy.ones(11)
    ct, cy = cleaned_array(t, y)
    assert len(ct) == 11
    ct, cy, cdy = cleaned_array(t, y, numpy.ones(11))
    assert len(ct) == 11
    t[3] = numpy.nan
    ct, cy = cleaned_array(t, y)
    assert len(ct) == 10

def test_user_template_parameters_not_overwritten():
    t = numpy.linspace(0, 30, 3000)
    y = numpy.ones_like(t)
    m = TLS(t, y, verbose=False)
    from transitleastsquares.validate import validate_args

    validate_args(m, dict(rp=0.1, a=10.0, per=5.0, inc=88.0))
    assert (m.rp, m.a, m.per, m.inc) == (0.1, 10.0, 5.0, 88.0)
    m = TLS(t, y, verbose=False)
    validate_args(m, dict(a=10.0, b=0.5))
    numpy.testing.assert_almost_equal(m.inc, numpy.degrees(numpy.arccos(0.5 / 10.0)))
    m = TLS(t, y, verbose=False)
    validate_args(m, {})  # defaults unchanged
    assert (m.rp, m.a, m.per, m.inc) == (
        tls_constants.DEFAULT_RP,
        tls_constants.DEFAULT_A,
        tls_constants.DEFAULT_PERIOD,
        tls_constants.DEFAULT_INC,
    )

def test_duration_grid_honours_stellar_limits():
    # long periods only, so that the 0.12 cap (FRACTIONAL_TRANSIT_DURATION_MAX)
    # does not hide the difference
    periods = period_grid(R_star=1, M_star=1, time_span=200, period_min=20)
    d_default = duration_grid(periods, shortest=1e-4, log_step=1.1)
    d_narrow = duration_grid(
        periods,
        shortest=1e-4,
        log_step=1.1,
        R_star_min=0.8,
        R_star_max=1.2,
        M_star_min=0.8,
        M_star_max=1.2,
    )
    assert max(d_narrow) < max(d_default)
    assert min(d_narrow) > min(d_default)
    # defaults unchanged vs. the repo test (test_duration_grid.py)
    d = duration_grid(period_grid(1, 1, 20, 0, 999, 3), log_step=1.05, shortest=2)
    assert len(d) == 69

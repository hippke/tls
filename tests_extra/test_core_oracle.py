"""Brute-force oracle for the TLS core (core.search_period).

The optimised numba kernels (running mean, sliding out-of-transit residuals,
edge-effect correction, template scaling) are re-implemented here in the most
literal way: for every trial duration and every phase shift, build the full
model of the phase-folded light curve and compute chi2 directly.

This is the correctness reference for upcoming performance work: any
re-implementation of search_period must agree with it (to float tolerance).
"""

import warnings

import numpy
import pytest

import transitleastsquares.tls_constants as C
from transitleastsquares import duration_grid, period_grid
from transitleastsquares.core import search_period
from transitleastsquares.grid import T14
from transitleastsquares.transit import get_cache

warnings.simplefilter("ignore")


def brute_force_period(
    period, t, y, dy, depth_min, lc_arr, ov, R_min, R_max, M_min, M_max, T0_fit_margin
):
    N = len(t)
    phases = (t / period) % 1.0
    s = numpy.argsort(phases, kind="mergesort")
    flux, w = y[s], 1.0 / dy[s] ** 2
    widths = numpy.unique(ov["width_in_samples"])
    dmax = T14(R_s=R_max, M_s=M_max, P=period, small=False)
    dmin = T14(R_s=R_min, M_s=M_min, P=period, small=True)
    length = t.max() - t.min()
    corr = (length / period + 1) / (length / period)
    widths = widths[
        (widths >= int(numpy.floor(dmin * N)))
        & (widths <= int(numpy.ceil(dmax * N * corr)))
    ]
    chi2_const = numpy.sum((flux - 1) ** 2 * w)
    best = (chi2_const, 0, 0.0)
    for width in widths:
        row = int(numpy.argmax(ov["width_in_samples"] == width))
        signal = lc_arr[row]
        xth = 1
        if T0_fit_margin > 0 and width > T0_fit_margin:
            xth = max(1, int(width * T0_fit_margin))
        maxw = int(max(numpy.unique(ov["width_in_samples"])))
        maxw += maxw % 2
        n_shifts = N + maxw - width + 1
        for i in range(n_shifts):
            if i % xth:
                continue
            idx = numpy.arange(i, i + width) % N  # wrap-around in phase
            depth = 1 - numpy.mean(flux[idx])
            if not depth > depth_min:
                continue
            target = depth * ov["overshoot"][row]
            model = numpy.ones(N)
            model[idx[: len(signal)]] = 1 - (1 - signal) * target / C.SIGNAL_DEPTH
            # points inside the window but beyond the template are not part of
            # the TLS statistic (template is never shorter than width, checked below)
            chi2 = numpy.sum((flux - model) ** 2 * w)
            if chi2 < best[0]:
                best = (chi2, row, 1 - target)
    return best


def make_case(seed, n=700, span=12.0, with_dy=True):
    rng = numpy.random.default_rng(seed)
    t = numpy.sort(rng.uniform(0, span, n))
    y = 1 + rng.normal(0, 3e-4, n)
    y[numpy.abs(((t - 0.7) % 2.3) - 1.15) > 1.15 - 0.06] -= 1.5e-3
    dy = rng.uniform(2e-4, 5e-4, n) if with_dy else numpy.full(n, numpy.std(y))
    return t, y, dy / numpy.mean(dy)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("T0_fit_margin", [0.0, 0.05])
def test_search_period_matches_brute_force(seed, T0_fit_margin):
    t, y, dy = make_case(seed)
    periods = period_grid(1, 1, t.max() - t.min())
    durations = duration_grid(periods, shortest=1 / len(t), log_step=1.1)
    maxw = int(max(durations) * len(y))
    maxw += maxw % 2
    ov, lc_arr = get_cache(
        durations,
        maxw,
        C.DEFAULT_PERIOD,
        C.DEFAULT_RP,
        C.DEFAULT_A,
        C.DEFAULT_INC,
        0,
        90,
        C.DEFAULT_U,
        "quadratic",
        verbose=False,
    )
    for row in range(len(ov)):
        assert len(lc_arr[row]) == ov["width_in_samples"][row]
    lim = (C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX)
    for period in [2.3, 1.7, 4.11, 0.93]:
        p, chi2, row, depth = search_period(
            period, t, y, dy, 1e-5, *lim, lc_arr, ov, T0_fit_margin
        )
        b_chi2, b_row, b_depth = brute_force_period(
            period, t, y, dy, 1e-5, lc_arr, ov, *lim, T0_fit_margin
        )
        numpy.testing.assert_allclose(chi2, b_chi2, rtol=1e-9)
        assert row == b_row
        numpy.testing.assert_allclose(depth, b_depth, rtol=1e-9)

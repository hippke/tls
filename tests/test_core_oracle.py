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
            # window points beyond the (trimmed) template are out of transit
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


TEMPLATES = {
    "default": (
        C.DEFAULT_PERIOD,
        C.DEFAULT_RP,
        C.DEFAULT_A,
        C.DEFAULT_INC,
        0,
        90,
        C.DEFAULT_U,
        "quadratic",
    ),
    # eccentric square-root law: trimmed templates, 1 sample shorter than the window
    "ecc_sqrt": (
        5.0,
        0.03,
        15.0,
        numpy.degrees(numpy.arccos(0.4 / 15.0)),
        0.3,
        40,
        [0.3, 0.3],
        "squareroot",
    ),
}


def cache(t, y, template):
    periods = period_grid(1, 1, t.max() - t.min())
    durations = duration_grid(periods, shortest=1 / len(t), log_step=1.1)
    maxw = int(max(durations) * len(y))
    maxw += maxw % 2
    return get_cache(durations, maxw, *TEMPLATES[template], verbose=False)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("T0_fit_margin", [0.0, 0.05])
@pytest.mark.parametrize("template", list(TEMPLATES))
def test_search_period_matches_brute_force(seed, T0_fit_margin, template):
    t, y, dy = make_case(seed)
    ov, lc_arr = cache(t, y, template)
    for row in range(len(ov)):
        assert len(lc_arr[row]) <= ov["width_in_samples"][row]
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


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("T0_fit_margin", [0.0, 0.01, 0.05])
@pytest.mark.parametrize("with_dy", [True, False])
@pytest.mark.parametrize("template", list(TEMPLATES))
def test_fused_kernel_matches_brute_force(seed, T0_fit_margin, with_dy, template):
    """core_fused.search_period_fused (backend "fused") vs. brute force."""
    from transitleastsquares.backends import SearchProblem
    from transitleastsquares.core_fused import FusedProblem

    t, y, dy = make_case(seed, n=900, with_dy=with_dy)
    ov, lc_arr = cache(t, y, template)
    lim = (C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX)
    fused = FusedProblem(SearchProblem(t, y, dy, lc_arr, ov, 1e-5, *lim, T0_fit_margin))
    # piecewise-linear path (idea L3) with every sample a knot: exact algebra
    fused_pl = FusedProblem(
        SearchProblem(t, y, dy, lc_arr, ov, 1e-5, *lim, T0_fit_margin)
    )
    fused_pl.set_pl(2, 1 << 62, 0.0)
    assert numpy.sum(fused_pl.pl[0] > 0) > 0
    fused_screen = FusedProblem(
        SearchProblem(t, y, dy, lc_arr, ov, 1e-5, *lim, T0_fit_margin)
    )
    fused_screen.set_screen(min_length=48)
    assert fused_screen.screen is not None
    for period in [2.3, 1.7, 4.11, 0.93, 0.61]:
        b_chi2, b_row, b_depth = brute_force_period(
            period, t, y, dy, 1e-5, lc_arr, ov, *lim, T0_fit_margin
        )
        for fp, rtol in [(fused, 1e-9), (fused_pl, 1e-8), (fused_screen, 1e-9)]:
            _, chi2, row, depth = fp.search(period)
            numpy.testing.assert_allclose(chi2, b_chi2, rtol=rtol)
            assert row == b_row
            numpy.testing.assert_allclose(depth, b_depth, rtol=1e-9)


@pytest.mark.parametrize("length", [2, 3, 17, 250])
@pytest.mark.parametrize("eps", [0.0, 1e-3, 1e-2])
def test_pl_fit_algebra(length, eps):
    """pl_fit: sum_j p_j x[i+j] == sum_t c_t D[i+pos_t] (double prefix sums),
    p within eps * max|a| of a, and the exact case eps = 0."""
    from transitleastsquares.core_fused import pl_fit

    rng = numpy.random.default_rng(length)
    a = numpy.sin(numpy.linspace(0, 3, length)) ** 2 + 0.01
    pos, c, psum, p = pl_fit(a, eps)
    if eps == 0:
        numpy.testing.assert_array_equal(p, a)
    else:
        assert numpy.max(numpy.abs(p - a)) <= 2 * eps * numpy.max(a)
    x = rng.normal(size=length + 50)
    S = numpy.concatenate([[0.0], numpy.cumsum(x)])
    D = numpy.concatenate([[0.0], numpy.cumsum(S)])
    for i in [0, 7, 48]:
        numpy.testing.assert_allclose(
            numpy.dot(c, D[i + pos]), numpy.dot(p, x[i : i + length]), atol=1e-10
        )
    numpy.testing.assert_allclose(psum, numpy.sum(p))

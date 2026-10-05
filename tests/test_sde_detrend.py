"""Tests of the SDE_detrend parameter ("median" vs "hybrid").

The default ("median") must reproduce the previous behaviour exactly; the
"hybrid" detrend (analytic a + b*lnP background + 3x wider running median of
the residual) must be deterministic, robust against a strong injected peak
(the robust fit must not be dragged by it), and recover the same period.
"""

import numpy
import pytest

from transitleastsquares import tls_constants, transitleastsquares
from transitleastsquares.helpers import running_median
from transitleastsquares.stats import (
    _robust_gev_fit,
    _robust_lnp_fit,
    sde_trend,
    spectra,
)


def noise_curve(seed, n=4000, ppm=300):
    rng = numpy.random.default_rng(seed)
    t = numpy.arange(0, 40, 40 / n)
    y = 1 + rng.normal(0, 1, len(t)) * ppm * 1e-6
    return t, y


def periodogram(t, y, **kwargs):
    return transitleastsquares(t, y, verbose=False).power(
        use_threads=1, show_progress_bar=False, verbose=False, **kwargs
    )


def test_default_is_median_and_backward_compatible():
    t, y = noise_curve(1)
    # the module default
    assert tls_constants.SDE_DETREND == "median"
    assert "SDE_detrend" in tls_constants.VALID_PARAMETERS
    # spectra() without periods behaves like the old median detrend
    chi2 = numpy.linspace(2, 1, 500) + numpy.random.default_rng(3).normal(0, 1e-3, 500)
    SR, pr, power, SDE_raw, SDE = spectra(chi2, 3)
    kernel = 3 * tls_constants.SDE_MEDIAN_KERNEL_SIZE
    if kernel % 2 == 0:
        kernel += 1
    expected = pr - running_median(pr, kernel)
    expected = expected - expected.mean()
    scale = (expected.max() / expected.std()) / expected.max()
    numpy.testing.assert_allclose(power, expected * scale)


def test_hybrid_matches_median_period_on_noise():
    t, y = noise_curve(2)
    r_med = periodogram(t, y, SDE_detrend="median")
    r_hyb = periodogram(t, y, SDE_detrend="hybrid")
    assert r_med.periods.shape == r_hyb.periods.shape
    # the same peak (or one of equal quality) must be found
    assert numpy.isclose(
        r_med.periods[numpy.argmax(r_med.power)],
        r_hyb.periods[numpy.argmax(r_hyb.power)],
        rtol=0.01,
    )
    # SDE within the intrinsic jitter (~1) of the method
    assert abs(r_med.SDE - r_hyb.SDE) < 2


def test_hybrid_transit_and_determinism():
    rng = numpy.random.default_rng(7)
    t = numpy.arange(0, 40, 40 / 6000)
    y = 1 + rng.normal(0, 300e-6, len(t))
    # inject a strong transit (robust fit must not chase it)
    P, T0, dur, depth = 3.0, 1.0, 0.1, 5000e-6
    in_transit = numpy.abs((t - T0 + 0.5 * P) % P - 0.5 * P) < 0.5 * dur
    y[in_transit] -= depth
    r1 = periodogram(t, y, SDE_detrend="hybrid")
    r2 = periodogram(t, y, SDE_detrend="hybrid")
    assert r1.SDE == r2.SDE  # deterministic
    numpy.testing.assert_array_equal(r1.power, r2.power)
    rel = abs(r1.period / P - 1)
    assert rel < 0.01 and r1.SDE > 10
    # signal leakage into the long-P background (partial transits in
    # few-epoch windows) can raise the fitted s above 0 — legitimate, the
    # main peak stays intact. Bound it:
    _, _, s, _ = _robust_gev_fit(r1.power_raw, r1.periods)
    assert 0 <= s < 1.6


def test_gev_s_zero_for_white_noise():
    """The GEV law must degenerate to the log law (s ~ 0) for a Gaussian
    periodogram background: fit a controlled synthetic spectrum."""
    rng = numpy.random.default_rng(31)
    n = 5000
    P = numpy.exp(numpy.linspace(numpy.log(0.6), numpy.log(30), n))
    a_true, b_true = -0.5, 0.5
    pr = a_true + b_true * numpy.log(P) + rng.normal(0, 0.8, n)
    a, b, s, x = _robust_gev_fit(pr, P)
    assert abs(s) < 0.15
    # a is defined on the ln(P/P0) basis; b is the log slope
    numpy.testing.assert_allclose(b, b_true, atol=0.1)
    numpy.testing.assert_allclose(
        a, a_true + b_true * numpy.log(numpy.exp(numpy.mean(numpy.log(P)))), atol=0.15
    )


def test_gev_law_tracks_heavy_tailed_background():
    """Heavy-tailed noise (rare strong outliers) drives the periodogram
    background up as a power law at long P; the GEV fit must detect this
    (s > 0) and leave flat residuals where the log law leaves a rise."""
    rng = numpy.random.default_rng(23)
    n = 5000
    P = numpy.exp(numpy.linspace(numpy.log(0.6), numpy.log(136), n))
    pr = rng.normal(0, 0.8, n)
    # synthetic power-law-in-P background (what outlier-driven min-chi2
    # produces at long P, cf. scratch/sde_analytic §8: Kepler-21-like)
    pr = pr + 8.0 * ((P / 136.0) ** 1.0 - 0.1)
    a, b, s, x = _robust_gev_fit(pr, P)
    assert s > 0.4  # the power law is detected
    resid = pr - a - b * x
    lnP = numpy.log(P)
    bins = numpy.linspace(lnP[0], lnP[-1], 6)
    edges = [
        resid[(lnP >= lo) & (lnP < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])
    ]
    assert numpy.max(numpy.abs(edges)) < 0.25  # flat within noise
    # the log law (s=0) leaves a strong rise
    a2, b2 = _robust_lnp_fit(pr, P)
    r2 = pr - a2 - b2 * lnP
    e2 = [r2[(lnP >= lo) & (lnP < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
    assert e2[-1] - e2[0] > 1.0


def test_sde_trend_none_for_short_grid():
    # shorter than twice the kernel: no detrending, like TLS <= 1.33
    pr = numpy.random.default_rng(5).normal(0, 0.8, 100)
    P = numpy.linspace(1, 2, 100)
    assert sde_trend(pr, P, 3) is None
    assert sde_trend(pr, P, 3, detrend="hybrid") is None
    with pytest.raises(ValueError):
        sde_trend(pr, P, 3, detrend="bogus")


def test_hybrid_edge_taper_tracks_steep_boundary_rise():
    """Real data can rise steeply above the analytic (white-noise) trend
    towards P_max. In the edge zone the hybrid kernel tapers to the
    "median" width, so the trend must track such a rise almost as well
    as the pure median detrend."""
    rng = numpy.random.default_rng(11)
    n = 6000
    P = numpy.exp(numpy.linspace(numpy.log(0.6), numpy.log(50), n))
    pr = rng.normal(0, 0.8, n) + 4.0 * (numpy.log(P / P[-1])) ** 2  # steep tail
    trend = sde_trend(pr, P, 3, detrend="hybrid")
    med = sde_trend(pr, P, 3, detrend="median")
    zone = slice(n - max(1, int(0.05 * n)), None)
    err_hybrid = numpy.mean(numpy.abs(trend[zone] - pr[zone]))
    err_median = numpy.mean(numpy.abs(med[zone] - pr[zone]))
    assert err_hybrid < 1.3 * err_median + 0.1

from __future__ import division, print_function
import numpy
import batman
from transitleastsquares import transitleastsquares


if __name__ == "__main__":
    print("Starting test: transit_depth_min...", end="")
    # Test for transit_depth_min=1000*10**-6, where no transit is found

    # Create test data
    start = 48
    days = 365.25 * 3
    samples_per_day = 12  # 48
    samples = int(days * samples_per_day)  # 48
    t = numpy.linspace(start, start + days, samples)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = (
        start + 20
    )  # time of inferior conjunction; first transit is X days after start
    ma.per = 365.25  # orbital period
    ma.rp = 6371 / 696342  # 6371 planet radius (in units of stellar radii)
    ma.a = 217  # semi-major axis (in units of stellar radii)
    ma.inc = 90  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.5]  # limb darkening coefficients
    ma.limb_dark = "linear"  # limb darkening model
    m = batman.TransitModel(ma, t)  # initializes model
    original_flux = m.light_curve(ma)  # calculates light curve

    # Create noise and merge with flux
    ppm = 5
    stdev = 10 ** -6 * ppm
    numpy.random.seed(seed=0)  # reproducibility (was unseeded)
    noise = numpy.random.normal(0, stdev, int(samples))
    y = original_flux + noise
    y[1] = numpy.nan
    model = transitleastsquares(t, y)
    results = model.power(
        transit_depth_min=1000 * 10 ** -6,
        period_min=360,
        period_max=370,
        oversampling_factor=5,
        duration_grid_step=1.02,
        T0_fit_margin=0.1,
    )

    numpy.testing.assert_equal(results.transit_times, numpy.nan)
    numpy.testing.assert_equal(results.period, numpy.nan)
    numpy.testing.assert_equal(results.depth, 1)
    numpy.testing.assert_equal(results.duration, numpy.nan)
    numpy.testing.assert_equal(results.snr, numpy.nan)
    numpy.testing.assert_equal(results.snr_pink_per_transit, numpy.nan)
    numpy.testing.assert_equal(results.odd_even_mismatch, numpy.nan)
    numpy.testing.assert_equal(results.SDE, 0)
    numpy.testing.assert_equal(results.SDE_raw, 0)
    numpy.testing.assert_equal(results.in_transit_count, numpy.nan)
    numpy.testing.assert_equal(results.after_transit_count, numpy.nan)
    numpy.testing.assert_equal(results.before_transit_count, numpy.nan)
    # Since TLS 1.33 (PR #112) the "no fit" chi2 is sum((y-1)^2/dy^2) with
    # dy = std(y) (no dy given), not len(y) as before.
    y_clean = y[numpy.isfinite(y)]
    chi2_expected = numpy.sum((y_clean - 1) ** 2 / numpy.std(y_clean) ** 2)
    chi2red_expected = chi2_expected / (len(y_clean) - 4)
    numpy.testing.assert_almost_equal(results.chi2_min, chi2_expected, decimal=5)
    numpy.testing.assert_almost_equal(results.chi2red_min, chi2red_expected)
    numpy.testing.assert_equal(len(results.periods), 278)
    numpy.testing.assert_almost_equal(max(results.periods), 369.9831654894093)
    numpy.testing.assert_almost_equal(min(results.periods), 360.0118189140635)
    numpy.testing.assert_almost_equal(max(results.power), 0)
    numpy.testing.assert_almost_equal(min(results.periods), 360.0118189140635)
    numpy.testing.assert_almost_equal(min(results.power), 0)
    numpy.testing.assert_almost_equal(max(results.chi2), chi2_expected, decimal=5)
    numpy.testing.assert_almost_equal(max(results.chi2red), chi2red_expected)
    print("Test passed: transit_depth_min=1000*10**-6, where no transit is fit")

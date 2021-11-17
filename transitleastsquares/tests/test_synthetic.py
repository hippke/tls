from __future__ import division, print_function
import os
import numpy
import batman
from transitleastsquares import transitleastsquares


if __name__ == "__main__":
    print("Starting test: synthetic...", end="")
    
    numpy.random.seed(seed=0)  # reproducibility
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
    noise = numpy.random.normal(0, stdev, int(samples))
    y = original_flux + noise
    y[1] = numpy.nan
    model = transitleastsquares(t, y)
    results = model.power(
        period_min=360,
        period_max=370,
        transit_depth_min=10 * 10 ** -6,
        oversampling_factor=5,
        duration_grid_step=1.02, 
        verbose=False 
    )

    numpy.testing.assert_almost_equal(results.chi2_min, 8831.654060613922, decimal=5)
    numpy.testing.assert_almost_equal(
        results.chi2red_min, 0.6719152511118321, decimal=5
    )

    numpy.testing.assert_almost_equal(
        results.period_uncertainty, 0.216212529678387, decimal=5
    )
    numpy.testing.assert_equal(results.per_transit_count[0], 7)
    numpy.testing.assert_equal(len(results.transit_times), 3)
    numpy.testing.assert_almost_equal(results.period, 365.2582192473641, decimal=5)
    numpy.testing.assert_almost_equal(
        results.transit_times[0], 68.00349264912924, decimal=5
    )
    """
    numpy.testing.assert_almost_equal(results.depth, 0.999897160189092, decimal=5)
    numpy.testing.assert_almost_equal(results.duration, 0.5908251624976649, decimal=5)
    numpy.testing.assert_almost_equal(
        min(results.chi2red), 0.6719167401148216, decimal=5
    )
    numpy.testing.assert_almost_equal(results.SDE, 5.691301613227594, decimal=5)
    numpy.testing.assert_almost_equal(
        results.odd_even_mismatch, 0.29083256866622437, decimal=5
    )
    numpy.testing.assert_almost_equal(results.rp_rs, 0.009119851811944274, decimal=5)

    # Full light curve model
    numpy.testing.assert_almost_equal(
        max(results.model_lightcurve_time), 1143.7472155961277, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results.model_lightcurve_time), 48.0059010663453, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.mean(results.model_lightcurve_time), 595.877471821318, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.sum(results.model_lightcurve_time), 39166430.34534341, decimal=5
    )

    numpy.testing.assert_almost_equal(max(results.model_lightcurve_model), 1, decimal=5)
    numpy.testing.assert_almost_equal(
        min(results.model_lightcurve_model), 0.999897160189092, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.mean(results.model_lightcurve_model), 0.9999998641488729, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.sum(results.model_lightcurve_model), 65728.99107064126, decimal=5
    )

    transit_times_expected = [68.003492, 433.261711, 798.519931]
    numpy.testing.assert_almost_equal(
        results.transit_times, transit_times_expected, decimal=5
    )
    numpy.testing.assert_almost_equal(results.duration, 0.590825, decimal=5)

    numpy.testing.assert_almost_equal(
        max(results.model_folded_phase), 1.0000380285975052, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results.model_folded_phase), 3.8028597505324e-05, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.mean(results.model_folded_phase), 0.5000380285975052, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.sum(results.model_folded_phase), 6574.499999999999, decimal=5
    )

    numpy.testing.assert_almost_equal(max(results.model_folded_model), 1, decimal=5)
    numpy.testing.assert_almost_equal(
        min(results.model_folded_model), 0.999897160189092, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.mean(results.model_folded_model), 0.9999998679702978, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.sum(results.model_folded_model), 13147.998264073476, decimal=5
    )

    numpy.testing.assert_almost_equal(
        max(results.folded_phase), 0.9999608485845858, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results.folded_phase), 1.44015016259047e-05, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.mean(results.folded_phase), 0.500000089528271, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.sum(results.folded_phase), 6574.001177117707, decimal=5
    )

    numpy.testing.assert_almost_equal(
        max(results.folded_y), 1.000019008301075, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results.folded_y), 0.9998860842491378, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.mean(results.folded_y), 0.9999997920032417, decimal=5
    )
    numpy.testing.assert_almost_equal(
        numpy.sum(results.folded_y), 13147.997265, decimal=5
    )

    numpy.testing.assert_almost_equal(
        results.depth_mean_even, (0.999915, 6.785539e-06), decimal=5
    )
    numpy.testing.assert_almost_equal(
        results.depth_mean_odd, (0.999920, 1.209993e-05), decimal=5
    )
    numpy.testing.assert_almost_equal(
        results.depth_mean, (0.999917, 6.086923e-06), decimal=5
    )

    numpy.testing.assert_almost_equal(
        results.transit_depths, [0.99991085, 0.99992095, 0.99992007], decimal=5
    )
    numpy.testing.assert_almost_equal(
        results.transit_depths_uncertainties,
        [4.19177855e-06, 1.20999330e-05, 1.26699399e-05],
        decimal=5,
    )
    numpy.testing.assert_almost_equal(
        results.odd_even_mismatch, 0.29083256866622437, decimal=5
    )
    numpy.testing.assert_almost_equal(
        results.per_transit_count, [7.0, 7.0, 7.0], decimal=5
    )
    numpy.testing.assert_almost_equal(results.transit_count, 3, decimal=5)
    numpy.testing.assert_almost_equal(results.distinct_transit_count, 3, decimal=5)
    numpy.testing.assert_almost_equal(results.empty_transit_count, 0, decimal=5)
    numpy.testing.assert_almost_equal(
        results.snr_per_transit, [38.92162, 34.51048, 34.89514], decimal=5
    )
    numpy.testing.assert_almost_equal(results.snr, 62.542764907612785, decimal=5)
    numpy.testing.assert_almost_equal(
        results.snr_pink_per_transit, [52.24377, 46.32278, 46.8391], decimal=5
    )
    """
    print("passed")

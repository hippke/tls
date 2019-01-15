import numpy
import batman
from transitleastsquares import transitleastsquares, catalog_info, period_grid, \
    get_duration_grid, FAP


if __name__ == '__main__':
    numpy.random.seed(seed=0)  # reproducibility
    print('Starting tests. This should take less than one minute...')

    numpy.testing.assert_equal(FAP(SDE=2), numpy.nan)
    numpy.testing.assert_equal(FAP(SDE=7), 0.009443778)
    numpy.testing.assert_equal(FAP(SDE=99), 8.0032e-05)
    print('Test passed: FAP table')

    periods = period_grid(
            R_star=1,  # R_sun
            M_star=1,  # M_sun
            time_span=0.1)  # days
    numpy.testing.assert_almost_equal(max(periods), 2.4999999999999987)
    numpy.testing.assert_almost_equal(min(periods), 0.6002621413799498)
    numpy.testing.assert_equal(len(periods), 179)

    periods = period_grid(
            R_star=1,  # R_sun
            M_star=1,  # M_sun
            time_span=20)  # days
    numpy.testing.assert_almost_equal(max(periods), 10)
    numpy.testing.assert_almost_equal(min(periods), 0.6009180713191087)
    numpy.testing.assert_equal(len(periods), 1145)

    periods = period_grid(
            R_star=5,  # R_sun
            M_star=1,  # M_sun
            time_span=20,  # days
            period_min=0,
            period_max=999,
            oversampling_factor=3)
    numpy.testing.assert_almost_equal(max(periods), 10)
    numpy.testing.assert_almost_equal(min(periods), 0.6009180713191087)
    numpy.testing.assert_equal(len(periods), 1145)

    periods = period_grid(
            R_star=1,  # R_sun
            M_star=1,  # M_sun
            time_span=20,  # days
            period_min=0,
            period_max=999,
            oversampling_factor=3)
    numpy.testing.assert_almost_equal(max(periods), 10)
    numpy.testing.assert_almost_equal(min(periods), 0.60155759)
    numpy.testing.assert_equal(len(periods), 1716)
    print('Test passed: period_grid')


    # Duration grid
    durations = get_duration_grid(periods, log_step=1.05, shortest=2)
    numpy.testing.assert_almost_equal(max(durations), 0.12)
    numpy.testing.assert_almost_equal(min(durations), 0.011618569353576557)
    numpy.testing.assert_equal(len(durations), 49)
    print('Test passed: get_duration_grid')

    # 266980320
    # 279741377
    # 394137592
    # 261136679
    try:
        (a, b), mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=261136679)
        numpy.testing.assert_equal((a, b), (0.4224, 0.3037))
        numpy.testing.assert_equal(mass, 0.509)
        numpy.testing.assert_equal(radius, 0.498)
    except:
        print('catalog_info for TIC_ID failed')
        error=True

    (a, b), mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(EPIC_ID=204099713)
    numpy.testing.assert_almost_equal((a, b), (0.4804, 0.1867))
    numpy.testing.assert_almost_equal(mass, 1.046)
    numpy.testing.assert_almost_equal(mass_min, 0.898)
    numpy.testing.assert_almost_equal(mass_max, 0.642)
    numpy.testing.assert_almost_equal(radius, 1.261)
    numpy.testing.assert_almost_equal(radius_min, 1.044)
    numpy.testing.assert_almost_equal(radius_max, 0.925)
    print('Test passed: EPIC catalog pull from Vizier using astroquery')

    (a, b), mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(KOI_ID='952.01')
    numpy.testing.assert_equal((a, b), (0.4224, 0.3037))
    numpy.testing.assert_equal(mass, 0.509)
    numpy.testing.assert_equal(radius, 0.498)
    print('Test passed: KIC catalog pull from MAST using kplr')

    # Create test data
    start = 48
    days = 365.25 * 3
    samples_per_day = 12  # 48
    samples = int(days * samples_per_day) # 48
    t = numpy.linspace(start, start + days, samples)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = start + 20 # time of inferior conjunction; first transit is X days after start
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
    stdev = 10**-6 * ppm
    noise = numpy.random.normal(0, stdev, int(samples))
    y = original_flux + noise
    model = transitleastsquares(t, y)
    results = model.power(
        period_min=360,
        period_max=370,
        transit_depth_min=10*10**-6,
        oversampling_factor=5,
        duration_grid_step=1.02)

    numpy.testing.assert_equal(results.per_transit_count[0], 7)
    numpy.testing.assert_equal(len(results.transit_times), 3)

    numpy.testing.assert_almost_equal(results.period, 365.2582192473641, decimal=5)
    numpy.testing.assert_almost_equal(results.transit_times[0], 68.00197123958793, decimal=5)
    numpy.testing.assert_almost_equal(results.depth, 0.999897160189092, decimal=5)
    numpy.testing.assert_almost_equal(results.duration, 0.5908701024202706, decimal=5)
    numpy.testing.assert_almost_equal(min(results.chi2red), 0.6719167401148216, decimal=5)
    numpy.testing.assert_almost_equal(results.SDE, 5.691301613227594, decimal=5)
    numpy.testing.assert_almost_equal(results.snr, 105.53669779138568, decimal=5)
    numpy.testing.assert_almost_equal(results.odd_even_mismatch, 0.005767912763555982, decimal=5)
    numpy.testing.assert_almost_equal(results.snr_per_transit[0], 47.52343719198146, decimal=5)
    numpy.testing.assert_almost_equal(results.snr_pink_per_transit[0], 53.37882224182496, decimal=5)
    numpy.testing.assert_almost_equal(results.rp_rs, 0.009119851811944274, decimal=5)


    if not error:
        print('All tests completed successfully.')
    else:
        print('Some tests failed')

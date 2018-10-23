import numpy
from TransitLeastSquares_v28 import period_grid, get_duration_grid, \
    get_depth_grid, get_catalog_info


# Period grid
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

# Duration grid
durations = get_duration_grid(periods, log_step=1.05, upper_limit=0.15)
numpy.testing.assert_almost_equal(max(durations), 0.15)
numpy.testing.assert_almost_equal(min(durations), 0.004562690)
numpy.testing.assert_equal(len(durations), 73)

# Depth grid
numpy.random.seed(0)
samples = 100
stdev = 0.01
y = numpy.random.normal(1, stdev, samples)
shallowest_transit = 10**-6
depths = get_depth_grid(y, deepest=None, shallowest=shallowest_transit, log_step=1.05)
numpy.testing.assert_almost_equal(1-max(depths), min(y))
numpy.testing.assert_almost_equal(1-min(depths), 1-shallowest_transit)
numpy.testing.assert_almost_equal(len(depths), 209)

# EPIC Catalog info
u, a, b, mass, radius, logg, Teff = get_catalog_info(EPIC=204099713)
numpy.testing.assert_equal(u, 0.606)
numpy.testing.assert_equal(a, 0.4804)
numpy.testing.assert_equal(b, 0.1867)
numpy.testing.assert_equal(mass, 1.046)
numpy.testing.assert_equal(radius, 1.261)
numpy.testing.assert_equal(logg, 4.214)
numpy.testing.assert_equal(Teff, 6025)

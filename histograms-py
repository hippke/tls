import numpy
import scipy
import matplotlib.pyplot as plt
from astropy.io import fits
import batman
from TransitLeastSquares import TransitLeastSquares, period_grid, fold


if __name__ == '__main__':

    # Create test data
    days = 365.25 * 2.1
    samples_per_day = 12  # 48
    samples = int(days * samples_per_day) # 48
    t = numpy.linspace(0, days, samples)

    stat_empty = numpy.array([])
    list_peaks = numpy.array([])

    # Empty noise runs
    for run in range(2):

        # Use batman to create transits
        ma = batman.TransitParams()
        ma.t0 = 10 # time of inferior conjunction; first transit is X days after start
        ma.per = 365.25  # orbital period
        ma.rp = 6.371 / 696342  # 6371 planet radius (in units of stellar radii)
        ma.a = 217  # semi-major axis (in units of stellar radii)
        ma.inc = 90  # orbital inclination (in degrees)
        ma.ecc = 0  # eccentricity
        ma.w = 90  # longitude of periastron (in degrees)
        ma.u = [0.5]  # limb darkening coefficients
        ma.limb_dark = "linear"  # limb darkening model
        m = batman.TransitModel(ma, t)  # initializes model
        original_flux = m.light_curve(ma)  # calculates light curve

        # Create noise and merge with flux
        ppm = 40
        stdev = 10**-6 * ppm
        noise = numpy.random.normal(0, stdev, int(samples))
        y_filt = original_flux + noise
        dy = numpy.full(numpy.size(y_filt), numpy.std(y_filt))

        periods = period_grid(
            R_star=1,
            M_star=1,
            time_span=(max(t) - min(t)),
            period_min=100,
            period_max=370,
            oversampling_factor=2)

        # Define grids of transit depths and widths
        depths = numpy.geomspace(50*10**-6, 0.01, 20)
        durations = numpy.geomspace(1.01/numpy.size(t), 0.01, 20)

        model = TransitLeastSquares(t, y_filt, dy)
        results = model.power(periods, durations, depths, limb_darkening=0.5)

        stat_empty = numpy.append(stat_empty, results.chi2red)


    # Signal runs
    for run in range(10):

        # Use batman to create transits
        ma = batman.TransitParams()
        ma.t0 = 10 # time of inferior conjunction; first transit is X days after start
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
        noise = numpy.random.normal(0, stdev, int(samples))
        y_filt = original_flux + noise
        dy = numpy.full(numpy.size(y_filt), numpy.std(y_filt))

        model = TransitLeastSquares(t, y_filt, dy)
        periods = numpy.linspace(365, 365.5, 10)
        results = model.power(periods, durations, depths, limb_darkening=0.5)

        max_peak = numpy.max(results.chi2red)
        list_peaks = numpy.append(list_peaks, max_peak)


    # Histograms
    size = 4.5
    aspect_ratio = 1.5
    plt.figure(figsize=(size, size / aspect_ratio))
    ax = plt.gca()
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    n, bins, patches = ax.hist(stat_empty, 50, range=(min(stat_empty), max_peak), color='red', alpha=0.5)  # density=1, 
    n, bins, patches = ax.hist(list_peaks, 50, range=(min(stat_empty), max_peak), color='blue', alpha=0.5)
    plt.yscale('log')
    ax.set_xlabel('Signal')
    ax.set_ylabel('Probability density')
    plt.xlim(min(stat_empty), max_peak*1.1)
    plt.savefig('fig_example_07_histogram_box.pdf', bbox_inches='tight')

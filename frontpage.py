import numpy
import scipy
import matplotlib.pyplot as plt
from astropy.io import fits
numpy.set_printoptions(threshold=numpy.nan)
from TransitLeastSquares import TransitLeastSquares, period_grid, fold
import batman


if __name__ == '__main__':
    

    # Create test data
    start = 0
    days = 365.25 * 3.5  # change this and it breaks
    samples_per_day = 12  # 48
    samples = int(days * samples_per_day) # 48
    t = numpy.linspace(start, start + days, samples)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = start + 50 # time of inferior conjunction; first transit is X days after start
    ma.per = 365.25  # orbital period
    ma.rp = 6371 / 696342  # planet radius (in units of stellar radii)
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
    print('Measured noise (std) is', format(numpy.std(noise), '.8f'))
    if numpy.std(noise) == 0:
        dy = numpy.full_like(y, 1)
    else:
        dy = numpy.full_like(y, numpy.std(noise))

    periods = period_grid(
        R_star=1,
        M_star=1,
        time_span=(max(t) - min(t)),
        period_min=365,  # 10.04
        period_max=365.5,  # 10.05
        oversampling_factor=3)

    # Define grids of transit depths and widths
    depths = numpy.geomspace(50*10**-6, 150*10**-6, 20)  # 50
    durations = numpy.geomspace(0.001, 0.002, 20)  # 50

    model = TransitLeastSquares(t, y, dy)
    results = model.power(periods, durations, depths, limb_darkening=0.5) # likelihood  # snr

    print('Period', format(results.best_period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
            ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.best_depth, '.5f'))
    print('Best duration (fractional period)', format(results.best_duration, '.5f'))
    print('Best duration (days)', format(results.transit_duration_in_days, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)

    #print(results.power)
    
    # Test statistic
    plt.figure(figsize=(4.5, 4.5 / 1.5))
    ax = plt.gca()
    ax.axvline(results.best_period, alpha=0.4, lw=3)
    plt.xlim(numpy.min(periods), numpy.max(periods))
    for n in range(2, 10):
        ax.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.savefig('fig_test_stat_raw.pdf', bbox_inches='tight')
    
    
    # Folded model
    phases = fold(t, results.best_period, T0=results.best_T0+ results.best_period/2)
    #phases = fold(t, 1.796, T0=0)
    sort_index = numpy.argsort(phases)
    phases = phases[sort_index]
    flux = y[sort_index]
    plt.figure(figsize=(4.5, 4.5 / 1.5))
    plt.scatter(phases, flux, color='blue', s=10, alpha=0.5, zorder=2)
    plt.plot(numpy.linspace(0, 1, numpy.size(results.folded_model)), 
        results.folded_model, color='red', zorder=3)
    plt.ylim(0.99985, 1.00005)
    plt.xlim(0.498, 0.502)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux (ppm)')
    plt.savefig('fig_test_fold.pdf', bbox_inches='tight')
    
    """
    # Final plot
    plt.subplot(221)
    plt.plot(data[0])
    # 2x2, second axis
    plt.subplot(222)
    plt.plot(data[1])
    # 2x2, third axis
    plt.subplot(223)
    plt.plot(data[2])
    # 2x2, fourth axis
    plt.subplot(224)
    plt.plot(data[3])
    """

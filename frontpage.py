import numpy
import scipy
import matplotlib.pyplot as plt
from astropy.io import fits
numpy.set_printoptions(threshold=numpy.nan)
from TransitLeastSquares import TransitLeastSquares, period_grid, fold
import batman


if __name__ == '__main__':
    
    numpy.random.seed(seed=2)  # reproducibility 
    # 2: points look OK; 15.6/15.8 (10ppm, 24)
    # 0: points mhm; 10.2/10.3 (10ppm, 24)
    # Create test data
    start = 0
    days = 365.25 * 3
    samples_per_day = 24  # 48
    samples = int(days * samples_per_day) # 48
    t = numpy.linspace(start, start + days, samples)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = start + 5 # time of inferior conjunction; first transit is X days after start
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
        period_min=0,  # 10.04
        period_max=500,  # 10.05
        oversampling_factor=5)

    # Define grids of transit depths and widths
    depths = numpy.geomspace(90*10**-6, 110*10**-6, 10)  # 50
    durations = numpy.geomspace(0.0013, 0.0018, 5)  # 50

    model = TransitLeastSquares(t, y, dy)
    results = model.power(periods, durations, depths, limb_darkening=0.5) # likelihood  # snr

    print('Period', format(results.best_period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
            ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.best_depth, '.5f'))
    print('Best duration (fractional period)', format(results.best_duration, '.5f'))
    print('Best duration (days)', format(results.transit_duration_in_days, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)
    
    # Final plot
    phases = fold(t, results.best_period, T0=results.best_T0+ results.best_period/2)
    sort_index = numpy.argsort(phases)
    phases = phases[sort_index]
    flux = y[sort_index]
    plt.figure(figsize=(8, 6))


    plt.subplot(221)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.errorbar(numpy.linspace((-results.best_period/2)*24, (results.best_period/2)*24, numpy.size(results.folded_model)), (flux-1)*10**6, yerr=ppm, fmt='o', color='red',
        alpha=0.5, markersize=1, zorder=2)
    plt.plot(numpy.linspace((-results.best_period/2)*24, (results.best_period/2)*24, numpy.size(results.folded_model)),
        (results.folded_model-1)*10**6, color='blue', zorder=0)
    plt.ylim(-120, 20)
    plt.xlim(-15, 15)
    plt.ylabel(r'Flux (ppm)')
    plt.xlabel('Time from mid-transit (days)')

    plt.subplot(223)
    plt.axvline(results.best_period, alpha=0.4, lw=3)
    for n in range(2, 10):
        plt.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
        plt.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.xlim(0, 500)
    max_sde_ld = results.SDE
    scale = 1.1
    max_plot_height = max_sde_ld * scale
    plt.ylim(-1, max_plot_height)
    plt.text(350, max_sde_ld, r'SDE$_{\rm peak}=$' + str(round(results.SDE, 1)), ha='right')


    # Box
    model = TransitLeastSquares(t, y, dy)
    results = model.power(periods, durations, depths, limb_darkening=0.0) # likelihood  # snr

    plt.subplot(222)

    # Make the box look like a box when plotting
    #modelflux = (results.folded_model-1)*10**6
    #depth = results.best_depth
    # Left
    plt.plot(
        ((-results.best_period/2)*24, -(results.transit_duration_in_days/2)*24),
        (0, 0),
        color='blue',
        zorder=10)
    # Right
    plt.plot(
        ((+results.best_period/2)*24, +(results.transit_duration_in_days/2)*24),
        (0, 0),
        color='blue',
        zorder=10)
    # Bottom
    plt.plot(
        ((-(results.transit_duration_in_days/2)*24, +(results.transit_duration_in_days/2)*24)),
        ((-results.best_depth)*10**6, (-results.best_depth)*10**6),
        color='blue',
        zorder=10)
    # Left
    plt.plot(
        ((-(results.transit_duration_in_days/2)*24, -(results.transit_duration_in_days/2)*24)),
        (0, (-results.best_depth)*10**6),
        color='blue',
        zorder=10)
    # Right
    plt.plot(
        ((+(results.transit_duration_in_days/2)*24, +(results.transit_duration_in_days/2)*24)),
        ((-results.best_depth)*10**6, 0),
        color='blue',
        zorder=10)

    print((results.best_depth)*10**6)



    plt.errorbar(numpy.linspace((-results.best_period/2)*24, (results.best_period/2)*24, numpy.size(results.folded_model)), (flux-1)*10**6, yerr=ppm, fmt='o', color='red',
        alpha=0.5, markersize=1, zorder=2)
    #plt.plot(numpy.linspace((-results.best_period/2)*24, (results.best_period/2)*24, numpy.size(results.folded_model)),
    #    (results.folded_model-1)*10**6, color='blue', zorder=0)

    plt.ylim(-120, 20)
    plt.xlim(-15, 15)
    #plt.ylabel(r'Flux (ppm)')
    plt.xlabel('Time from mid-transit (days)')

    plt.subplot(224)
    plt.axvline(results.best_period, alpha=0.4, lw=3)
    for n in range(2, 10):
        plt.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
        plt.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    #plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.xlim(0, 500)
    plt.ylim(-1, max_plot_height)
    plt.text(350, max_sde_ld, r'SDE$_{\rm peak}=$' + str(round(results.SDE, 1)), ha='right')

    plt.savefig('frontpage.pdf', bbox_inches='tight')

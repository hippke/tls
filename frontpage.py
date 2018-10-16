import numpy
import matplotlib.pyplot as plt
from TransitLeastSquares_v41 import TransitLeastSquares, period_grid, fold,\
    running_mean, foldfast

import batman
import time


if __name__ == '__main__':
    
    numpy.random.seed(seed=0)  # reproducibility 
    # 2: points look OK; 15.6/15.8 (10ppm, 24)
    # 0: points mhm; 10.2/10.3 (10ppm, 24)
    # Create test data
    start = 0
    days = 365.25 * 3
    samples_per_day = 12  # 48
    samples = int(days * samples_per_day) # 48
    t = numpy.linspace(start, start + days, samples)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = start + 180 # time of inferior conjunction; first transit is X days after start
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
    """
    t1 = time.perf_counter()
    for i in range(12000):
        phases = fold(t, i, 0)
        sort_index = numpy.argsort(phases, kind='mergesort')
    t2 = time.perf_counter()
    print(t2-t1)
    """


    # Create noise and merge with flux
    ppm = 1
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
        period_min=350,  # 10.04
        period_max=400,  # 10.05
        oversampling_factor=10)

    #print(periods[0], periods[1], periods[0]-periods[1])

    # Define grids of transit depths and widths
    #depths = numpy.linspace(50*10**-6, 120*10**-6, 7)  # 50
    #durations = numpy.geomspace(0.001, 0.002, 5)  # 50

    #durations = get_duration_grid(periods, log_step=1.02)
    #print(len(durations), 'durations from', min(durations), 'to', max(durations))
    #depths = get_depth_grid(y, shallowest=10*10**-6, log_step=1.1)
    #print(len(depths), 'depths from', min(depths), 'to', max(depths))


    model = TransitLeastSquares(t, y, dy)
    results = model.power(
        period_min=350,
        period_max=400)

    print('Period', format(results.best_period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
            ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.best_depth, '.5f'))
    print('Best duration (fractional period)', format(results.best_duration, '.5f'))
    print('Best duration (days)', format(results.transit_duration_in_days, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)
    print('SDE TLS', results.SDE, max(results.SDE_power))
    
    # Final plot
    phases = fold(t, results.best_period, T0=results.best_T0+ results.best_period/2)
    sort_index = numpy.argsort(phases)
    phases = phases[sort_index]
    flux = y[sort_index]
    plt.figure(figsize=(8, 6))


    plt.subplot(221)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    data = (flux-1)*10**6
    data = numpy.append(data, [99])
    time = numpy.linspace((-results.best_period/2)*24, (results.best_period/2)*24, 
        numpy.size(results.folded_model))
    time = numpy.append([99], time)
    plt.errorbar(time, data, yerr=ppm, fmt='o', color='red', alpha=1, markersize=1, zorder=2)
    plt.plot(numpy.linspace(
        (-results.best_period/2)*24,
        (results.best_period/2)*24,
        numpy.size(results.folded_model)),
        (results.folded_model-1)*10**6, color='blue', zorder=3)
    #plt.ylim(-120, 20)
    plt.xlim(-15, 15)
    plt.text(0, 0, 'TLS', ha='center')
    plt.ylabel(r'Flux (ppm)')
    plt.xlabel('Time from mid-transit (hours)')

    plt.subplot(223)
    plt.axvline(results.best_period, alpha=0.4, lw=3)
    for n in range(2, 10):
        plt.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
        plt.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.cleaned_SDE_power, color='black', lw=0.5)
    #plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.xlim(min(periods), max(periods))
    max_sde_ld = results.cleaned_SDE
    scale = 1.1
    max_plot_height = max_sde_ld * scale
    plt.ylim(-1, max_plot_height)
    plt.text(360, results.cleaned_SDE, r'SDE=' + str(round(results.cleaned_SDE, 1)), ha='right')

    # Box
    """
    model = TransitLeastSquares(t, y, dy)
    results = model.power(periods, durations, depths, limb_darkening=0.0) # likelihood  # snr
    """
    from astropy.stats import BoxLeastSquares
    import time
    durations = numpy.linspace(0.2, 1, 25)
    t1 = time.perf_counter()
    model = BoxLeastSquares(t, y)
    results_bls = model.power(periods, durations)
    t2 = time.perf_counter()
    print('Time BLS', t2-t1)
    chi2 = 1/results_bls.power
    SR = min(chi2) / chi2
    SDE = (1 - numpy.mean(SR)) / numpy.std(SR)
    SDE_power = SR - numpy.min(SR)      # shift down to touch 0
    scale = SDE / numpy.max(SDE_power)  # scale factor to touch max=SDE
    SDE_power = SDE_power * scale
    print('SDE BLS', SDE, max(SDE_power))


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

    data = (flux-1)*10**6
    data = numpy.append(data, [99])
    time = numpy.linspace((-results.best_period/2)*24, (results.best_period/2)*24, 
        numpy.size(results.folded_model))
    time = numpy.append([99], time)
    plt.errorbar(time, data, yerr=ppm, fmt='o', color='red', alpha=0.5, markersize=1, zorder=2)
    plt.text(0, 0, 'BLS', ha='center')
    plt.ylim(-120, 20)
    plt.xlim(-15, 15)
    #plt.ylabel(r'Flux (ppm)')
    plt.xlabel('Time from mid-transit (hours)')

    plt.subplot(224)
    plt.axvline(results.best_period, alpha=0.4, lw=3)
    for n in range(2, 10):
        plt.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
        plt.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")

    plt.plot(results_bls.period, SDE_power, color='black', lw=0.5)
    #plt.plot(results_bls.period, results_bls.power, color='black', lw=0.5)
    #plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.xlim(min(periods), max(periods))
    plt.ylim(-1, max_plot_height)
    plt.text(360, results.cleaned_SDE, r'SDE=' + str(round(SDE, 1)), ha='right')

    plt.savefig('frontpage.pdf', bbox_inches='tight')
    plt.savefig('frontpage.png', bbox_inches='tight', dpi=300)

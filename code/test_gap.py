import numpy
import matplotlib.pyplot as plt
from TransitLeastSquares import TransitLeastSquares, fold, running_mean, foldfast, pink_noise, transit_mask
import batman
import time


if __name__ == '__main__':
    
    numpy.random.seed(seed=0)  # reproducibility 
    # 2: points look OK; 15.6/15.8 (10ppm, 24)
    # 0: points mhm; 10.2/10.3 (10ppm, 24)
    # Create test data
    start = 10
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
    print('Measured noise (std) is', format(numpy.std(noise), '.8f'))

    # Insert gap
    t_new = []
    y_new = []
    for i in range(len(y)):
        if i <= 6000 or i >= 8000:
            t_new.append(t[i])
            y_new.append(y[i])
    t = numpy.array(t_new)
    y = numpy.array(y_new)
    dy = numpy.full_like(y, numpy.std(noise))

    plt.figure(figsize=(8, 6))
    plt.scatter(t, y, color='blue', alpha=0.5, s=1)
    plt.ylim(0.999875, 1.00003)
    plt.ylabel(r'Flux (ppm)')
    plt.xlabel('Phase')
    plt.savefig('series.pdf', bbox_inches='tight')

    model = TransitLeastSquares(t, y, dy)
    results = model.power(
        period_min=360,
        period_max=370,
        transit_depth_min=10*10**-6,
        oversampling_factor=10,
        duration_grid_step=1.02)

    print('Period', format(results.period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
            ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.depth, '.5f'))
    print('Best duration (days)', format(results.duration, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)
    print('SDE TLS', results.SDE, max(results.power))

    print(results.per_transit_count)
    print(results.transit_depths)
    print(results.depth_mean)
    print(results.snr)
    print(results.depth_mean_odd)
    print(results.depth_mean_even)
    print(results.odd_even_mismatch)
    print(results.empty_transit_count)
    print(results.snr_per_transit)
    print(results.snr_pink_per_transit)
    print(results.transit_count, results.distinct_transit_count, results.empty_transit_count)

    # Final plot
    plt.figure(figsize=(8, 6))
    plt.subplot(221)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.errorbar(
        results.folded_phase,
        (results.folded_y-1)*10**6,
        yerr=results.folded_dy*10**6,
        fmt='o', color='red', alpha=1, markersize=1, zorder=2)
    plt.plot(
        results.model_folded_phase,
        (results.model_folded_model-1)*10**6, color='blue', alpha=0.5, zorder=0)
    plt.xlim(0.4985, 0.5015)
    plt.ylabel(r'Flux (ppm)')
    plt.xlabel('Phase')



    plt.subplot(223)
    plt.axvline(results.period, alpha=0.4, lw=3)
    for n in range(2, 10):
        plt.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        plt.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    #plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.xlim(min(results.periods), max(results.periods))
    max_sde_ld = results.SDE
    scale = 1.1
    max_plot_height = max_sde_ld * scale
    plt.text(360, results.SDE, r'SDE=' + str(round(results.SDE, 1)), ha='right')

    plt.savefig('frontpage.pdf', bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    in_transit = transit_mask(
        t,
        results.period,
        results.duration,
        results.T0)
    plt.scatter(
        t[in_transit],
        y[in_transit],
        color='red',
        s=2,
        zorder=0)
    plt.scatter(
        t[~in_transit],
        y[~in_transit],
        color='blue',
        alpha=0.5,
        s=2,
        zorder=0)
    plt.plot(
        results.model_lightcurve_time,
        results.model_lightcurve_model, alpha=0.5, color='red', zorder=1)
    plt.xlim(min(t), max(t))
    plt.ylim(min(y), max(y))
    plt.xlabel('Time (days)')
    plt.ylabel('Relative flux');
    plt.savefig('model.pdf', bbox_inches='tight')

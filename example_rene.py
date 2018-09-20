import numpy
import scipy
import matplotlib.pyplot as plt
from astropy.io import fits
numpy.set_printoptions(threshold=numpy.nan)
from TransitLeastSquares import get_data

if __name__ == '__main__':
    
    data = numpy.genfromtxt(
        'EPIC_204099713_LC.txt',
        skip_header=1,
        dtype='f8, f8, f8, f8',
        names = ['time', 'y', 'y_filt', 'dy'])

    y_filt = data['y_filt']
    y = data['y']
    #y = data['y']
    t = data['time']
    dy = data['dy']

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 3))
    ax.plot(t, y, "k")
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    plt.savefig('01_allflux.pdf', bbox_inches='tight')

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax = axes[0]
    ax.plot(t, y, "k")
    ax.set_ylabel("relative flux [ppt]")

    ax = axes[1]
    ax.plot(t, y_filt, "k")
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    plt.savefig('02_allflux_detrend.pdf', bbox_inches='tight')

    u, a, b, mass, radius, logg, Teff = get_data(204099713)
    print(u, mass, radius)

    from TransitLeastSquares import TransitLeastSquares, period_grid, fold
    periods = period_grid(
        R_star=radius,
        M_star=mass,
        time_span=(max(t) - min(t)),
        period_min=1.78,  # 10.04
        period_max=1.80,  # 10.05
        oversampling_factor=5)

    # Define grids of transit depths and widths
    depths = numpy.geomspace(0.01, 0.02, 20)  # 50
    durations = numpy.geomspace(0.05, 0.1, 20)  # 50

    model = TransitLeastSquares(t, y_filt, dy)
    results = model.power(periods, durations, depths, limb_darkening=u) # likelihood  # snr

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

    # Plot the in-transit points using 
    in_transit = model.transit_mask(
        t,
        results.best_period,
        results.transit_duration_in_days,
        results.best_T0)
    plt.figure(figsize=(4.5, 4.5 / 1.5))
    plt.plot(t[in_transit], y_filt[in_transit], '.r', ms=3, zorder=0)
    plt.plot(t[~in_transit], y_filt[~in_transit], '.b', ms=3, zorder=0)
    plt.plot(numpy.linspace(min(t), max(t), numpy.size(results.model_flux)), 
        results.model_flux, alpha=0.5, color='red', zorder=1)
    #plt.ylim(0.998, 1.00025)
    plt.xlabel('Time (days)')
    plt.ylabel('Relative flux (ppm)')
    #plt.show()
    plt.savefig('fig_test_intransit.pdf', bbox_inches='tight')

    # Folded model
    phases = fold(t, results.best_period, T0=results.best_T0+ results.best_period/2)
    #phases = fold(t, 1.796, T0=0)
    sort_index = numpy.argsort(phases)
    phases = phases[sort_index]
    flux = y_filt[sort_index]
    plt.figure(figsize=(4.5, 4.5 / 1.5))
    plt.scatter(phases, flux, color='blue', s=10, alpha=0.5, zorder=2)
    plt.plot(numpy.linspace(0, 1, numpy.size(results.folded_model)), 
        results.folded_model, color='red', zorder=3)
    plt.ylim(0.98, 1.01)
    plt.xlim(0.4, 0.6)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux (ppm)')
    plt.savefig('fig_test_fold.pdf', bbox_inches='tight')

    plt.figure(figsize=(4.5, 4.5 / 1.5))
    plt.errorbar(results.transit_times, results.transit_depths, yerr=results.transit_stds, fmt='o')
    plt.plot((min(t) - results.best_period, max(t) + results.best_period), (1, 1), color='black', linewidth=1)
    plt.plot((min(t) - results.best_period, max(t) + results.best_period), \
        (results.total_depth, results.total_depth), color='black', linewidth=1, linestyle='dashed')
    plt.xlim(min(t) - results.best_period, max(t) + results.best_period)
    text = 'Mean transit depth: ' + format(results.total_depth, '.4f') + ', SNR: ' + format(results.total_snr, '.2f')
    plt.text(min(t), 1.005, text)
    plt.xlabel('Time (days)')
    plt.ylabel('Transit depth')
    plt.savefig('fig_depths.pdf', bbox_inches='tight')

    print(results.total_depth, results.total_std, results.total_snr)

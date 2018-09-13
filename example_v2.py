import numpy
import scipy
import matplotlib.pyplot as plt
from astropy.io import fits
numpy.set_printoptions(threshold=numpy.nan)


if __name__ == '__main__':
    file = 'hlsp_everest_k2_llc_201367065-c01_kepler_v2.0_lc.fits'
    url = "https://archive.stsci.edu/hlsps/everest/v2/c01/201300000/67065/"\
        "hlsp_everest_k2_llc_201367065-c01_kepler_v2.0_lc.fits"
    with fits.open(url) as hdus:
        data = hdus[1].data
        t = data["TIME"]
        y = data["FLUX"]
        q = data["QUALITY"]

    # This is from the EVEREST source. These are the flagged data points
    # that should be removed. Ref: https://github.com/rodluger/everest
    m = numpy.isfinite(t) & numpy.isfinite(y)
    for b in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17]:
        m &= (q & (2 ** (b - 1))) == 0

    t = numpy.ascontiguousarray(t[m], dtype=numpy.float64)
    y = numpy.ascontiguousarray(y[m], dtype=numpy.float64)
    y = y / numpy.median(y)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 3))
    ax.plot(t, y, "k")
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    plt.savefig('01_allflux.pdf', bbox_inches='tight')

    from scipy.signal import medfilt
    trend = medfilt(y, 45)
    y_filt = y - trend + 1
    dy = numpy.full(numpy.size(y_filt), numpy.std(y_filt))  #numpy.std(y_filt))  # 

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax = axes[0]
    ax.plot(t, y, "k")
    ax.plot(t, trend)
    ax.set_ylabel("relative flux [ppt]")

    ax = axes[1]
    ax.plot(t, y_filt, "k")
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    plt.savefig('02_allflux_detrend.pdf', bbox_inches='tight')

    from TransitLeastSquares import TransitLeastSquares, period_grid, fold
    periods = period_grid(
        R_star=1,
        M_star=1,
        time_span=(max(t) - min(t)),
        period_min=9,
        period_max=11,
        oversampling_factor=2)

    # Define grids of transit depths and widths
    depths = numpy.geomspace(50*10**-6, 0.01, 50)  # 50
    durations = numpy.geomspace(1.01/numpy.size(t), 0.02, 50)  # 50
    print(durations)

    model = TransitLeastSquares(t, y_filt, dy)
    results = model.power(periods, durations, depths, limb_darkening=0.5, 
        objective='likelihood') # likelihood  # snr
    #results = model.autopower()

    print('Period', format(results.best_period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
            ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.best_depth, '.5f'))
    print('Best duration (fractional period)', format(results.best_duration, '.5f'))
    print('Best duration (days)', format(results.transit_duration_in_days, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)

    print(results.power)

    # Test statistic
    plt.figure(figsize=(4.5, 4.5 / 1.5))
    ax = plt.gca()
    ax.axvline(results.best_period, alpha=0.4, lw=3)
    plt.xlim(numpy.min(periods), numpy.max(periods))
    for n in range(2, 10):
        ax.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.ylabel(r'$1 / (\chi^2_{\rm red}) - 1$')
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
    plt.ylim(0.998, 1.00025)
    plt.xlabel('Time (days)')
    plt.ylabel('Relative flux (ppm)')
    plt.savefig('fig_test_intransit.pdf', bbox_inches='tight')

    # Folded model
    phases = fold(t, results.best_period, T0=results.best_T0+ results.best_period/2)
    sort_index = numpy.argsort(phases)
    phases = phases[sort_index]
    flux = y_filt[sort_index]
    plt.figure(figsize=(4.5, 4.5 / 1.5))
    plt.scatter(phases, flux, color='blue', s=10, alpha=0.5, zorder=2)
    plt.plot(numpy.linspace(0, 1, numpy.size(results.folded_model)), 
        results.folded_model, color='red', zorder=1)
    plt.ylim(0.998, 1.00025)
    #plt.xlim(0.48, 0.52)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux (ppm)')
    plt.savefig('fig_test_fold.pdf', bbox_inches='tight')

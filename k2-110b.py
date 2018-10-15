import numpy
import scipy
import everest
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.stats import BoxLeastSquares
import time

numpy.set_printoptions(threshold=numpy.nan)

# Disable logging from everest
import logging
# set DEBUG for everything
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.WARNING)




if __name__ == '__main__':    
    star = everest.Everest(212521166)  # 201367065 201635569
    t = star.time
    y = star.fcor
    t = numpy.delete(t, star.badmask)
    y = numpy.delete(y, star.badmask)
    t = t[~numpy.isnan(y)]
    y = y[~numpy.isnan(y)]
    t = numpy.array(t, dtype='float32')
    y = numpy.array(y, dtype='float32')
    y = y / numpy.median(y)
    y = sigma_clip(y, sigma_upper=3, sigma_lower=float('inf'))

    from scipy.signal import medfilt
    trend = medfilt(y, 45)
    y_filt = y - trend + 1
    dy = numpy.full(numpy.size(y_filt), numpy.std(y_filt))  #numpy.std(y_filt))  # 

    # Noise estimate
    print('std of filtered data (ppm)', numpy.std(y_filt)*10**6)
    # Optimistic when taking the middle half in flux

    lower, upper = numpy.percentile(y_filt, [10, 90])
    selection = numpy.where(numpy.logical_and(y_filt > lower, y_filt < upper))
    y_filt_middle = y_filt[selection]
    #y_filt_middle = outliers_iqr(y_filt)
    print('std of robust filtered data (ppm)', numpy.std(y_filt_middle)*10**6)

    durations_in_samples = numpy.linspace(1, len(y_filt_middle)*0.12, 10)
    snr_desired = 5
    for dur in durations_in_samples:
        print(dur, ((numpy.std(y_filt_middle)*10**6) / numpy.sqrt(dur))*5)

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

    from TransitLeastSquares_v40 import TransitLeastSquares, period_grid, fold,\
        get_duration_grid, running_median
    """
    periods = period_grid(
        R_star=0.7,
        M_star=0.752,
        time_span=(max(t) - min(t)),
        period_min=12,
        period_max=18,
        oversampling_factor=3)

    print('len(periods)', len(periods))
    """

    # Define grids of transit depths and widths
    #depths = numpy.geomspace(500*10**-6, 5000*10**-6, 50)  # 50
    #durations = numpy.geomspace(0.001, 0.005, 50)  # 50
    #durations = numpy.geomspace(5.01/numpy.size(t), 0.01, 50)  # 50
    #print(durations)
    #durations = get_duration_grid(periods, log_step=1.05)
    #print(len(durations), 'durations from', min(durations), 'to', max(durations))
    #depths = get_depth_grid(y, shallowest=100*10**-6, log_step=1.1)
    #depths = numpy.linspace(50*10**-6, 0.01, 50)
    #print('depths', depths)
    #print(len(depths), 'depths from', min(depths), 'to', max(depths))

    #from astropy.stats import BoxLeastSquares

    t1 = time.perf_counter()
    model = TransitLeastSquares(t, y_filt, dy)
    results = model.power(
        transit_depth_min=50*10**-6,
        b=0.0,
        R_star=0.7,
        M_star=0.752,
        period_min=0,  # 13.87108
        period_max=99,  # 14.1
        n_transits_min=2,
        R_star_min=0.5,
        R_star_max=1.5,
        M_star_min=0.5,
        M_star_max=1.5,
        oversampling_factor=3,
        limb_dark='linear',
        u=[0.5]
                )

    t2 = time.perf_counter()
    print('Time TLS', t2-t1)
    #results = model.autopower()
    #durations = numpy.linspace(0.02, 0.1, 10)
    #model = BoxLeastSquares(t, y_filt)
    #results = model.autopower(durations, frequency_factor=5.0)
    #print(results)
    
    print('Period', format(results.best_period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
            ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.best_depth, '.5f'))
    print('Best duration (fractional period)', format(results.best_duration, '.5f'))
    print('Best duration (days)', format(results.transit_duration_in_days, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)
    
    #print(min(results.chi2red))

    # Test statistic
    plt.rc('font',  family='serif', serif='Computer Modern Roman')
    plt.rc('text', usetex=True)
    plt.figure(figsize=(3.75, 3.75 / 1.5))
    ax = plt.gca()
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    #ax.axvline(results.best_period, alpha=0.4, lw=3)
    #plt.xlim(numpy.min(periods), numpy.max(periods))
    #plt.xlim(0, 40)
    #plt.ylim(-1, 60)
    
    #for n in range(2, 10):
    #    ax.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
    #    ax.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    
    #plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    #ax.text(15, 50, round(max(results.cleaned_SDE_power),1))
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')

    plt.plot(results.periods, results.cleaned_SDE_power, color='black', lw=0.5)
    print('min(results.chi2)', min(results.chi2))
    print('max(results.chi2)', max(results.chi2))
    print('Cleaned SDE', results.cleaned_SDE)

    plt.savefig('k2-110b-stat-tls.pdf', bbox_inches='tight')

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
    plt.ylim(0.998, 1.001)
    plt.xlabel('Time (days)')
    plt.ylabel('Relative flux (ppm)')
    plt.savefig('fig_test_intransit.pdf', bbox_inches='tight')

    # Folded model
    phases = fold(t, results.best_period, T0=results.best_T0+ results.best_period/2)
    sort_index = numpy.argsort(phases)
    phases = phases[sort_index]
    flux = y_filt[sort_index]
    flux = (flux - 1) * 10**6
    model_flux = (results.folded_model - 1) * 10**6
    plt.rc('font',  family='serif', serif='Computer Modern Roman')
    plt.rc('text', usetex=True)
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    plt.figure(figsize=(3.75, 3.75 / 1.5))
    plt.scatter(phases, flux, color='blue', s=10, alpha=0.5, zorder=2)
    plt.plot(numpy.linspace(0, 1, numpy.size(results.folded_model)), 
        model_flux, color='red', zorder=1)
    #plt.ylim(-1600, 250)
    plt.xlim(0.485, 0.515)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux (ppm)')
    plt.savefig('k2-110b-fold-tls.pdf', bbox_inches='tight')


    """
    
    import time
    #durations = numpy.linspace(0.000406, 0.4, 48)
    #durations = numpy.linspace(0.020833333333333, 0.4, 48)
    t1 = time.perf_counter()
    #model = BoxLeastSquares(t, y)
    #results_bls = model.power(periods, durations)
    durations = numpy.linspace(0.02, 0.4, 48)
    model = BoxLeastSquares(t, y_filt)
    results_bls = model.autopower(durations, frequency_factor=5.0)
    t2 = time.perf_counter()
    print('Time BLS', t2-t1)
    chi2 = 1/results_bls.power
    SR = min(chi2) / chi2
    SDE = (1 - numpy.mean(SR)) / numpy.std(SR)
    SDE_power = SR - numpy.min(SR)      # shift down to touch 0
    scale = SDE / numpy.max(SDE_power)  # scale factor to touch max=SDE
    SDE_power = SDE_power * scale
    print('SDE BLS', SDE, max(SDE_power))

    # Test statistic
    plt.rc('font',  family='serif', serif='Computer Modern Roman')
    plt.rc('text', usetex=True)
    plt.figure(figsize=(3.75, 3.75 / 1.5))
    ax = plt.gca()
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.axvline(results.best_period, alpha=0.4, lw=3)
    #plt.xlim(numpy.min(periods), numpy.max(periods))
    plt.xlim(0, 40)
    plt.ylim(-1, 60)
    
    for n in range(2, 10):
        ax.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")

    plt.plot(results_bls.period, SDE_power, color='black', lw=0.5)
    #ax.text(15, 50, round(max(results.SDE_power),1))
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    
    plt.savefig('k2-110b-stat-bls.pdf', bbox_inches='tight')
    """

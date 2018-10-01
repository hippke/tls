import numpy
#import math
#import scipy
import matplotlib.pyplot as plt
import batman
#from TransitLeastSquares_cuda_multi_ootr2 import TransitLeastSquares, period_grid, fold
from TransitLeastSquares import TransitLeastSquares, period_grid, fold,\
    get_duration_grid, get_depth_grid

#from TransitLeastSquares import TransitLeastSquares, period_grid, fold
#from astropy.stats import BoxLeastSquares

#from numba import cuda, autojit, float32

if __name__ == "__main__":

    # Create test data
    start = 0
    days = 365.25 * 2.77  # change this and it breaks
    samples_per_day = 12  # 48
    samples = int(days * samples_per_day) # 48
    t = numpy.linspace(start, start + days, samples)
    #print(t)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = start + 5  #5 # time of inferior conjunction; first transit is X days after start
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
    ppm = 1
    stdev = 10**-6 * ppm
    noise = numpy.random.normal(0, stdev, int(samples))
    y = original_flux + noise
    print('Measured noise (std) is', format(numpy.std(noise), '.8f'))
    if numpy.std(noise) == 0:
        dy = numpy.full_like(y, 1)
    else:
        dy = numpy.full_like(y, stdev)#numpy.std(noise))


    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 3))
    ax.plot(t, y, "k")
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("time [days]")
    ax.set_ylabel("relative flux [ppt]")
    plt.savefig('dfm_01_allflux.pdf', bbox_inches='tight')
    #plt.show()

    y_filt = y
    t = t #- min(t) # + 100
    print('Samples', numpy.size(y), 'duration', max(t)-min(t))
    #print()
    
    periods = period_grid(
        R_star=1,  # R_sun
        M_star=1,  # M_sun
        time_span=(max(t) - min(t)),  # days
        period_min=350,
        period_max=400,
        oversampling_factor=3)
    #periods = numpy.linspace(365.2, 365.3, 100)


    # Define grids of transit depths and widths
    #depths = numpy.geomspace(1*10**-6, 120*10**-6, 50)  # 50
    #durations = numpy.geomspace(0.0015, 0.0020, 10)  # 50

    durations = get_duration_grid(periods, log_step=1.1)
    print(len(durations), 'durations from', min(durations), 'to', max(durations))
    depths = get_depth_grid(y, log_step=1.1)
    #depths = numpy.linspace(50*10**-6, 0.01, 50)
    #print('depths', depths)
    print(len(depths), 'depths from', min(depths), 'to', max(depths))

    model = TransitLeastSquares(t, y_filt, dy)
    results = model.power(periods, durations, depths, limb_darkening=0.5)
    print('chi2min,max', min(results.chi2), max(results.chi2))
    print('SDE TLS', max(results.chi2))
    print(results.transit_times)
    print(results.transit_snrs)

    #model = TransitLeastSquares(t, y_filt, dy)
    #results = model.power(periods, durations, depths, limb_darkening=0.0)
    #print('SDE BLS', max(results.SDE_power))

    #from astropy.io import fits
    #from astropy import units as u
    #periods = numpy.linspace(200, 500, 10000) #* u.day
    #import time
    #t1 = time.perf_counter()
    #bls_model = BoxLeastSquares(t, y_filt)
    #durations = numpy.linspace(0.05, 0.2, 10) #* u.day
    #results_bls = bls_model.power(periods, numpy.linspace(0.1, 0.7, 20))
    #t2 = time.perf_counter()
    #print('BLS time', t2-t1)
    #results_bls = bls_model.autopower(durations, frequency_factor=3.0)

    """
    print('Period', format(results.best_period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
            ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.best_depth, '.5f'))
    print('Best duration (fractional period)', format(results.best_duration, '.5f'))
    print('Best duration (days)', format(results.transit_duration_in_days, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)
    
    plt.rc('font',  family='serif', serif='Computer Modern Roman')
    plt.rc('text', usetex=True)
    plt.figure(figsize=(3.75, 3.75 / 1.5))
    ax = plt.gca()
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    #ax.axvline(results.best_period, alpha=0.4, lw=3)
    #plt.xlim(numpy.min(periods), numpy.max(periods))
    #plt.xlim(0, 40)
    #plt.ylim(-1, 65)
    #for n in range(2, 10):
    #    ax.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
    #    ax.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    #plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    plt.plot(results_bls.period, results_bls.power, "k", lw=0.5)
    #ax.text(15, 60, round(max(results.SDE_power),1))
    plt.ylabel(r'BLS Teststatistik')
    plt.xlabel('Period (days)')
    #plt.show()

    plt.savefig('BLS_T0_180d.pdf', bbox_inches='tight')
    """

    #plt.rc('font',  family='serif', serif='Computer Modern Roman')
    #plt.rc('text', usetex=True)
    """
    plt.figure(figsize=(3.75, 3.75 / 1.5))
    ax = plt.gca()
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    #ax.axvline(results.best_period, alpha=0.4, lw=3)
    #plt.xlim(numpy.min(periods), numpy.max(periods))
    #plt.xlim(0, 40)
    #plt.ylim(-1, 65)
    #for n in range(2, 10):
    #    ax.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
    #    ax.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.chi2red, color='black', lw=0.5)
    #plt.plot(results_bls.period, results_bls.power, "k", lw=0.5)
    #ax.text(15, 60, round(max(results.SDE_power),1))
    plt.ylabel(r'$\chi^2_{\rm red}$')
    #plt.ylabel(r'$\chi^2$')
    plt.xlabel('Period (days)')
    #plt.show()
    plt.ylim(1, max(results.chi2red)*1.1)
    plt.savefig('test_stat_tls_chi2red_ld0.pdf', bbox_inches='tight')
    print(min(results.SDE_power), max(results.chi2red))
    """

    #plt.rc('font',  family='serif', serif='Computer Modern Roman')
    #plt.rc('text', usetex=True)
    """
    plt.figure(figsize=(3.75, 3.75 / 1.5))
    ax = plt.gca()
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.axvline(results.best_period, alpha=0.4, lw=3)
    plt.xlim(numpy.min(periods), numpy.max(periods))
    #plt.xlim(0, 40)
    plt.ylim(0, max(results.SDE_power)*1.1)
    for n in range(2, 10):
        ax.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.SDE_power, color='black', lw=0.5)
    #plt.plot(results_bls.period, results_bls.power, "k", lw=0.5)
    ax.text(results.best_period, results.SDE, round(max(results.SDE_power),1))
    #lt.ylabel(r'$\chi^2_{\rm red}$')
    plt.ylabel(r'S')
    plt.xlabel('Period (days)')
    #plt.show()
    plt.savefig('test_stat_tls_SDE_power.pdf', bbox_inches='tight')
    print(min(results.SDE_power), max(results.chi2red))
    """


    plt.figure(figsize=(3.75, 3.75 / 1.5))
    ax = plt.gca()
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    #ax.axvline(results.best_period, alpha=0.4, lw=3)
    plt.xlim(numpy.min(periods), numpy.max(periods))
    #plt.xlim(0, 40)
    plt.ylim(0, max(results.SR)*1.1)
    #for n in range(2, 10):
    #    ax.axvline(n*results.best_period, alpha=0.4, lw=1, linestyle="dashed")
    #    ax.axvline(results.best_period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.plot(results.periods, results.SR, color='black', lw=0.5)
    #plt.plot(results_bls.period, results_bls.power, "k", lw=0.5)
    #ax.text(results.best_period, results.SR, round(max(results.SR),1))
    #lt.ylabel(r'$\chi^2_{\rm red}$')
    plt.ylabel(r'SR')
    plt.xlabel('Period (days)')
    #plt.show()
    plt.savefig('test_stat_tls_SR.pdf', bbox_inches='tight')

    """
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
    plt.ylim(0.9998, 1.0001)
    plt.xlabel('Time (days)')
    plt.ylabel('Relative flux (ppm)')
    #plt.show()
    plt.savefig('test_intransit.pdf', bbox_inches='tight')
    """
    # Folded model
    phases = fold(t, results.best_period, T0=results.best_T0+ results.best_period/2)
    sort_index = numpy.argsort(phases)
    phases = phases[sort_index]
    flux = y_filt[sort_index]
    #flux = (flux - 1) * 10**6
    model_flux = (results.folded_model)# - 1) * 10**6
    #plt.rc('font',  family='serif', serif='Computer Modern Roman')
    #plt.rc('text', usetex=True)
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    plt.figure(figsize=(3.75, 3.75 / 1.5))
    plt.scatter(phases, flux, color='blue', s=10, alpha=0.5, zorder=2)
    plt.plot(numpy.linspace(0, 1, numpy.size(results.folded_model)), 
        model_flux, color='red', zorder=3)
    plt.ylim(0.9998, 1.0001)
    plt.xlim(0.498, 0.502)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    #plt.show()
    plt.savefig('test_fold.pdf', bbox_inches='tight')

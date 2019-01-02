import numpy
import batman
import matplotlib.pyplot as plt
from transitleastsquares import transitleastsquares, resample

if __name__ == '__main__':
    numpy.random.seed(seed=0)  # reproducibility 

    # Create test data
    time_start = 3.14
    data_duration = 200
    samples_per_day = 48  # 48
    samples = int(data_duration * samples_per_day) # 48
    time = numpy.linspace(time_start, time_start + data_duration, samples)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = time_start  # time of inferior conjunction; first transit is X days after start
    ma.per = 10.123  # orbital period
    ma.rp = 6371 / 696342  # 6371 planet radius (in units of stellar radii)
    ma.a = 19  # semi-major axis (in units of stellar radii)
    ma.inc = 90  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.4, 0.4]  # limb darkening coefficients
    ma.limb_dark = "quadratic"  # limb darkening model
    m = batman.TransitModel(ma, time)  # initializes model
    synthetic_signal = m.light_curve(ma)  # calculates light curve

    # Create noise and merge with flux
    ppm = 50  # Noise level in parts per million
    noise = numpy.random.normal(0, 10**-6 * ppm, int(samples))
    flux = synthetic_signal + noise

    # Plot raw data
    plt.figure()
    ax = plt.gca()
    ax.plot(time, flux, "k")
    ax.set_ylabel("Flux")
    ax.set_xlabel("Time (s)")
    plt.savefig('synt_fig1.pdf', bbox_inches='tight')

    # Run TLS
    model = transitleastsquares(time, flux)
    results = model.power(
        R_star=1,
        M_star=1,
        R_star_min=0.8,
        R_star_max=1.2,
        M_star_min=0.8,
        M_star_max=1.2,
        oversampling_factor=3,
        period_min=5,
        period_max=20)

    # Print statistics
    print('Period', format(results.period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
            ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.depth, '.5f'))
    print('Best duration (days)', format(results.duration, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)

    # Make figure: SDE power spectrum
    plt.figure()
    ax = plt.gca()
    ax.axvline(results.period, alpha=0.4, lw=3)
    plt.xlim(numpy.min(results.periods), numpy.max(results.periods))
    for n in range(2, 10):
        ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))
    plt.savefig('synt_fig2.pdf', bbox_inches='tight')

    # Make figure: Phase-folded data with model
    plt.figure()
    plt.plot(results.model_folded_phase, results.model_folded_model, color='red')
    plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
    plt.xlim(0.48, 0.52)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    plt.savefig('synt_fig3.pdf', bbox_inches='tight')

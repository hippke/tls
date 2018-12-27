import numpy
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
from TransitLeastSquares_new import TransitLeastSquares, transit_mask
#from TransitLeastSquares_refactored import TransitLeastSquares, transit_mask


if __name__ == '__main__':

    # lc_star000_jitterairbus.txt
    # lc_star000.txt
    data = numpy.genfromtxt("lc_star000_jitterairbus.txt")
    time = data[:,0]
    flux = data[:,1]
    SECONDS_PER_DAY = 86400
    time = time / SECONDS_PER_DAY
    flux = flux / numpy.median(flux)

    #trend = scipy.signal.medfilt(flux, 1501)
    #y_filt = flux / trend


    # Bin from 25s cadence to 600s cadence to reduce TLS runtime
    input_cadence = 25  # s
    output_cadence = 600  # s
    f = scipy.interpolate.interp1d(time, flux)
    samples = int(len(data) / (output_cadence / input_cadence))
    tnew = numpy.linspace(min(time), max(time), samples)
    ynew = f(tnew)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax = axes[0]
    ax.plot(time, flux, "k")
    #ax.plot(time, trend)
    ax.set_ylabel("Flux")
    ax.set_xlabel("Time (s)")
    ax = axes[1]
    ax.plot(tnew, ynew, "k")
    #ax.set_xlim(min(time), max(time))
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Binned flux");
    plt.savefig('plato_fig1.pdf', bbox_inches='tight')


    model = TransitLeastSquares(tnew, ynew)
    #model = TransitLeastSquares(time, flux)
    results = model.power(
        R_star=1.45,
        R_star_min=1.3,
        R_star_max=1.6,
        M_star=1.45,
        M_star_min=1.3,
        M_star_max=1.6,
        transit_depth_min=1000*10**-6)
    print(results.period)

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


    plt.figure()
    ax = plt.gca()
    ax.axvline(results.period, alpha=0.4, lw=3)
    plt.xlim(numpy.min(results.periods), numpy.max(results.periods))
    for n in range(2, 10):
        ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.plot(results.periods, results.chi2, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))
    plt.ylim(0, 2000)
    plt.savefig('plato1.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(
        results.model_folded_phase,
        results.model_folded_model,
        color='red')
    plt.scatter(
        results.folded_phase,
        results.folded_y,
        color='blue',
        s=10,
        alpha=0.5,
        zorder=2)
    plt.xlim(0.45, 0.55)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    plt.savefig('plato2.pdf', bbox_inches='tight')

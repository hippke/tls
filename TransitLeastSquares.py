#                Optimized algorithm to search for transits 
#                of small extrasolar planets
#                                                                            /
#       ,        AUTHORS                                                   O/
#    \  :  /     Michael Hippke (1) [michael@hippke.org]                /\/|
# `. __/ \__ .'  Rene' Heller (2) [heller@mps.mpg.de]                       |
# _ _\     /_ _  _________________________________________________________/ \_
#    /_   _\                                                             
#  .'  \ /  `.   (1) Sonneberg Observatory, Sternwartestr. 32, Sonneberg
#    /  :  \     (2) Max Planck Institute for Solar System Research,
#       '            Justus-von-Liebig-Weg 3, 37077 G\"ottingen, Germany


import numpy
import numba
import scipy
import sys
import multiprocessing
import math

# batman-package for transit light curves
# https://www.cfa.harvard.edu/~lkreidberg/batman/
import batman
import scipy.interpolate
from tqdm import tqdm
from functools import partial
from numpy import pi, sqrt, arccos, degrees


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def out_of_transit_residuals(data, width_signal, dy):
    width_data = len(data)
    residuals = numpy.zeros(width_data - width_signal + 1)
    for i in numba.prange(width_data - width_signal + 1):
        value = 0
        start_transit = i
        end_transit = i + width_signal
        for j in numba.prange(width_data - width_signal + 1):
            if j < start_transit or j > end_transit:
                # dy has already been inverted and squared (for speed)
                value = value + (1 - data[j])**2 * dy[j]
        residuals[i] = value
    return residuals


# As periods are searched in parallel, the double parallel option for numba
# results in a speed penalty (factor two worse), so choose parallel=False here
@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def all_transit_residuals(data, signal, dy, out_of_transit_residuals):
    width_signal = signal.shape[0]
    width_data = data.shape[0]
    chi2 = numpy.zeros(width_data - width_signal + 1)
    for i in numba.prange(width_data - width_signal + 1):
        value = 0
        for j in range(width_signal):
            # dy has already been inverted and squared (for speed)
            value = value + ((data[i+j]-signal[j])**2) * dy[i+j]
        chi2[i] = value + out_of_transit_residuals[i]
    return chi2


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def ll_out_of_transit_residuals(data, width_signal, dy):
    # simplified with normalization of data to 1:
    # b = sum( (signal_out - 1)**2 / dy_out**2)
    # bb = -1./2 * b
    width_data = len(data)
    bb = numpy.zeros(width_data - width_signal + 1)
    for i in numba.prange(width_data - width_signal + 1):
        b1 = 0
        b2 = 0
        dys = 0
        start_transit = i
        end_transit = i + width_signal
        for j in numba.prange(width_data - width_signal + 1):
            if j < start_transit or j > end_transit:
                # dy has already been inverted and squared (for speed)
                b1 = b1 + data[j] * dy[j]
                b2 = b2 + dy[j]
            dys = dys + dy[j]
        b = b1 / b2
        signal_points = width_data - width_signal + 1
        bb[i] = -0.5 * (signal_points - b)**2 * dys

        #-1./2 * sum( (signal_out - b)**2 / dy_out**2)

    return bb


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def ll_all_transit_residuals(data, signal, dy, out_of_transit_residuals):
    width_signal = signal.shape[0]
    width_data = data.shape[0]
    a3s = numpy.zeros(width_data - width_signal + 1)
    a4s = numpy.zeros(width_data - width_signal + 1)

    for i in numba.prange(width_data - width_signal + 1):
        a3 = 0
        for j in range(width_signal):
            # dy has already been inverted and squared (for speed)
            a3 = a3 + data[i+j] * dy[i+j]
            #a3 = a3 + data[i+j]# * dy[i+j]
        a3s[i] = a3

    for k in numba.prange(width_data - width_signal + 1):
        a4 = 0
        for l in range(width_signal):
            a4 = a4 + ((signal[l] - a3s[k])**2 * dy[k+l])
            #a4 = a4 + ((signal[l] - a3s[k])**2)# * dy[k+l])
        a4s[k] = a4
    a5s = -1./2 * a4s
    ll = out_of_transit_residuals + a5s
    return ll



"""
# As periods are searched in parallel, the double parallel option for numba
# results in a speed penalty (factor two worse), so choose parallel=False here
@numba.njit(fastmath=True, parallel=False, cache=True)  
def numba_chi2(data, signal, dy, transit_weight):
    Takes numpy arrays of data and signal, returns squared residuals for each possible shift
    See https://stackoverflow.com/questions/52001974/fast-iteration-over-numpy-array-for-squared-residuals
    
    # dy, weight: inverted beforehand (1/dy) to use multiplication for speed!
    chi2 = numpy.empty(data.shape[0] + 1 - signal.shape[0], dtype=data.dtype)
    width_signal = signal.shape[0]
    for i in numba.prange(data.shape[0] - signal.shape[0] + 1):
        value = 0
        for j in range(signal.shape[0]):
            value = value + ((data[i+j]-signal[j])**2) #/ dy[i+j]**2
        chi2[i] = value
    return chi2 * transit_weight
"""






@numba.njit(fastmath=True, parallel=False, cache=True)  
def fold(time, period, T0):#=0.0):
    return (time - T0) / period - numpy.floor((time - T0) / period)


def period_grid(R_star, M_star, time_span, period_min=0, period_max=sys.maxsize, 
        oversampling_factor=2):
    """Returns array of optimal sampling periods for transit search in light curves
       Following Ofir (2014, A&A, 561, A138)"""

    # astrophysical constants
    G = 6.673e-11  # [m^3 / kg / s^2]
    day   = 86400.0  # [s]

    # Values
    R_sun = 695508000  # [m]
    M_sun = 1.989*10**30  # [kg]
    R_star = R_star * R_sun
    M_star = M_sun * M_star
    time_span = time_span * day  # seconds

    # boundary conditions
    f_min = 2./time_span
    f_max = 1./(2*pi) * sqrt( G*M_star /(3*R_star)**3 )

    # optimal frequency sampling, Equations (5), (6), (7)
    A = ( 2*pi )**(2./3) / pi * R_star / (G*M_star)**(1./3) / \
        (time_span * oversampling_factor)
    C = f_min**(1./3) - A/3.
    N_opt = (f_max**(1./3) - f_min**(1./3) + A/3)*3/A

    X = numpy.arange(N_opt)+1
    f_x = ( A/3*X + C )**3
    P_x = 1/f_x

    # Cut to given (optional) selection of periods
    periods = P_x / day
    selected_index = numpy.where(
        numpy.logical_and(periods>period_min, periods<=period_max))

    if numpy.size(periods[selected_index]) == 0:
        raise ValueError('Empty period array')

    return periods[selected_index]  # periods in [days]


class TransitLeastSquares(object):
    """Compute the transit least squares of limb-darkened transit models"""

    def __init__(self, t, y, dy=None):
        self.t, self.y, self.dy = self._validate_inputs(t, y, dy)


    def _validate_inputs(self, t, y, dy):
        """Private method used to check the consistency of the inputs"""

        if numpy.size(t) != numpy.size(y):
            raise ValueError('Arrays (t, y) must be of the same dimensions')

        if t.ndim != 1:
            raise ValueError('Inputs (t, y, dy) must be 1-dimensional')

        if dy is not None:
            if numpy.size(t) != numpy.size(dy):
                raise ValueError('Arrays (t, dy) must be of the same dimensions')

        return t, y, dy


    def _validate_period_and_duration(self, periods, durations, depths):
        """Private method used to check a set of periods, durations and depths"""

        duration = numpy.max(periods) - numpy.min(periods)

        if periods.ndim != 1 or periods.size == 0:
            raise ValueError('period must be 1-dimensional')

        # TBD
        #if numpy.min(periods) > numpy.max(durations):
        #    raise ValueError('Periods must be shorter than the duration')

        return periods, durations, depths

    """
    def _scipy_chi2(self, data, signal, dy, transit_weight):
        sigma = 1  # dy[0]
        width_signal = numpy.size(signal)
        hW = width_signal//2
        l = len(data)-len(signal)+1
        part1 = uniform_filter(data**2,width_signal)[hW:hW+l]*width_signal
        part3 = numpy.convolve(data, signal[::-1],'valid')   
        stat = (part1 + (signal**2).sum(0) - 2*part3) / (width_signal * sigma**2)
        return stat * transit_weight
    """


    def fractional_transit(self, duration, maxwidth, depth, samples, 
            limb_darkening=0.5, impact=0, cached_reference_transit=None):
        """Returns a scaled reference transit with fractional width and depth"""

        reference_time = numpy.linspace(-0.5, 0.5, samples)

        if cached_reference_transit is None:
            reference_flux = self.reference_transit(
                samples=samples, limb_darkening=limb_darkening, impact=impact)
        else:
            reference_flux = cached_reference_transit

        # Interpolate to shorter interval
        f = scipy.interpolate.interp1d(reference_time, reference_flux)
        occupied_samples = int((duration / maxwidth) * samples)
        xnew = numpy.linspace(-0.5, 0.5, occupied_samples)
        ynew = f(xnew)

        # Patch ends with ones ("1")
        missing_samples = samples - occupied_samples
        emtpy_segment = numpy.ones(int(missing_samples / 2))
        result = numpy.append(emtpy_segment, ynew)
        result = numpy.append(result, emtpy_segment)
        if numpy.size(result) < samples:  # If odd number of samples
            result = numpy.append(result, numpy.ones(1))

        # Depth rescaling
        result = 1 - ((1 - result) * depth)  

        return result


    def _impact_to_inclination(self, impact, semimajor_axis):
            """Converts planet impact parameter b = [0..1.x] to inclination [deg]"""
            return degrees(arccos(impact / semimajor_axis))


    def reference_transit(self, samples, limb_darkening=0.5, impact=0):
        """Returns an Earth-like transit of width 1 and depth 1"""

        # Box-shaped transit
        if limb_darkening == 0:
            rescaled = numpy.zeros(samples)

        # Limb-darkened transit
        else:
            # We use a large oversampling and down-sample (bin) from there
            # Reason: "Don’t fit an unbinned model to binned data." 
            # Reference: Kipping, D., "Binning is sinning: morphological light-curve 
            #           distortions due to finite integration time"
            #           MNRAS, Volume 408, Issue 3, pp. 1758-1769
            #           http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1004.3741
            # This is not time-critical as it has to be done only once

            supersample_size = 10000

            f = numpy.ones(supersample_size)
            duration = 1  # transit duration in days. Increase for exotic cases
            t = numpy.linspace(-duration/2, duration/2, supersample_size)
            ma = batman.TransitParams()
            ma.t0 = 0  # time of inferior conjunction
            ma.per = 365.25  # orbital period, use Earth as a reference
            ma.rp = 6371 / 696342  # planet radius (in units of stellar radii)
            ma.a = 217  # semi-major axis (in units of stellar radii)
            # orbital inclination (in degrees)
            ma.inc = self._impact_to_inclination(impact=impact, semimajor_axis=ma.a)  
            ma.ecc = 0  # eccentricity
            ma.w = 90  # longitude of periastron (in degrees)
            ma.u = [limb_darkening]  # limb darkening coefficients
            ma.limb_dark = "linear"  # limb darkening model
            m = batman.TransitModel(ma, t)  # initializes model
            flux = m.light_curve(ma)  # calculates light curve

            # Determine start of transit (first value < 1)
            idx_first = numpy.argmax(flux < 1)
            intransit_flux = flux[idx_first:-idx_first+1]
            intransit_time = t[idx_first:-idx_first+1]

            # Downsample (bin) to target sample size
            f = scipy.interpolate.interp1d(intransit_time, intransit_flux)
            xnew = numpy.linspace(t[idx_first], t[-idx_first-1], samples)
            downsampled_intransit_flux = f(xnew)

            # Rescale to height [0..1]
            rescaled = (numpy.min(
                downsampled_intransit_flux) - downsampled_intransit_flux) / \
                (numpy.min(downsampled_intransit_flux) - 1)

        return rescaled


    def _get_cache(self, durations, depths, maxwidth_in_samples, 
            limb_darkening=0.5, impact=0):
        """Fetches (size(durations)*size(depths)) light curves of length 
        maxwidth_in_samples and returns these LCs in a 2D array, together with 
        their metadata in a separate array."""

        text = 'Creating model cache: ' + \
            str(len(durations)) + ' durations x ' + \
            str(len(depths)) + ' depths = ' + \
            str(len(durations) * len(depths)) + ' curves'

        print(text)
        rows = numpy.size(depths) * numpy.size(durations)
        lc_cache = numpy.ones([rows,maxwidth_in_samples])
        lc_cache_overview = numpy.zeros(
            rows, dtype=[('depth', 'f8'), ('duration', 'f8')])
        cached_reference_transit = self.reference_transit(
            samples=maxwidth_in_samples,
            limb_darkening=limb_darkening,
            impact=impact)
        row = 0
        pbar = tqdm(total=len(depths)*len(durations))
        for depth in depths:
            for duration in durations:
                scaled_transit = self.fractional_transit(
                    duration=duration,
                    maxwidth=numpy.max(durations),
                    depth=depth,
                    samples=maxwidth_in_samples,
                    limb_darkening=limb_darkening,
                    impact=impact,
                    cached_reference_transit=cached_reference_transit)
                lc_cache[row] = scaled_transit
                lc_cache_overview['depth'][row] = depth
                lc_cache_overview['duration'][row] = duration

                # Deeper transit shapes have stronger weight (Kovacz+2002 Section 2)
                # Use pre-calculated weights for speed
                # Use inverse to multiply the weight later (much faster than division)
                #lc_cache_overview['weight'][row] = 1 / numpy.sum(1 - scaled_transit)**2
                row = row + 1
                pbar.update(1)
        pbar.close()
        return lc_cache, lc_cache_overview


    def _search_period(self, period, t, y, dy, lc_cache, lc_cache_overview, objective='snr'):
        """Core routine to search the flux data set 'injected' over all 'periods'"""


        # duration (in samples) of widest transit in lc_cache (axis 0: rows; axis 1: columns)
        maxwidth_in_samples = numpy.shape(lc_cache)[1]

        # Phase fold
        phases = fold(t, period, T0=0)
        sort_index = numpy.argsort(phases)
        phases = phases[sort_index]
        flux = y[sort_index]
        dy = dy[sort_index]

        # faster to multiply than divide
        # SQUARE THESE HERE ALREADY?
        patched_dy = numpy.append(dy, dy[:maxwidth_in_samples])
        inverse_squared_patched_dy = 1 / patched_dy**2

        # Due to phase folding, the signal could start near the end of the data
        # and continue at the beginning. To avoid (slow) rolling, 
        # we patch the beginning again to the end of the data
        patched_data = numpy.append(flux, flux[:maxwidth_in_samples])

        if objective == 'snr':
            ootr = out_of_transit_residuals(
                patched_data, maxwidth_in_samples, inverse_squared_patched_dy)
        elif objective == 'likelihood':
            ootr = ll_out_of_transit_residuals(
                patched_data, maxwidth_in_samples, inverse_squared_patched_dy)
        else:
            ValueError("Unknown objective. Possible values: 'snr' and 'likelihood'")

        print('ootr', ootr)

        # Set "best of" counters to max, in order to find smaller residuals
        smallest_residuals_in_period = float('inf')
        summed_residual_in_rows = float('inf')

        # Iterate over all transit shapes (depths and durations)
        for row in range(len(lc_cache)):
            # This is the current transit (of some width, depth) to test
            scaled_transit = lc_cache[row]

            if objective == 'snr':
                stats = all_transit_residuals(
                    data=patched_data,
                    signal=scaled_transit,
                    dy=inverse_squared_patched_dy,
                    out_of_transit_residuals=ootr)
                best_roll = numpy.argmin(stats)
            else:
                stats = ll_all_transit_residuals(
                    data=patched_data,
                    signal=scaled_transit,
                    dy=inverse_squared_patched_dy,
                    out_of_transit_residuals=ootr)
                #stats = abs(stats)
                #print('stats nans', numpy.sum(numpy.isnan(stats)))
                #print(numpy.min(ll), numpy.max(ll), numpy.mean(ll))
                #stats = 1/stats
                #print('stats', stats)
                best_roll = numpy.argmax(stats)


            current_smallest_residual = stats[best_roll]
            #print('current_smallest_residual', current_smallest_residual)

            # Propagate results to outer loop (best duration, depth)
            if current_smallest_residual < summed_residual_in_rows:
                summed_residual_in_rows = current_smallest_residual
                best_row = row

        # Best values in this period
        if summed_residual_in_rows < smallest_residuals_in_period:
            smallest_residuals_in_period = summed_residual_in_rows
            best_shift = best_roll

        
        return [period, smallest_residuals_in_period, best_shift, best_row]


    def _perform_search(self, periods, t, y, dy, depths, durations, 
            limb_darkening=0.5, impact=0, objective='snr'):
        """Multicore distributor of search to search through all 'periods' """

        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        lc_cache, lc_cache_overview = self._get_cache(
            durations=durations,
            depths=depths,
            maxwidth_in_samples=maxwidth_in_samples,
            limb_darkening=limb_darkening,
            impact=impact)
        
        # Prepare result arrays
        test_statistic_periods = []
        test_statistic_residuals = []
        test_statistic_rolls = []
        test_statistic_rows = []
        
        text = 'Searching ' + str(len(self.y)) + ' data points, ' + \
            str(len(periods)) + ' periods from ' + \
            str(round(min(periods), 3)) + ' to ' + \
            str(round(max(periods), 3)) + ' days, using all ' + \
            str(multiprocessing.cpu_count()) + ' CPU threads'
        print(text)

        p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pbar = tqdm(total=numpy.size(periods), smoothing=0)
        
        params = partial(
            self._search_period,
            t=t,
            y=y,
            dy=dy,
            lc_cache=lc_cache,
            lc_cache_overview=lc_cache_overview,
            objective=objective)
        for data in p.imap_unordered(params, periods):
            test_statistic_periods.append(data[0])
            test_statistic_residuals.append(data[1])
            test_statistic_rolls.append(data[2])
            test_statistic_rows.append(data[3])

            # Update progress bar by 1 period step
            pbar.update(1)
        p.close()
        pbar.close()

        # imap_unordered delivers results in unsorted order ==> sort
        test_statistic_periods = numpy.array(test_statistic_periods)
        test_statistic_residuals = numpy.array(test_statistic_residuals)
        test_statistic_rolls = numpy.array(test_statistic_rolls)
        test_statistic_rows = numpy.array(test_statistic_rows)
        sort_index = numpy.argsort(test_statistic_periods)
        test_statistic_periods = test_statistic_periods[sort_index]
        test_statistic_residuals = test_statistic_residuals[sort_index]
        test_statistic_rolls = test_statistic_rolls[sort_index]
        test_statistic_rows = test_statistic_rows[sort_index]

        return test_statistic_periods, test_statistic_residuals, \
        test_statistic_rolls, test_statistic_rows, lc_cache_overview


    def power(self, periods, durations, depths, limb_darkening=0.5, impact=0, objective='snr'):
        """Compute the periodogram for a set user-defined parameters
        Parameters:
        periods : array of periods where the power should be computed
        durations : array of set of durations to test
        depths : array of set of depths to test
        method : {'scipy', 'numba'}, optional
            The computational method used to compute the periodogram. Both
            yield identical results but may differ in speed and availability,
            depending on the available hardware.
        Returns: BoxLeastSquaresResults"""

        periods, durations, depths = self._validate_period_and_duration(
            periods, durations, depths)

        test_statistic_periods, test_statistic_residuals, test_statistic_rolls, \
            test_statistic_rows, lc_cache_overview = self._perform_search(
            periods,
            self.t,
            self.y,
            self.dy,
            depths,
            durations,
            limb_darkening=limb_darkening,
            impact=impact,
            objective=objective)
        
        # Sort residuals for best

        if objective == 'snr':
            idx_best = numpy.argmin(test_statistic_residuals)
        else:
            idx_best = numpy.argmax(test_statistic_residuals)
        best_power = test_statistic_residuals[idx_best]
        best_period = test_statistic_periods[idx_best]
        best_roll = test_statistic_rolls[idx_best]
        best_row = test_statistic_rows[idx_best]
        best_roll_save = test_statistic_rolls[idx_best]

        best_depth = lc_cache_overview['depth'][best_row]
        best_duration = lc_cache_overview['duration'][best_row]

        # Now we know the best period, width and duration. But T0 was not preserved
        # due to speed optimizations. Thus, iterate over T0s using the given parameters

        # Create all possible T0s from the start of [t] to [t+period] in [samples] steps
        
        # ideal step size: number of samples per period
        duration = numpy.max(self.t) - numpy.min(self.t)
        no_of_periods = duration / best_period
        samples_per_period = numpy.size(self.y) #/ no_of_periods

        T0_array = numpy.linspace(
            start=numpy.min(self.t),
            stop=numpy.min(self.t) + best_period,
            num=samples_per_period)


        # Fold to all T0s so that the transit is expected at phase = 0
        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.t))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        #t0 =time.perf_counter()
        # Make a model transit with the best fit parameters
        signal = self.fractional_transit(
            duration=best_duration,
            maxwidth=max(durations), 
            depth=best_depth,
            samples=maxwidth_in_samples,
            limb_darkening=limb_darkening,
            impact=impact)
        #t1 =time.perf_counter()
        #print(t1-t0)
        lowest_chi2 = sys.maxsize
        best_T0 = 0
        #signal = lc_cache[best_row]
        start_transit = 0.5 - numpy.max(durations) / 2
        print('Finding best T0 for period', format(best_period, '.5f'))
        pbar2 = tqdm(total=numpy.size(T0_array))
        #t1 = time.perf_counter()
        for Tx in T0_array:
            # "fold" is fast (7% of time in this loop) using numba
            phases = fold(time=self.t, period=best_period, T0=Tx + best_period/2)

            # Mergesort is faster than the default quicksort for our data
            # Sorting takes >50% of the time in this loop
            sort_index = numpy.argsort(phases, kind='mergesort')
            phases = phases[sort_index]
            flux = self.y[sort_index]
            dy = self.dy[sort_index]

            # Sorted array is not perfectly linar: phase = 0.5 != numpy.max(phases)/2
            # as there are more points at some phases than others 
            # Thus, we need to find the points around phase 0.5
            points_in_transit_phase = numpy.size(numpy.where(numpy.logical_and(
                phases>=0.5 - numpy.max(durations) / 2,
                phases<=0.5 + numpy.max(durations) / 2)))
            offset = int((maxwidth_in_samples - points_in_transit_phase)/2)# - 1

            # Roll the array so that phase = 0.5 is in the middle
            # Roll is 33% of the time in this loop
            #flux = numpy.roll(flux, offset)
            #dy = numpy.roll(dy, offset)

            # Instead of rolling, shift the id_flux_start by (- offset)
            id_flux_start = numpy.argmax(phases > start_transit)
            id_flux_start = id_flux_start - offset
            data_segment = flux[id_flux_start:id_flux_start+maxwidth_in_samples]
            phase_segment = phases[id_flux_start:id_flux_start+maxwidth_in_samples]
            dy_segment = dy[id_flux_start:id_flux_start+maxwidth_in_samples]

            # dynot inverted!?
            current_chi2 = numpy.sum((data_segment - signal)**2 / dy_segment)  
            pbar2.update(1)
            if current_chi2 < lowest_chi2:
                lowest_chi2 = current_chi2
                best_T0 = Tx

        pbar2.close()
        #t2 = time.perf_counter()
        #print(t2-t1)
        # SDE
        
        print('best_T0 ##########', best_T0)

        # Calculate all mid-transit times
        if best_T0 < min(self.t):
            transit_times = [best_T0 + best_period]
        else:
            transit_times = [best_T0]
        previous_transit_time = transit_times[0]
        transit_number = 0
        while True:
            transit_number = transit_number + 1
            next_transit_time = previous_transit_time + best_period
            if next_transit_time < (min(self.t) + (max(self.t)-min(self.t))):
                transit_times.append(next_transit_time)
                previous_transit_time = next_transit_time
            else:
                break

        # Calculate transit duration in days
        duration_timeseries = (max(self.t) - min(self.t)) / best_period
        epochs = len(transit_times)
        stretch = duration_timeseries / epochs
        transit_duration_in_days = best_duration * stretch * best_period

        if objective == 'snr':
            # reduced chi^2
            test_statistic_residuals = test_statistic_residuals / (len(self.t) - 4)
            # Squash to range 0..1 with peak at 1
            test_statistic_residuals = 1/test_statistic_residuals - 1
            chi2red = test_statistic_residuals
            signal_residue = test_statistic_residuals / numpy.max(test_statistic_residuals)
        else:
            signal_residue = test_statistic_residuals
            chi2red = None

        power = signal_residue

        # Make folded model
        phases = fold(self.t, best_period, T0=best_T0+ best_period/2)
        sort_index = numpy.argsort(phases)
        phases = phases[sort_index]
        flux = self.y[sort_index]

        folded_model = self.fractional_transit(
            duration=(best_duration * maxwidth_in_samples),
            maxwidth=maxwidth_in_samples / stretch,
            depth=best_depth,
            samples=int(len(self.t/epochs)),
            limb_darkening=limb_darkening,
            impact=impact)


        # Full model
        # We oversample the model internally
        internal_samples = 100000

        # Append one more transit after and before end of nominal time series
        # to fully cover beginning and end with out of transit calculations
        earlier_tt = transit_times[0] - best_period
        extended_transit_times = numpy.append(earlier_tt, transit_times)
        next_tt = transit_times[-1] + best_period
        extended_transit_times = numpy.append(extended_transit_times, next_tt)
        full_x_array = numpy.array([])
        full_y_array = numpy.array([])
        rounds = len(extended_transit_times)

        # The model for one period
        y_array = self.fractional_transit(
            duration=(best_duration * maxwidth_in_samples),
            maxwidth=maxwidth_in_samples / stretch,
            depth=best_depth,
            samples=internal_samples,
            limb_darkening=limb_darkening,
            impact=impact)

        # Append all periods
        for i in range(rounds):
            xmin = extended_transit_times[i] - best_period/2
            xmax = extended_transit_times[i] + best_period/2
            x_array = numpy.linspace(xmin, xmax, internal_samples)
            full_x_array = numpy.append(full_x_array, x_array)
            full_y_array = numpy.append(full_y_array, y_array)

        # Cut to output time range and sample down to desired resolution
        f = scipy.interpolate.interp1d(full_x_array, full_y_array)
        xnew = numpy.linspace(min(self.t), max(self.t), len(self.t))
        model_flux = f(xnew)

        SDE = (numpy.max(power) - numpy.mean(power)) / \
            numpy.std(power)

        return TransitLeastSquaresResults(test_statistic_periods, \
            power, test_statistic_rolls, test_statistic_rows, \
            lc_cache_overview, best_period, best_T0, best_row, best_power, \
            SDE, test_statistic_rolls, best_depth, best_duration,\
            transit_times, transit_duration_in_days, maxwidth_in_samples, \
            folded_model, model_flux, chi2red)


    def autopower(self):
        periods = period_grid(
            R_star=1,
            M_star=1,
            time_span=(max(self.t) - min(self.t)),
            oversampling_factor=2)
        depths = numpy.geomspace(50*10**-6, 0.02, 50)
        durations = numpy.geomspace(1.01/numpy.size(self.t), 0.05, 50)
        results = self.power(periods, durations, depths)
        return results


    def transit_mask(self, t, period, duration, transit_time):
        half_period = 0.5 * period
        return numpy.abs((t-transit_time+half_period) % period - half_period) \
            < 0.5*duration


class TransitLeastSquaresResults(dict):
    """The results of a TransitLeastSquares search"""

    def __init__(self, *args):
        super(TransitLeastSquaresResults, self).__init__(zip(
            ("periods", "power", "phase", "rows", "lc_cache_overview", \
                "best_period", "best_T0", "best_row", "best_power", \
                "SDE", "rolls", "best_depth", \
                "best_duration", "transit_times", "transit_duration_in_days", \
                "maxwidth_in_samples", "folded_model", "model_flux", "chi2red"), args))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

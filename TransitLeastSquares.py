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
from numba import cuda, autojit, float32
from numba import njit, jit, vectorize


def T14(R_s, M_s, P):
        """Input:  Stellar radius and mass; planetary period
                   Units: Solar radius and mass; days
           Output: Maximum planetary transit duration T_14max
                   Unit: Fraction of period P"""

        # astrophysical constants
        G = 6.673e-11  # [m^3 / kg / s^2]
        R_sun = 695508000  # [m]
        M_sun = 1.989*10**30  # [kg]

        P = P * 24 * 3600
        R_s = R_sun * R_s
        M_s = M_sun * M_s

        T14max = R_s * ((4*P) / (pi * G * M_s))**(1/3)
        result = T14max / P
        if result > 0.15:
            result = 0.15

        return result


def get_duration_grid(periods, log_step=1.1):
    duration_max = T14(R_s=3.5, M_s=1.0, P=min(periods))
    duration_min = T14(R_s=0.13, M_s=0.1, P=max(periods))
    # Make grid
    durations = [duration_min]
    current_depth = duration_min
    while current_depth* log_step < duration_max:
        current_depth = current_depth * log_step
        durations.append(current_depth)
    durations.append(duration_max)  # Append endpoint

    return durations


def get_depth_grid(y, deepest=None, shallowest=None, log_step=None):
    if deepest is None:
        deepest = 1 - numpy.min(y)
    if shallowest is None:
        shallowest = 10*10**-6  # 10 ppm
    if log_step is None:
        log_step = 1.1
    
    # Make grid
    depths = [shallowest]
    current_depth = shallowest
    while current_depth* log_step < deepest:
        current_depth = current_depth * log_step
        depths.append(current_depth)
    depths.append(deepest)  # Append endpoint

    return depths


def get_catalog_info(EPIC):
    """Takes EPIC ID, returns limb darkening parameters u (linear) and 
        a,b (quadratic), and stellar parameters. Values are pulled for minimum
        absolute deviation between given/catalog Teff and logg. Data are from:
        - K2 Ecliptic Plane Input Catalog, Huber+ 2016, 2016ApJS..224....2H
        - New limb-darkening coefficients, Claret+ 2012, 2013, 
          2012A&A...546A..14C, 2013A&A...552A..16C"""

    star = numpy.genfromtxt(
        'JApJS2242table5.csv',
        skip_header=1,
        delimiter=',',
        dtype='int32, int32, f8, f8, f8',
        names = ['EPIC', 'Teff', 'logg', 'radius', 'mass'])
    ld = numpy.genfromtxt(
        'JAA546A14limb1-4.csv',
        skip_header=1,
        delimiter=',',
        dtype='f8, int32, f8, f8, f8',
        names = ['logg', 'Teff', 'u', 'a', 'b'])

    idx = numpy.where(star['EPIC']==EPIC)
    Teff = star['Teff'][idx]
    logg = star['logg'][idx]
    radius = star['radius'][idx]
    mass = star['mass'][idx]

    # Find nearest Teff and logg
    nearest_Teff = ld['Teff'][(numpy.abs(ld['Teff'] - Teff)).argmin()]
    idx_all_Teffs = numpy.where(ld['Teff']==nearest_Teff)
    relevant_lds = numpy.copy(ld[idx_all_Teffs])
    idx_nearest = numpy.abs(relevant_lds['logg'] - logg).argmin()
    nearest_logg = relevant_lds['logg'][idx_nearest]
    u = relevant_lds['u'][idx_nearest]
    a = relevant_lds['a'][idx_nearest]
    b = relevant_lds['b'][idx_nearest]

    return u, a, b, mass[0], radius[0], logg[0], Teff[0]


"""
@cuda.jit
def itr_cuda(data, dys, signals, ootr, signal_row_sums):
    i = cuda.grid(1)
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    xstride = cuda.gridsize(1)
    a = signals.shape[1]+1
    b = len(data)-signals.shape[1]
    for signal_trial in range(i, signals.shape[0], xstride):  # +1 ?
        # Iterate over all phase shifts
        smallest_residual = 10**10
        for phase_position in range(b):
            in_transit_sum = 0
            for in_transit_point in range(a):
                # Calculate squared sum of residuals at this position
                datapoint = data[phase_position+in_transit_point]
                signal = signals[signal_trial, in_transit_point]
                error = dys[phase_position+in_transit_point]
                residual = (datapoint - signal)**2 * error
                in_transit_sum = in_transit_sum + residual
            phase_sum = in_transit_sum + ootr[phase_position]
            if phase_sum < smallest_residual:
                smallest_residual = phase_sum
        signal_row_sums[signal_trial] = smallest_residual
"""


@cuda.jit
def itr_cuda(data, a, s, c, result):
    i = cuda.grid(1)
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    xstride = cuda.gridsize(1)
    for j in range(i, s.shape[0], xstride):
        smallest = 10**10
        for k in range(len(data)-s.shape[1]):
            in_sum = 0
            for l in range(s.shape[1]+1):
                in_sum += (data[k+l] - s[j,l])**2 * a[k+l]
            if in_sum + c[k] < smallest:
                smallest = in_sum + c[k]
        result[j] = smallest


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def ootr_efficient(data, width_signal, dy):
    chi2 = numpy.zeros(len(data) - width_signal + 1)
    fullsum = numpy.sum(((1 - data)**2) * dy)
    window = numpy.sum(((1 - data[:width_signal])**2) * dy[:width_signal])
    chi2[0] = fullsum-window
    for i in range(1, len(data) - width_signal + 1):
        becomes_visible = i-1
        becomes_invisible = i-1 + width_signal
        add_visible_left = (1 - data[becomes_visible])**2 * dy[becomes_visible]
        remove_invisible_right = (1 - data[becomes_invisible])**2 * dy[becomes_invisible]
        chi2[i] = chi2[i-1] + add_visible_left - remove_invisible_right
    return chi2


#@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def running_mean(data, width_signal):
    cumsum = numpy.cumsum(numpy.insert(data, 0, 0)) 
    a = (cumsum[width_signal:] - cumsum[:-width_signal]) / float(width_signal)
    return numpy.max(a)


# As periods are searched in parallel, the double parallel option for numba
# results in a speed penalty (factor two worse), so choose parallel=False here
@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def in_transit_residuals(data, signal, dy):
    outer_loop_length = len(data) - len(signal) + 1
    inner_loop_length = len(signal)
    chi2 = numpy.zeros(outer_loop_length + 0)
    for i in range(outer_loop_length):
        value = 0
        for j in range(inner_loop_length):
            value = value + ((data[i+j]-signal[j])**2) * dy[i+j]
        chi2[i] = value
    return chi2


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)
def fold(time, period, T0):
    return (time - T0) / period - numpy.floor((time - T0) / period)


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)
def foldfast(time, period):
    return time / period - numpy.floor(time / period)


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


    def _search_period(self, period, t, y, dy, lc_cache, lc_cache_overview):
        """Core routine to search the flux data set 'injected' over all 'periods'"""


        # duration (in samples) of widest transit in lc_cache (axis 0: rows; axis 1: columns)
        maxwidth_in_samples = numpy.shape(lc_cache)[1]

        # Phase fold
        phases = foldfast(t, period)
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

        # Edge effect correction
        regular_squared_residuals = numpy.sum(((1-flux)**2)*1/dy**2)
        patched_squared_residuals = numpy.sum(((1-patched_data)**2)*inverse_squared_patched_dy)
        edge_effect_correction = patched_squared_residuals - regular_squared_residuals
        #print('period', period, edge_effect_correction)
        #print('edge', regular_squared_residuals, patched_squared_residuals, edge_effect_correction)

        #print(period, 'GGGGGGGGGGGGG', a, b, modder)

        #ootr = out_of_transit_residuals(
        #        patched_data, maxwidth_in_samples, inverse_squared_patched_dy)
        ootr2 = ootr_efficient(
                patched_data, maxwidth_in_samples, inverse_squared_patched_dy)

        #ootr2 = ootr_efficient(
        #        flux, maxwidth_in_samples, dy)

        #print(len(ootr2), len(ootr))
        #for i in range(len(ootr2)):
        #    print(ootr[i])#, ootr[i])
        #print(ootr2)
        #print(ootr)
        
        # Set "best of" counters to max, in order to find smaller residuals
        smallest_residuals_in_period = float('inf')
        summed_residual_in_rows = float('inf')


        use_cuda = False;

        if use_cuda:
            phase_shift_loop = len(patched_data) - lc_cache.shape[1] + 1
            #print(lc_cache.shape[1])
            ootr = numpy.array(ootr2-2*edge_effect_correction, dtype='float32')
            #print('lc_cache.shape[0]', lc_cache.shape[0])
            #chi2map = numpy.zeros((lc_cache.shape[0], len(patched_data)-lc_cache.shape[1]), dtype='float32')
            data = numpy.array(patched_data, dtype='float32')
            dys = numpy.array(inverse_squared_patched_dy, dtype='float32')
            signals = numpy.array(lc_cache, dtype='float32')
            #print('BRAAAH')
            #print(signals)
            signal_row_sums = numpy.zeros(lc_cache.shape[0]-1, dtype='float32')
            #cuda.syncthreads()
            #blockdim = (8, 16)
            #griddim = (128, 128)

            #blockdim = (32, 8)
            #griddim = (32,16)
            #chi2_cuda = cuda.to_device(chi2)
            #data_cuda = cuda.to_device(data)
            #dys_cuda = cuda.to_device(dys)
            #signals_cuda = cuda.to_device(signals)
            #chi2map = cuda.device_array_like(chi2)
            #chi2map = 
            itr_cuda[32, 32](data, dys, signals, ootr, signal_row_sums) 
            #chi2map = numpy.array(chi2map, dtype=numpy.float32)
            #print(numpy.sum(chi2map))

            #chi2_cuda.to_host()
            signal_row_sums = numpy.array(signal_row_sums, dtype=numpy.float32)
            #print(signal_row_sums)
            #print(len(chi2[0,:]), len(ootr))
            #print('chi2', chi2.shape[0], chi2.shape[1])
            #print('lc_cache', lc_cache.shape[0], lc_cache.shape[1])
            #print(signal_row_sums)
            
            #for row in range(len(lc_cache)):
            #itr = chi2[row,:]
            #stats = itr + ootr
            #print('YEAHHH###', numpy.sum(itr))
            row = numpy.argmin(signal_row_sums)
            #print()
            current_smallest_residual = numpy.min(signal_row_sums)
            #current_smallest_residual = numpy.min(chi2map)
            
            #print(min(signal_row_sums), signal_row_sums)


            # Propagate results to outer loop (best duration, depth)
            if current_smallest_residual < summed_residual_in_rows:
                summed_residual_in_rows = current_smallest_residual
                best_row = row#ä1
                best_roll = 0

        else:
            # Iterate over all transit shapes (depths and durations)

            # Check if this signal shall be examined (or not: if too deep)
            # It would be better to test this not globally but per phase position
            # But to do that, the loop would need to change: first the phase,
            # then every signal at that phase. Test at the end what is faster
            # min_mean = 1 - running_mean(patched_data, len(lc_cache[0]))
            # trial_depth = lc_cache_overview['depth'][row]
            #   if min_mean < trial_depth:

            for row in range(len(lc_cache)):
                # This is the current transit (of some width, depth) to test
                scaled_transit = lc_cache[row]
                itr = in_transit_residuals(
                    data=patched_data,
                    signal=scaled_transit,
                    dy=inverse_squared_patched_dy)
                stats = ootr2 + itr - edge_effect_correction
                best_roll = numpy.argmin(stats)
                current_smallest_residual = stats[best_roll]

                # Propagate results to outer loop (best duration, depth)
                if current_smallest_residual < summed_residual_in_rows:
                    summed_residual_in_rows = current_smallest_residual
                    best_row = row

        if summed_residual_in_rows < smallest_residuals_in_period:
            smallest_residuals_in_period = summed_residual_in_rows
            best_shift = best_roll

        return [period, smallest_residuals_in_period, best_shift, best_row]


    def _perform_search(self, periods, t, y, dy, depths, durations, 
            limb_darkening=0.5, impact=0):
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
            lc_cache_overview=lc_cache_overview)
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


    def power(self, periods, durations, depths, limb_darkening=0.5, impact=0):
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
            impact=impact)

        # Sort residuals for best

        idx_best = numpy.argmin(test_statistic_residuals)
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

            # Instead of rolling, shift the id_flux_start by (- offset)
            id_flux_start = numpy.argmax(phases > start_transit)
            id_flux_start = id_flux_start - offset
            data_segment = flux[id_flux_start:id_flux_start+maxwidth_in_samples]
            phase_segment = phases[id_flux_start:id_flux_start+maxwidth_in_samples]
            dy_segment = dy[id_flux_start:id_flux_start+maxwidth_in_samples]

            # dynot inverted!?
            current_chi2 = numpy.sum((data_segment - signal)**2 / dy_segment**2)  
            pbar2.update(1)
            if current_chi2 < lowest_chi2:
                lowest_chi2 = current_chi2
                best_T0 = Tx

        pbar2.close()
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


        chi2 = test_statistic_residuals
        chi2red = test_statistic_residuals
        chi2red = chi2red / (len(self.t) - 4)
        SR = (1/chi2red)
        SR = SR/max(SR)

        SDE_power = ((1/chi2red)/numpy.std(chi2red))
        SDE_power = SDE_power-min(SDE_power)


        #/max(chi2red)
        #SDE_power = SR-numpy.mean(SR)
        #SDE_power = SDE_power / numpy.std(SDE_power)
        power = SDE_power
        SDE = max(SDE_power)

        # Make folded model
        #phases = fold(self.t, best_period, T0=best_T0+ best_period/2)
        #sort_index = numpy.argsort(phases)
        #phases = phases[sort_index]
        #flux = self.y[sort_index]
        #flux = numpy.roll(flux, 1)  

        folded_model = self.fractional_transit(
            duration=(best_duration * maxwidth_in_samples),
            maxwidth=maxwidth_in_samples / stretch,
            depth=best_depth,
            samples=int(len(self.t/epochs)),
            limb_darkening=limb_darkening,
            impact=impact)
        # Model and data are off by one cadence
        folded_model = numpy.roll(folded_model, -1)  

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
        #f = scipy.interpolate.interp1d(full_x_array, full_y_array)
        #xnew = numpy.linspace(min(self.t), max(self.t), len(self.t))
        model_flux = 1# f(xnew)

        # Get transit depth, standard deviation and SNR per transit
        transit_depths = numpy.zeros([len(transit_times)])
        transit_stds = numpy.zeros([len(transit_times)])
        transit_snrs = numpy.zeros([len(transit_times)])
        all_flux_intransit = numpy.array([])
        all_idx_intransit = numpy.array([])
        
        for i in range(len(transit_times)):
            mid_transit = transit_times[i]
            tmin = mid_transit - 0.5 * transit_duration_in_days
            tmax = mid_transit + 0.5 * transit_duration_in_days            
            idx_intransit = numpy.where(numpy.logical_and(self.t > tmin, self.t < tmax))
            all_idx_intransit = numpy.append(all_idx_intransit, idx_intransit)
            flux_intransit = self.y[idx_intransit]            
            all_flux_intransit = numpy.append(all_flux_intransit, flux_intransit)
            mean_flux = numpy.mean(self.y[idx_intransit])            
            #std_flux = numpy.std(self.y[idx_intransit]) ??? in-transit, all, or out?
            std_flux = numpy.std(self.y)
            snr = (1 - mean_flux) / std_flux
            transit_depths[i] = mean_flux
            transit_stds[i] = std_flux
            transit_snrs[i] = snr

        flux_ootr = numpy.delete(self.y, all_idx_intransit)
        total_depth = numpy.mean(all_flux_intransit)
        total_std = numpy.std(all_flux_intransit)
        total_snr = (1 - numpy.mean(all_flux_intransit)) / numpy.std(flux_ootr)
        #total_snr = (numpy.sqrt(len(transit_depths)) * total_depth) / numpy.mean(flux_ootr)
        #print('all_flux_intransit', all_flux_intransit)
        return TransitLeastSquaresResults(test_statistic_periods, \
            power, test_statistic_rolls, test_statistic_rows, \
            lc_cache_overview, best_period, best_T0, best_row, best_power, \
            SDE, test_statistic_rolls, best_depth, best_duration,\
            transit_times, transit_duration_in_days, maxwidth_in_samples, \
            folded_model, model_flux, chi2red, SDE_power, transit_depths,\
            transit_stds, transit_snrs, total_depth, total_std, total_snr, SR, chi2)


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
                "SDE", "rolls", "best_depth", "best_duration", "transit_times", \
                "transit_duration_in_days", "maxwidth_in_samples", \
                "folded_model", "model_flux", "chi2red", "SDE_power", \
                "transit_depths", "transit_stds", "transit_snrs", \
                "total_depth", "total_std", "total_snr", "SR", "chi2"), args))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

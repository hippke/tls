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
import multiprocessing
import math
import time

# batman-package for transit light curves
# https://www.cfa.harvard.edu/~lkreidberg/batman/
import batman
import scipy.interpolate
from tqdm import tqdm
from functools import partial
from numpy import pi, sqrt, arccos, degrees, floor, ceil
from array import array


"""Magic constants"""

# astrophysical constants
G = 6.673e-11  # [m^3 / kg / s^2]
R_sun = 695508000  # [m]
R_earth = 6371000
M_sun = 1.989*10**30  # [kg]
SECONDS_PER_DAY = 86400

# Default values as described in the paper
TRANSIT_DEPTH_MIN = 10 * 10**-6  # 10 ppm
LIMB_DARKENING = 0.5  # linear limb darkening
B = 0.0  # impact parameter

# For the period grid
R_STAR = 1.0
M_STAR = 1.0
OVERSAMPLING_FACTOR = 3
N_TRANSITS_MIN = 2

# For the duration grid
M_STAR_MIN = 0.1
M_STAR_MAX = 1.0
R_STAR_MIN = 0.13
R_STAR_MAX = 3.5
DURATION_GRID_STEP = 1.1

# For the transit template
PER = 365.25  # orbital period (in days)
RP = R_earth / R_sun  # planet radius (in units of stellar radii)
A = 217  # semi-major axis (in units of stellar radii)
INC = 90  # orbital inclination (in degrees)
B = 0  # impact parameter
ECC = 0  # eccentricity
W = 90  # longitude of periastron (in degrees)
# quadratic limb darkening for a G2V star in the Kepler bandpass
# http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BA/552/A16
U = [0.4804, 0.1867]  
LIMB_DARK = 'quadratic'

# Unique depth of trial signals (at various durations). These are rescaled in
# depth so that their integral matches the mean flux in the window in question.
# In principle, "signal_depth" is an arbitrary value. By using a standard close
# to the expected signal level, floating point inaccuracies are minimized 
SIGNAL_DEPTH = 0.5

# Maximum fractional transit duration ever observed is 0.117
# for Kepler-1368 b (as of Oct 2018), so we set upper_limit=0.15
# Long fractional transit durations are computationally expensive 
# following a quadratic relation. If applicable, use a different value.
# Longer transits can still be found, but at decreasing sensitivity
FRACTIONAL_TRANSIT_DURATION_MAX = 0.12

# Oversampling of the reference transit:
# "Don’t fit an unbinned model to binned data." 
# Reference: Kipping, D., "Binning is sinning: morphological light-curve 
#            distortions due to finite integration time"
#            MNRAS, Volume 408, Issue 3, pp. 1758-1769
#            http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1004.3741
# This is not time-critical as it has to be done only once
SUPERSAMPLE_SIZE = 10000



@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)
def T14(R_s, M_s, P, upper_limit=FRACTIONAL_TRANSIT_DURATION_MAX):
        """Input:  Stellar radius and mass; planetary period
                   Units: Solar radius and mass; days
           Output: Maximum planetary transit duration T_14max
                   Unit: Fraction of period P"""

        P = P * SECONDS_PER_DAY
        R_s = R_sun * R_s
        M_s = M_sun * M_s
        T14max = R_s * ((4*P) / (pi * G * M_s))**(1/3)
        result = T14max / P
        if result > upper_limit:
            result = upper_limit
        return result


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def get_edge_effect_correction(flux, patched_data, dy, inverse_squared_patched_dy):
    regular_squared_residuals = numpy.sum(((1-flux)**2)*1/dy**2)
    patched_squared_residuals = numpy.sum(((1 - patched_data)**2) * inverse_squared_patched_dy)
    edge_effect_correction = patched_squared_residuals - regular_squared_residuals
    return edge_effect_correction


def get_duration_grid(periods, log_step=1.1):
    duration_max = T14(R_s=3.50, M_s=1.0, P=min(periods))
    duration_min = T14(R_s=0.13, M_s=0.1, P=max(periods))
    durations = [duration_min]
    current_depth = duration_min
    while current_depth* log_step < duration_max:
        current_depth = current_depth * log_step
        durations.append(current_depth)
    durations.append(duration_max)  # Append endpoint. Not perfectly spaced.
    return durations


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


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def get_residuals(data, signal, dy):
    value = 0
    for i in range(len(data)):
        value = value + ((data[i]-signal[i])**2) * dy[i]
    return value


"""
@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def get_residuals_scale_transit(data, signal, dy, ratio):
    value = 0
    for i in range(len(data)):
        signal[i] = 1 - signal[i] * ratio
        value = value + ((data[i]-signal[i])**2) * dy[i]
    return value
"""

@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def get_residuals_scale_transit_iterator(data, sig, dy, target_depth):

    def f(k):
        value = 0
        for i in range(len(data)):
            #mod_signal = signal[i]
            #mod_signal = 1-signal[i]
            mod_signal = sig[i] * k
            mod_signal = 1-mod_signal
            value += ((data[i]-mod_signal)**2) * dy[i]
        return value

    #print('new')
    for i in range(len(sig)):
        buff = sig[i]
        #print(buff)
        scale = SIGNAL_DEPTH/target_depth
        #print(buff)
        sig[i] = 1-sig[i]
        sig[i] = sig[i]/scale
        #sig[i] = 1-sig[i]
        #print(SIGNAL_DEPTH, target_depth, scale, buff, sig[i])
    left = 0.1
    right = 2  # große Zahl macht den Transit tiefer
    required_precision = 0.1
    iters = 0
    while True:
        iters += 1
        left_third = left + (right - left) / 3
        right_third = right - (right - left) / 3
        if f(left_third) > f(right_third):
            left = left_third
        else:
            right = right_third
        if abs(right - left) < required_precision:
            break
    result = f((left + right)/2)
    k = (left + right)/2
    #print(iters)
    return result, k

"""
@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)  
def get_residuals_scale_transit_iterator(data, signal, dy, ratio):
    invphi = (math.sqrt(5) - 1) / 2 # 1/phi                                                                                                                     
    invphi2 = (3 - math.sqrt(5)) / 2 # 1/phi^2   
    #print(ratio)
    def f(k):
        value = 0
        for i in range(len(data)):
            #ratio = ratio * k
            test = 1 - signal[i] * ratio #* k
            value = value + ((data[i]-test)**2) * dy[i]
        return value
    #print('ratio', ratio)
    a = 0.5
    b = 1.5
    tol = 0.05
    (a,b)=(min(a,b),max(a,b))
    h = b - a
    if h <= tol: return (a,b)
    # required steps to achieve tolerance                                                                                                                   
    n = int(math.ceil(math.log(tol/h)/math.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)
    for z in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi*h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi*h
            d = a + invphi * h
            yd = f(d)
    result = f(d)
    return result, (a-b)/2
"""


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


def running_mean_max(data, width_signal):
    """Returns the maximum value in the running mean window"""
    cumsum = numpy.cumsum(numpy.insert(data, 0, 0)) 
    a = (cumsum[width_signal:] - cumsum[:-width_signal]) / float(width_signal)
    return numpy.max(a)


def running_mean(data, width_signal):
    """Returns the running mean in a given window"""
    cumsum = numpy.cumsum(numpy.insert(data, 0, 0)) 
    return (cumsum[width_signal:] - cumsum[:-width_signal]) / float(width_signal)


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)
def running_median(data, width_signal):
    idx = numpy.arange(width_signal) + numpy.arange(len(data)-width_signal+1)[:,None]
    a = numpy.median(data[idx], axis=1)
    first_values = a[0]
    last_values = a[-1]
    missing_values = len(data) - len(a)
    values_front = int(missing_values/2)
    values_end = missing_values - values_front
    a = numpy.append(numpy.full(values_front, first_values), a)
    a = numpy.append(a, numpy.full(values_end, last_values))
    return a


def running_median(data, width_signal):
    idx = numpy.arange(width_signal) + numpy.arange(len(data)-width_signal+1)[:,None]
    a = numpy.median(data[idx], axis=1)
    first_values = a[0]
    last_values = a[-1]
    missing_values = len(data) - len(a)
    values_front = int(missing_values/2)
    values_end = missing_values - values_front
    a = numpy.append(numpy.full(values_front, first_values), a)
    a = numpy.append(a, numpy.full(values_end, last_values))
    return a


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)
def fold(time, period, T0):
    return (time - T0) / period - numpy.floor((time - T0) / period)


@numba.jit(fastmath=True, parallel=False, cache=True, nopython=True)
def foldfast(time, period):
    return time / period - numpy.floor(time / period)


def period_grid(R_star, M_star, time_span, period_min=0, period_max=float('inf'), 
        oversampling_factor=2, n_transits_min=N_TRANSITS_MIN):
    """Returns array of optimal sampling periods for transit search in light curves
       Following Ofir (2014, A&A, 561, A138)"""

    R_star = R_star * R_sun
    M_star = M_sun * M_star
    time_span = time_span * SECONDS_PER_DAY  # seconds

    # boundary conditions
    f_min = n_transits_min / time_span
    f_max = 1. / (2*pi) * sqrt(G*M_star / (3 * R_star)**3)

    # optimal frequency sampling, Equations (5), (6), (7)
    A = ( 2*pi )**(2./3) / pi * R_star / (G*M_star)**(1./3) / \
        (time_span * oversampling_factor)
    C = f_min**(1./3) - A/3.
    N_opt = (f_max**(1./3) - f_min**(1./3) + A/3)*3/A

    X = numpy.arange(N_opt)+1
    f_x = ( A/3*X + C )**3
    P_x = 1/f_x

    # Cut to given (optional) selection of periods
    periods = P_x / SECONDS_PER_DAY
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


    def fractional_transit(self, duration, maxwidth, depth, samples, 
            per, rp, a, inc, ecc, w, u, limb_dark, cached_reference_transit=None):
        """Returns a scaled reference transit with fractional width and depth"""

        reference_time = numpy.linspace(-0.5, 0.5, samples)

        if cached_reference_transit is None:
            reference_flux = self.reference_transit(
                samples=samples,
                per=per,
                rp=rp,
                a=a,
                inc=inc,
                ecc=ecc,
                w=w,
                u=u,
                limb_dark=limb_dark)
        else:
            reference_flux = cached_reference_transit

        # Interpolate to shorter interval
        f = scipy.interpolate.interp1d(reference_time, reference_flux)
        occupied_samples = int((duration / maxwidth) * samples)
        #print('occupied_samples', occupied_samples)
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
        #print('result', result)

        return result


    def _impact_to_inclination(self, b, semimajor_axis):
            """Converts planet impact parameter b = [0..1.x] to inclination [deg]"""
            return degrees(arccos(b / semimajor_axis))


    def reference_transit(self, samples, per, rp, a, inc, ecc, w, u, limb_dark):
        """Returns an Earth-like transit of width 1 and depth 1"""

        # Box-shaped transit
        if limb_dark == 0:
            rescaled = numpy.zeros(samples)

        # Limb-darkened transit
        else:
            f = numpy.ones(SUPERSAMPLE_SIZE)
            duration = 1  # transit duration in days. Increase for exotic cases
            t = numpy.linspace(-duration/2, duration/2, SUPERSAMPLE_SIZE)
            ma = batman.TransitParams()
            ma.t0 = 0  # time of inferior conjunction
            ma.per = per  # orbital period, use Earth as a reference
            ma.rp = rp  # planet radius (in units of stellar radii)
            ma.a = a  # semi-major axis (in units of stellar radii)
            # orbital inclination (in degrees)
            ma.inc = inc
            ma.ecc = ecc  # eccentricity
            ma.w = w  # longitude of periastron (in degrees)
            ma.u = u  # limb darkening coefficients
            ma.limb_dark = limb_dark  # limb darkening model
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


    def _get_cache(self, durations, maxwidth_in_samples, \
        per, rp, a, inc, ecc, w, u, limb_dark):
        """Fetches (size(durations)*size(depths)) light curves of length 
        maxwidth_in_samples and returns these LCs in a 2D array, together with 
        their metadata in a separate array."""

        print('Creating model cache for', str(len(durations)), ' durations')
        rows = numpy.size(durations)
        lc_cache = numpy.ones([rows,maxwidth_in_samples])
        lc_cache_overview = numpy.zeros(
            rows, dtype=[
                ('duration', 'f8'),
                ('width_in_samples', 'i8'),
                ('first_sample', 'i8'),
                ('last_sample', 'i8'),
                ('flux_ratio', 'f8')])  # between transit shape and box
        cached_reference_transit = self.reference_transit(
            samples=maxwidth_in_samples,
            per=per,
            rp=rp,
            a=a,
            inc=inc,
            ecc=ecc,
            w=w,
            u=u,
            limb_dark=limb_dark)
        row = 0
        for duration in durations:  
            #print('duratio', duration)
            scaled_transit = self.fractional_transit(
                duration=duration,
                maxwidth=numpy.max(durations),
                depth=SIGNAL_DEPTH,
                samples=maxwidth_in_samples,
                per=per,
                rp=rp,
                a=a,
                inc=inc,
                ecc=ecc,
                w=w,
                u=u,
                limb_dark=limb_dark,
                cached_reference_transit=cached_reference_transit)
            lc_cache[row] = scaled_transit
            lc_cache_overview['duration'][row] = duration
            used_samples = int((duration / numpy.max(durations)) * maxwidth_in_samples)
            lc_cache_overview['width_in_samples'][row] = used_samples
            #empty_samples = maxwidth_in_samples - used_samples
            #first_sample = int(empty_samples/2)
            #last_sample = first_sample + used_samples
            cutoff = 0.01*10**-6  # 0.01 ppm tolerance for numerical stability
            full_values = numpy.where(scaled_transit<(1-cutoff))
            first_sample = numpy.min(full_values)
            last_sample = numpy.max(full_values) + 1
            #print('first_sample', first_sample, 'last_sample', last_sample)
            #print(lc_cache[row][first_sample:last_sample])
            lc_cache_overview['first_sample'][row] = first_sample
            lc_cache_overview['last_sample'][row] = last_sample

            # Scale factor
            #magic_number = 1.05#1.18
            signal = lc_cache[row][first_sample:last_sample]
            #signal = numpy.append(1, signal)
            #signal = numpy.append(signal, 1)
            integral_transit = numpy.sum(1-signal)
            integral_box = len(signal) * (1-min(signal))
            ratio = integral_transit / integral_box
            #ratio = ratio * magic_number
            lc_cache_overview['flux_ratio'][row] = ratio
            print('flux_ratio', ratio)
            #print(ratio)
            #print(min(scaled_transit), scaled_transit)

            row += + 1
      
        return lc_cache, lc_cache_overview


    def _search_period(self, period, t, y, dy, lc_cache, lc_cache_overview, \
            transit_depth_min, R_star_min, R_star_max, M_star_min, M_star_max):
        """Core routine to search the flux data set 'injected' over all 'periods'"""
        
        # duration (in samples) of widest transit in lc_cache (axis 0: rows; axis 1: columns)
        maxwidth_in_samples = numpy.shape(lc_cache)[1]

        # Phase fold
        phases = foldfast(t, period)
        sort_index = numpy.argsort(phases, kind='mergesort')  # 8% faster than Quicksort
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

        # Edge effect correction (numba speedup 40x) 
        edge_effect_correction = get_edge_effect_correction(
            flux, patched_data, dy, inverse_squared_patched_dy)
        # Strangely, this second part doesn't work with numba
        summed_edge_effect_correction = numpy.sum(edge_effect_correction)
        
        # Set "best of" counters to max, in order to find smaller residuals
        smallest_residuals_in_period = float('inf')
        summed_residual_in_rows = float('inf')

        # Make unique to avoid duplicates in dense grids
        durations = numpy.unique(lc_cache_overview['width_in_samples'])

        duration_max = T14(R_s=3.50, M_s=1.0, P=period)
        duration_min = T14(R_s=0.13, M_s=0.1, P=period)
        samples = len(y)
        duration_min_in_samples = int(floor(duration_min * samples))
        duration_max_in_samples = int(ceil(duration_max * samples))
        durations = durations[durations>=duration_min_in_samples]
        durations = durations[durations<=duration_max_in_samples]

        # Iterating over the full lc_cache_overview is slow
        # Thus, make a temporary reduced version
        # Faster to fetch from a list than numpy array, and Python array
        # All variants have been tested
        lc_cache_overview_list_first = lc_cache_overview['first_sample'][:].tolist()
        lc_cache_overview_list_last = lc_cache_overview['last_sample'][:].tolist()
        lc_cache_overview_list_flux_ratio = lc_cache_overview['flux_ratio'][:].tolist()
        patched_data_arr = array('f', patched_data)
        inverse_squared_patched_dy_arr = array('f', inverse_squared_patched_dy)

        # In case all sliding window means are smaller than half the
        # shallowest signal, all tests will be skipped. Then, treat all flux
        # as out of transit, as this is the better model
        skipped_all = True
        
        
        for duration in durations:
            ootr = ootr_efficient(
                patched_data, duration, inverse_squared_patched_dy)
            # It has shown to be slower to convert "mean" to Python array
            mean = 1 - running_mean(patched_data, duration)
            # Get the row with matching duration
            chosen_transit_row = 0
            while lc_cache_overview['width_in_samples'][chosen_transit_row] != duration:
                chosen_transit_row += 1

            
            array_to_check = numpy.where(mean > transit_depth_min)[0]
            #print(transit_depth_min, len(array_to_check))

            if len(array_to_check) > 0:
                skipped_all = False

                for k in array_to_check:
                    #start = lc_cache_overview_list_first[chosen_transit_row]  # 0.05
                    #end = lc_cache_overview_list_last[chosen_transit_row]  # 0.05


                    start = lc_cache_overview['first_sample'][chosen_transit_row]
                    end = lc_cache_overview['last_sample'][chosen_transit_row]
                    # Scale signal integral to match mean in this window
                    signal = lc_cache[chosen_transit_row][start:end]  # 0.3
                    siggi = numpy.copy(signal)
                    #print('yeah signal', signal)
                    #print('min(signal)', min(signal))
                    #print(signal)
                    #signal = 1 - signal
                    #corr = lc_cache_overview['flux_ratio'][chosen_transit_row]
                    #min_signal = max(signal)
                    #target_depth = mean[k]
                    #ratio = min_signal/target_depth
                    #ratio = ratio #/ corr
                    #signal = signal / ratio
                    #signal = signal * corr
                    #signal = 1-signal
                    #print(chosen_transit_row, start, end, signal)

                    #print(min(signal))

                    
                    # to fine-tune the depth, use get_residuals_scale_transit_iterator
                    itr_here, correction_factor = get_residuals_scale_transit_iterator(  # 2.0
                        numpy.copy(patched_data_arr[k:k+duration]), # data 0.3
                        siggi,  # signal 0.3
                        numpy.copy(inverse_squared_patched_dy_arr[k:k+duration]),  # dy 0.3
                        numpy.copy(mean[k]))
                    
                    current_stat = itr_here + ootr[k] - summed_edge_effect_correction  # 0.3
                    if current_stat < summed_residual_in_rows:  # 0.1
                        summed_residual_in_rows = current_stat
                        best_row = chosen_transit_row
                        best_roll = k/len(y)
                        if best_roll>1:  # in patched appendix
                            best_roll = best_roll - 1
                        #deepest_dip = #/ correction_factor
                        #mean[k] / (lc_cache_overview_list_flux_ratio[chosen_transit_row]*(1*correction_factor))
                        fill_factor = lc_cache_overview['flux_ratio'][chosen_transit_row]
                        scale = SIGNAL_DEPTH/mean[k]
                        best_depth = (1-min(siggi)) / scale
                        best_depth = best_depth / correction_factor
                        best_depth = best_depth * fill_factor
                        best_depth = 1-best_depth


                    

        if skipped_all:
            my_signal = numpy.ones(len(y))
            summed_residual_in_rows = get_residuals(
                y, my_signal, 1/dy**2)
            best_row = 0  # shortest and shallowest transit
            best_shift = 1
            best_roll = 1
            best_depth = 0

        if summed_residual_in_rows < smallest_residuals_in_period:
            smallest_residuals_in_period = summed_residual_in_rows
            best_shift = best_roll
        
        try:
            print(period, best_depth, correction_factor)
        except:
            pass

        return [period, smallest_residuals_in_period, best_shift, best_row, best_depth]


    def power(self, **kwargs):
        """Compute the periodogram for a set user-defined parameters"""

        # Validate **kwargs and set to defaults where missing 
        self.limb_darkening = kwargs.get('limb_darkening', LIMB_DARKENING)
        self.transit_depth_min = kwargs.get('transit_depth_min', TRANSIT_DEPTH_MIN)
        #self.b = kwargs.get('b', B)

        self.R_star =  kwargs.get('R_star', R_STAR)
        self.M_star =  kwargs.get('M_star', M_STAR)
        self.oversampling_factor =  kwargs.get('oversampling_factor', OVERSAMPLING_FACTOR)
        self.period_max =  kwargs.get('period_max', float('inf'))
        self.period_min =  kwargs.get('period_min', 0)
        self.n_transits_min = kwargs.get('n_transits_min', N_TRANSITS_MIN)

        self.R_star_min = kwargs.get('R_star_min', R_STAR_MIN)
        self.R_star_max = kwargs.get('R_star_max', R_STAR_MAX)
        self.M_star_min = kwargs.get('M_star_min', M_STAR_MIN)
        self.M_star_max = kwargs.get('M_star_max', M_STAR_MIN)
        self.duration_grid_step = kwargs.get('duration_grid_step', DURATION_GRID_STEP)

        self.per = kwargs.get('per', PER)
        self.rp = kwargs.get('rp', RP)
        self.a = kwargs.get('a', A)
        self.inc = kwargs.get('inc', INC)

        # If an impact parameter is given, it overrules the supplied inclination
        if 'b' in kwargs:
            self.b = kwargs.get('b')
            self.inc = self._impact_to_inclination(b=self.b, semimajor_axis=self.a)  

        self.ecc = kwargs.get('ecc', ECC)
        self.w = kwargs.get('w', W)
        self.u = kwargs.get('u', U)
        self.limb_dark = kwargs.get('limb_dark', LIMB_DARK)

        periods = period_grid(
            R_star=self.R_star,
            M_star=self.M_star,
            time_span=numpy.max(self.t)-numpy.min(self.t),
            period_min=self.period_min,
            period_max=self.period_max, 
            oversampling_factor=self.oversampling_factor,
            n_transits_min=self.n_transits_min)
        print('len(periods)', len(periods))
        print('self.transit_depth_min', self.transit_depth_min)

        durations = get_duration_grid(periods, log_step=self.duration_grid_step)

        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        lc_cache, lc_cache_overview = self._get_cache(
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark)

        # Prepare result arrays
        test_statistic_periods = []
        test_statistic_residuals = []
        test_statistic_rolls = []
        test_statistic_rows = []
        test_statistic_depths = []

        chosen_transit_row = 5
        
        """
        for row in range(len(lc_cache)):
            start = lc_cache_overview['first_sample'][row]
            end = lc_cache_overview['last_sample'][row]
            print(row, start, end, lc_cache[row][start:end])
        """


        text = 'Searching ' + str(len(self.y)) + ' data points, ' + \
            str(len(periods)) + ' periods from ' + \
            str(round(min(periods), 3)) + ' to ' + \
            str(round(max(periods), 3)) + ' days, using all ' + \
            str(multiprocessing.cpu_count()) + ' CPU threads'
        print(text)
        p = multiprocessing.Pool(processes=1)#multiprocessing.cpu_count())
        params = partial(
            self._search_period,
            t=self.t,
            y=self.y,
            dy=self.dy,
            lc_cache=lc_cache,
            lc_cache_overview=lc_cache_overview,
            transit_depth_min=self.transit_depth_min,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=self.M_star_min,
            M_star_max=self.M_star_max)
        bar_format = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} periods | {elapsed}<{remaining}'#' | {rate_fmt}'
        pbar = tqdm(total=numpy.size(periods), smoothing=0.3, mininterval=1, bar_format=bar_format)
        for data in p.imap_unordered(params, reversed(periods)):  # small to large
            test_statistic_periods.append(data[0])
            test_statistic_residuals.append(data[1])
            test_statistic_rolls.append(data[2])
            test_statistic_rows.append(data[3])
            test_statistic_depths.append(data[4])
            pbar.update(1)
        p.close()
        pbar.close()

        # imap_unordered delivers results in unsorted order ==> sort
        test_statistic_periods = numpy.array(test_statistic_periods)
        test_statistic_residuals = numpy.array(test_statistic_residuals)
        test_statistic_rolls = numpy.array(test_statistic_rolls)
        test_statistic_rows = numpy.array(test_statistic_rows)
        test_statistic_depths = numpy.array(test_statistic_depths)
        sort_index = numpy.argsort(test_statistic_periods)
        test_statistic_periods = test_statistic_periods[sort_index]
        test_statistic_residuals = test_statistic_residuals[sort_index]
        test_statistic_rolls = test_statistic_rolls[sort_index]
        test_statistic_rows = test_statistic_rows[sort_index]
        test_statistic_depths = test_statistic_depths[sort_index]

        # Sort residuals for best
        idx_best = numpy.argmin(test_statistic_residuals)
        best_power = test_statistic_residuals[idx_best]
        best_period = test_statistic_periods[idx_best]
        best_roll = test_statistic_rolls[idx_best]
        best_row = test_statistic_rows[idx_best]
        best_depth = test_statistic_depths[idx_best]
        best_roll_save = test_statistic_rolls[idx_best]

        #best_depth = lc_cache_overview['depth'][best_row]
        best_duration = lc_cache_overview['duration'][best_row]

        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.t))
        print('best_roll (phase)', best_roll)
        print('best depth', best_depth)
        
        

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
        #best_depth = best_depth * 1.1
        signal = self.fractional_transit(
            duration=best_duration,
            maxwidth=max(durations), 
            depth=1-best_depth,
            samples=maxwidth_in_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark)        

        lowest_chi2 = float('inf')
        best_T0 = 0
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

            # Sorted array is not perfectly linear: phase = 0.5 != numpy.max(phases)/2
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

            # dy not inverted
            current_chi2 = numpy.sum((data_segment - signal)**2 / dy_segment**2)  
            pbar2.update(1)
            if current_chi2 < lowest_chi2:
                lowest_chi2 = current_chi2
                best_T0 = Tx

        pbar2.close()
        print('best_T0 ##########', best_T0)
        best_T0_calculated = (best_roll - 0.5*best_duration)*best_period + numpy.min(self.t)
        print('best T_0 calculated', best_T0_calculated)
        #best_T0 = best_T0_calculated

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

        SR = min(chi2) / chi2
        SDE = (1 - numpy.mean(SR)) / numpy.std(SR)
        print('SDE_max', SDE)

        # Scale SDE_power from 0 to SDE
        SDE_power = SR - numpy.mean(SR)      # shift down to the mean being zero
        scale = SDE / numpy.max(SDE_power)  # scale factor to touch max=SDE
        SDE_power = SDE_power * scale
        power = SDE_power

        # Detrended SDE
        kernel_size = 151
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        if len(SDE_power) > 2*kernel_size:
            my_median = running_median(SDE_power, kernel_size)
            cleaned_SDE_power = SDE_power - my_median
            # Re-normalize to range between median = 0 and peak = SDE
            # shift down to the mean being zero
            cleaned_SDE_power = cleaned_SDE_power - numpy.mean(cleaned_SDE_power)
            cleaned_SDE = numpy.max(cleaned_SDE_power / numpy.std(cleaned_SDE_power))     
            # scale factor to touch max=SDE 
            scale = cleaned_SDE / numpy.max(cleaned_SDE_power)  
            cleaned_SDE_power = cleaned_SDE_power * scale
            # Recalculate SDE
        else:
            cleaned_SDE_power = SDE_power
            cleaned_SDE = SDE
            
            print('cleaned_SDE', cleaned_SDE)


        #ratio = 0.8
        #signal = signal / ratio
        folded_model = self.fractional_transit(
            duration=(best_duration * maxwidth_in_samples),
            maxwidth=maxwidth_in_samples / stretch,
            depth=1-best_depth,
            samples=int(len(self.t/epochs)),
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark)
        # Model and data are off by one cadence
        #folded_model = numpy.roll(folded_model, -1)  

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
            depth=1-best_depth,
            samples=internal_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark)

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


        return TransitLeastSquaresResults(test_statistic_periods, \
            power, test_statistic_rolls, test_statistic_rows, \
            lc_cache_overview, best_period, best_T0, best_row, best_power, \
            SDE, test_statistic_rolls, best_depth, best_duration,\
            transit_times, transit_duration_in_days, maxwidth_in_samples, \
            folded_model, model_flux, chi2red, SDE_power, transit_depths,\
            transit_stds, transit_snrs, total_depth, total_std, total_snr, SR, chi2,\
            cleaned_SDE, cleaned_SDE_power)


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
                "total_depth", "total_std", "total_snr", "SR", "chi2",\
                "cleaned_SDE", "cleaned_SDE_power"), args))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#  Optimized algorithm to search for transits of small extrasolar planets
#                                                                            /
#       ,        AUTHORS                                                   O/
#    \  :  /     Michael Hippke (1) [michael@hippke.org]                /\/|
# `. __/ \__ .'  Rene' Heller (2) [heller@mps.mpg.de]                      |
# _ _\     /_ _  _________________________________________________________/ \_
#    /_   _\
#  .'  \ /  `.   (1) Sonneberg Observatory, Sternwartestr. 32, Sonneberg
#    /  :  \     (2) Max Planck Institute for Solar System Research,
#       '            Justus-von-Liebig-Weg 3, 37077 G\"ottingen, Germany


import batman  # https://www.cfa.harvard.edu/~lkreidberg/batman/
import http.client as httplib
import json
import multiprocessing
import numba
import numpy
import time
import scipy.interpolate
import sys
import warnings
import argparse
import configparser
from os import path
from functools import partial
from numpy import pi, sqrt, arccos, degrees, floor, ceil
from tqdm import tqdm
from urllib.parse import quote as urlencode


"""Magic constants"""
TLS_VERSION = (
    "Transit Least Squares TLS 1.0.12 (07 January 2019)"
)
numpy.set_printoptions(threshold=numpy.nan)
resources_dir = path.join(path.dirname(__file__))

# astrophysical constants
G = 6.673e-11  # gravitational constant [m^3 / kg / s^2]
R_sun = 695508000  # radius of the Sun [m]
R_earth = 6371000  # radius of the Earth [m]
R_jup = 69911000  # radius of Jupiter [m]
M_sun = 1.989 * 10 ** 30  # mass of the Sun [kg]
SECONDS_PER_DAY = 86400

# Default values as described in the paper
TRANSIT_DEPTH_MIN = 10 * 10 ** -6  # 10 ppm

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
# quadratic limb darkening for a G2V star in the Kepler bandpass
# http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BA/552/A16
U = [0.4804, 0.1867]
LIMB_DARK = "quadratic"
ECC = 0  # eccentricity
W = 90  # longitude of periastron (in degrees)

# Unique depth of trial signals (at various durations). These are rescaled in
# depth so that their integral matches the mean flux in the window in question.
# In principle, "signal_depth" is an arbitrary value >0 and <1
SIGNAL_DEPTH = 0.5

# Maximum fractional transit duration ever observed is 0.117
# for Kepler-1368 b (as of Oct 2018), so we set upper_limit=0.15
# Long fractional transit durations are computationally expensive
# following a quadratic relation. If applicable, use a different value.
# Longer transits can still be found, but at decreasing sensitivity
FRACTIONAL_TRANSIT_DURATION_MAX = 0.12

# Oversampling ==> Downsampling of the reference transit:
# "Do not fit an unbinned model to binned data."
# Reference: Kipping, D., "Binning is sinning: morphological light-curve
#            distortions due to finite integration time"
#            MNRAS, Volume 408, Issue 3, pp. 1758-1769
#            http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1004.3741
# This is not time-critical as it has to be done only once
# To work for all periods correctly, it has to be re-done at each period
# This feature is currently not implemented
SUPERSAMPLE_SIZE = 10000

# Order in which the periods are searched: "shuffled"", "descending", "ascending"
# Shuffled has the advantage of the best estimate for the remaining time
PERIODS_SEARCH_ORDER = "shuffled"

# When converting power_raw to power, a median of a certain window size is subtracted.
# For periodograms of smaller width, no smoothing is applied. The kernel size is
# calculated as kernel = oversampling_factor * SDE_MEDIAN_KERNEL_SIZE
# This value has proven to yield numerically stable results.
SDE_MEDIAN_KERNEL_SIZE = 30

# AFFECTS ONLY THE FINAL T0 FIT, NOT THE SDE
# We can give user the option to not scan the phase space for T0 at every cadence
# For speed reasons, it may be acceptable to approximate T0 to within X %
# Useful in large datasets. 100k points: Extra runtime of order 2 minutes
# While individual transits often are only a few cadences long, in the stacked
# phase space it is (N transits * transit duration) [cadences] long
T0_FIT_MARGIN = 0.01  # of transit duration e.g., 0.01 (=1%)

# The secondary fit for T0 can take negligible or significant time, depending on the
# number of data points and on T0_FIT_MARGIN. Set an empirical threshold to avoid
# displaying a progress bar when the estimated runtime is low (<~1 sec)
# To display the progress bar in more cases, use a lower number
PROGRESSBAR_THRESHOLD = 5000


def FAP(SDE):
    """Returns FAP (False Alarm Probability) for a given SDE"""
    data = numpy.genfromtxt(
            path.join(resources_dir, "fap.csv"),
            dtype="f8, f8",
            names=["FAP", "SDE"],
        )
    return data["FAP"][numpy.argmax(data["SDE"]>SDE)]


def resample(time, flux, factor):
    f = scipy.interpolate.interp1d(
        time, flux, assume_sorted=True
    )
    time_grid = int(len(flux) / factor)
    time_resampled = numpy.linspace(
        min(time), max(time), time_grid
    )
    flux_resampled = f(time_resampled)
    return time_resampled, flux_resampled


def rp_rs_from_depth(depth, a, b):
    """Takes the maximum transit depth and quadratic limb darkening params a, b
    Returns R_P / R_S (ratio of planetary to stellar radius
    Source: Heller et al. 2018 in prep"""

    return ((depth - 1) * (2 * a + b - 6)) ** (
        1 / 2
    ) / 6 ** (1 / 2)


def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays."""

    # Start with empty Python lists and convert to numpy arrays later (reason: speed)
    clean_t = []
    clean_y = []
    if dy is not None:
        clean_dy = []

    # Cleaning numpy arrays with both NaN and None values is not trivial, as the usual
    # mask/delete filters do not accept their simultanous ocurrence.
    # Instead, we iterate over the array once; this is not Pythonic but works reliably.
    for i in range(len(y)):
        if (
            (y[i] is not None)
            and (y[i] is not numpy.nan)
            and (y[i] >= 0)
            and (y[i] < numpy.inf)
        ):
            clean_y.append(y[i])
            clean_t.append(t[i])
            if dy is not None:
                clean_dy.append(dy[i])

    clean_t = numpy.array(clean_t, dtype=float)
    clean_y = numpy.array(clean_y, dtype=float)

    if dy is None:
        return clean_t, clean_y
    else:
        clean_dy = numpy.array(clean_dy, dtype=float)
        return clean_t, clean_y, clean_dy


@numba.jit(fastmath=True, parallel=False, nopython=True)
def transit_mask(t, period, duration, T0):
    mask = (
        numpy.abs(
            (t - T0 + 0.5 * period) % period - 0.5 * period
        )
        < 0.5 * duration
    )
    return mask


@numba.jit(fastmath=True, parallel=False, nopython=True)
def T14(
    R_s,
    M_s,
    P,
    upper_limit=FRACTIONAL_TRANSIT_DURATION_MAX,
    small=False,
):
    """Input:  Stellar radius and mass; planetary period
                   Units: Solar radius and mass; days
       Output: Maximum planetary transit duration T_14max
               Unit: Fraction of period P"""

    P = P * SECONDS_PER_DAY
    R_s = R_sun * R_s
    M_s = M_sun * M_s

    if small:  # small planet assumption
        T14max = R_s * ((4 * P) / (pi * G * M_s)) ** (1 / 3)
    else:  # planet size 2 R_jup
        T14max = (R_s + 2 * R_jup) * (
            (4 * P) / (pi * G * M_s)
        ) ** (1 / 3)

    result = T14max / P
    if result > upper_limit:
        result = upper_limit
    return result


@numba.jit(fastmath=True, parallel=False, nopython=True)
def get_edge_effect_correction(
    flux, patched_data, dy, inverse_squared_patched_dy
):
    regular = numpy.sum(((1 - flux) ** 2) * 1 / dy ** 2)
    patched = numpy.sum(
        ((1 - patched_data) ** 2)
        * inverse_squared_patched_dy
    )
    return patched - regular


def get_duration_grid(periods, shortest, log_step=1.1):
    duration_max = T14(R_s=3.50, M_s=1.0, P=min(periods))
    # duration_min = 2*shortest
    duration_min = T14(R_s=0.13, M_s=0.1, P=max(periods))
    durations = [duration_min]
    current_depth = duration_min
    while current_depth * log_step < duration_max:
        current_depth = current_depth * log_step
        durations.append(current_depth)
    durations.append(
        duration_max
    )  # Append endpoint. Not perfectly spaced.
    return durations


def mastQuery(request):
    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        "User-agent": "python-requests/"
        + ".".join(map(str, sys.version_info[:3])),
    }
    conn = httplib.HTTPSConnection("mast.stsci.edu")
    conn.request(
        "POST",
        "/api/v0/invoke",
        "request=" + urlencode(json.dumps(request)),
        headers,
    )
    response = conn.getresponse()
    header = response.getheaders()
    content = response.read().decode("utf-8")
    conn.close()
    return header, content


def get_tic_data(TIC_ID):
    adv_filters = [
        {
            "paramName": "ID",
            "values": [{"min": TIC_ID, "max": TIC_ID}],
        }
    ]
    headers, outString = mastQuery(
        {
            "service": "Mast.Catalogs.Filtered.Tic",
            "format": "json",
            "params": {
                "columns": "c.*",
                "filters": adv_filters,
            },
        }
    )
    print(outString)
    return json.loads(outString)["data"]


def catalog_info(EPIC_ID=None, TIC_ID=None, KOI_ID=None):

    """Takes EPIC ID, returns limb darkening parameters u (linear) and
        a,b (quadratic), and stellar parameters. Values are pulled for minimum
        absolute deviation between given/catalog Teff and logg. Data are from:
        - K2 Ecliptic Plane Input Catalog, Huber+ 2016, 2016ApJS..224....2H
        - New limb-darkening coefficients, Claret+ 2012, 2013,
          2012A&A...546A..14C, 2013A&A...552A..16C"""

    if (EPIC_ID is None) and (TIC_ID is None) and (KOI_ID is None):
        raise ValueError("No ID was given")
    if (EPIC_ID is not None) and (TIC_ID is not None):
        raise ValueError("Only one ID allowed")
    if (EPIC_ID is not None) and (KOI_ID is not None):
        raise ValueError("Only one ID allowed")
    if (TIC_ID is not None) and (KOI_ID is not None):
        raise ValueError("Only one ID allowed")

    # KOI CASE (Kepler K1)
    if KOI_ID is not None:
        try:
            import kplr
        except:
            raise ImportError(
                'Package kplr required for KOI_ID but failed to import'
            )
        koi = kplr.API().koi(KOI_ID)
        a = koi.koi_ldm_coeff1
        b = koi.koi_ldm_coeff2
        mass = koi.koi_smass
        mass_min = koi.koi_smass_err1
        mass_max = koi.koi_smass_err2
        radius = koi.koi_srad
        radius_min = koi.koi_srad_err1
        radius_max = koi.koi_srad_err2

    # EPIC CASE (Kepler K2)
    if EPIC_ID is not None:
        if type(EPIC_ID) is not int:
            raise TypeError(
                'EPIC_ID ID must be of type "int"'
            )
        if (EPIC_ID < 201000001) or (EPIC_ID > 251813738):
            raise TypeError(
                "EPIC_ID ID must be in range 201000001 to 251813738"
            )

        try:
            from astroquery.vizier import Vizier
        except:
            raise ImportError(
                'Package astroquery.vizier required for EPIC_ID but failed to import'
            )

        columns=["Teff", "logg", "Rad", "E_Rad", "e_Rad", "Mass", "E_Mass", "e_Mass"]
        catalog="IV/34/epic"
        result = Vizier(columns=columns).query_constraints(ID=EPIC_ID, catalog=catalog)[0].as_array()
        Teff = result[0][0]
        logg = result[0][1]
        radius = result[0][2]
        radius_max = result[0][3]
        radius_min = result[0][4]
        mass = result[0][5]
        mass_max = result[0][6]
        mass_min = result[0][7]

        # Kepler limb darkening, load from locally saved CSV file
        ld = numpy.genfromtxt(
            path.join(
                resources_dir, "JAA546A14limb1-4.csv"
            ),
            skip_header=1,
            delimiter=",",
            dtype="f8, int32, f8, f8, f8",
            names=["logg", "Teff", "u", "a", "b"],
        )

    # TESS CASE
    if TIC_ID is not None:
        if type(TIC_ID) is not int:
            raise TypeError(
                'TIC_ID ID must be of type "int"'
            )

        # Load entry for TESS Input Catalog from MAST
        tic_data = get_tic_data(TIC_ID)

        if len(tic_data) != 1:
            raise ValueError("TIC_ID not in catalog")

        star = tic_data[0]
        ld = numpy.genfromtxt(
            path.join(resources_dir, "ld_claret_tess.csv"),
            skip_header=1,
            delimiter=";",
            dtype="f8, int32, f8, f8",
            names=["logg", "Teff", "a", "b"],
        )
        Teff = star["Teff"]
        logg = star["logg"]
        radius = star["rad"]
        radius_max = star[
            "e_rad"
        ]  # only one uncertainty is provided
        radius_min = star["e_rad"]
        mass = star["mass"]
        mass_max = star[
            "e_mass"
        ]  # only one uncertainty is provided
        mass_min = star["e_mass"]

        if logg is None:
            logg = 4
            warnings.warn(
                "No logg in catalog. Proceeding with logg=4"
            )

    """From here on, K2 and TESS catalogs work the same:
        - Take Teff from star catalog and find nearest entry in LD catalog
        - Same for logg, but only for the Teff values returned before
        - Return stellar parameters and best-match LD
    """
    if KOI_ID is None:
        # Find nearest Teff and logg
        nearest_Teff = ld["Teff"][
            (numpy.abs(ld["Teff"] - Teff)).argmin()
        ]
        idx_all_Teffs = numpy.where(ld["Teff"] == nearest_Teff)
        relevant_lds = numpy.copy(ld[idx_all_Teffs])
        idx_nearest = numpy.abs(
            relevant_lds["logg"] - logg
        ).argmin()
        a = relevant_lds["a"][idx_nearest]
        b = relevant_lds["b"][idx_nearest]

    return (
        (a, b),
        mass,
        mass_min,
        mass_max,
        radius,
        radius_min,
        radius_max,
    )


@numba.jit(fastmath=True, parallel=False, nopython=True)
def pink_noise(data, width):
    std = 0
    datapoints = len(data) - width + 1
    for i in range(datapoints):
        std += numpy.std(data[i : i + width]) / width ** 0.5
    return std / datapoints


@numba.jit(fastmath=True, parallel=False, nopython=True)
def get_lowest_residuals_in_this_duration(
    mean,
    transit_depth_min,
    patched_data_arr,
    duration,
    signal,
    inverse_squared_patched_dy_arr,
    overshoot,
    ootr,
    summed_edge_effect_correction,
    chosen_transit_row,
    datapoints,
    T0_fit_margin,
):

    # if nothing is fit, we fit a straight line: signal=1. Then, at dy=1,
    # the squared sum of residuals equals the number of datapoints
    summed_residual_in_rows = datapoints
    best_row = 0
    best_depth = 0

    xth_point = 1
    if T0_fit_margin > 0 and duration > T0_fit_margin:
        T0_fit_margin = 1 / T0_fit_margin
        xth_point = int(duration / T0_fit_margin)
        if xth_point < 1:
            xth_point = 1

    for i in range(len(mean)):
        if (mean[i] > transit_depth_min) and (
            i % xth_point == 0
        ):
            data = patched_data_arr[i : i + duration]
            dy = inverse_squared_patched_dy_arr[
                i : i + duration
            ]
            target_depth = mean[i] * overshoot
            scale = SIGNAL_DEPTH / target_depth
            reverse_scale = (
                1 / scale
            )  # speed: one division now, many mults later

            # Scale model and calculate residuals
            intransit_residual = 0
            for j in range(len(signal)):
                sigi = (1 - signal[j]) * reverse_scale
                intransit_residual += (
                    (data[j] - (1 - sigi)) ** 2
                ) * dy[j]

            current_stat = (
                intransit_residual
                + ootr[i]
                - summed_edge_effect_correction
            )

            if current_stat < summed_residual_in_rows:
                summed_residual_in_rows = current_stat
                best_row = chosen_transit_row
                best_depth = 1 - target_depth

    return summed_residual_in_rows, best_row, best_depth


@numba.jit(fastmath=True, parallel=False, nopython=True)
def ootr_efficient(data, width_signal, dy):
    chi2 = numpy.zeros(len(data) - width_signal + 1)
    fullsum = numpy.sum(((1 - data) ** 2) * dy)
    window = numpy.sum(
        ((1 - data[:width_signal]) ** 2) * dy[:width_signal]
    )
    chi2[0] = fullsum - window
    for i in range(1, len(data) - width_signal + 1):
        becomes_visible = i - 1
        becomes_invisible = i - 1 + width_signal
        add_visible_left = (
            1 - data[becomes_visible]
        ) ** 2 * dy[becomes_visible]
        remove_invisible_right = (
            1 - data[becomes_invisible]
        ) ** 2 * dy[becomes_invisible]
        chi2[i] = (
            chi2[i - 1]
            + add_visible_left
            - remove_invisible_right
        )
    return chi2


def running_mean(data, width_signal):
    """Returns the running mean in a given window"""
    cumsum = numpy.cumsum(numpy.insert(data, 0, 0))
    return (
        cumsum[width_signal:] - cumsum[:-width_signal]
    ) / float(width_signal)


def running_median(data, kernel):
    """Returns sliding median of width 'kernel' and same length as data """
    idx = (
        numpy.arange(kernel)
        + numpy.arange(len(data) - kernel + 1)[:, None]
    )
    med = numpy.median(data[idx], axis=1)

    # Append the first/last value at the beginning/end to match the length of
    # data and returned median
    first_values = med[0]
    last_values = med[-1]
    missing_values = len(data) - len(med)
    values_front = int(missing_values * 0.5)
    values_end = missing_values - values_front
    med = numpy.append(
        numpy.full(values_front, first_values), med
    )
    med = numpy.append(
        med, numpy.full(values_end, last_values)
    )
    return med


@numba.jit(fastmath=True, parallel=False, nopython=True)
def fold(time, period, T0):
    """Normal phase folding"""
    return (time - T0) / period - numpy.floor(
        (time - T0) / period
    )


@numba.jit(fastmath=True, parallel=False, nopython=True)
def foldfast(time, period):
    """Fast phase folding with T0=0 hardcoded"""
    return time / period - numpy.floor(time / period)


def period_grid(
    R_star,
    M_star,
    time_span,
    period_min=0,
    period_max=float("inf"),
    oversampling_factor=2,
    n_transits_min=N_TRANSITS_MIN,
):
    """Returns array of optimal sampling periods for transit search in light curves
       Following Ofir (2014, A&A, 561, A138)"""

    R_star = R_star * R_sun
    M_star = M_sun * M_star
    time_span = time_span * SECONDS_PER_DAY  # seconds

    # boundary conditions
    f_min = n_transits_min / time_span
    f_max = (
        1.0
        / (2 * pi)
        * sqrt(G * M_star / (3 * R_star) ** 3)
    )

    # optimal frequency sampling, Equations (5), (6), (7)
    A = (
        (2 * pi) ** (2.0 / 3)
        / pi
        * R_star
        / (G * M_star) ** (1.0 / 3)
        / (time_span * oversampling_factor)
    )
    C = f_min ** (1.0 / 3) - A / 3.0
    N_opt = (
        (f_max ** (1.0 / 3) - f_min ** (1.0 / 3) + A / 3)
        * 3
        / A
    )

    X = numpy.arange(N_opt) + 1
    f_x = (A / 3 * X + C) ** 3
    P_x = 1 / f_x

    # Cut to given (optional) selection of periods
    periods = P_x / SECONDS_PER_DAY
    selected_index = numpy.where(
        numpy.logical_and(
            periods > period_min, periods <= period_max
        )
    )

    if numpy.size(periods[selected_index]) == 0:
        raise ValueError("Empty period array")

    return periods[selected_index]  # periods in [days]


class transitleastsquares(object):
    """Compute the transit least squares of limb-darkened transit models"""

    def __init__(self, t, y, dy=None):
        self.t, self.y, self.dy = self._validate_inputs(
            t, y, dy
        )

    def _validate_inputs(self, t, y, dy):
        """Check the consistency of the inputs"""

        duration = max(t) - min(t)
        if duration <= 0:
            raise ValueError("Time duration must positive")
        if (
            numpy.size(y) < 3 or numpy.size(t) < 3
        ):  # or numpy.size(dy) < 3:
            raise ValueError("Too few values in data set")
        if numpy.mean(y) > 1.01 or numpy.mean(y) < 0.99:
            warnings.warn(
                "Warning: The mean flux should be normalized to 1"
                + ", but it was found to be "
                + str(numpy.mean(y))
            )
        if min(y) < 0:
            raise ValueError("Flux values must be positive")
        if max(y) >= float("inf"):
            raise ValueError("Flux values must be finite")

        # If no dy is given, create it with the standard deviation of the flux
        if dy is None:
            dy = numpy.full(len(y), numpy.std(y))
        if numpy.size(t) != numpy.size(y) or numpy.size(
            t
        ) != numpy.size(dy):
            raise ValueError(
                "Arrays (t, y, dy) must be of the same dimensions"
            )
        if (
            t.ndim != 1
        ):  # Size identity ensures dimensional identity
            raise ValueError(
                "Inputs (t, y, dy) must be 1-dimensional"
            )

        return t, y, dy

    def fractional_transit(
        self,
        duration,
        maxwidth,
        depth,
        samples,
        per,
        rp,
        a,
        inc,
        ecc,
        w,
        u,
        limb_dark,
        cached_reference_transit=None,
    ):
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
                limb_dark=limb_dark,
            )
        else:
            reference_flux = cached_reference_transit

        # Interpolate to shorter interval
        f = scipy.interpolate.interp1d(
            reference_time,
            reference_flux,
            assume_sorted=True,
        )
        occupied_samples = int(
            (duration / maxwidth) * samples
        )
        ynew = f(
            numpy.linspace(-0.5, 0.5, occupied_samples)
        )

        # Patch ends with ones ("1")
        missing_samples = samples - occupied_samples
        emtpy_segment = numpy.ones(
            int(missing_samples * 0.5)
        )
        result = numpy.append(emtpy_segment, ynew)
        result = numpy.append(result, emtpy_segment)
        if (
            numpy.size(result) < samples
        ):  # If odd number of samples
            result = numpy.append(result, numpy.ones(1))

        # Depth rescaling
        result = 1 - ((1 - result) * depth)

        return result

    def _impact_to_inclination(self, b, semimajor_axis):
        """Converts planet impact parameter b = [0..1.x] to inclination [deg]"""
        return degrees(arccos(b / semimajor_axis))

    def reference_transit(
        self, samples, per, rp, a, inc, ecc, w, u, limb_dark
    ):
        """Returns an Earth-like transit of width 1 and depth 1"""

        f = numpy.ones(SUPERSAMPLE_SIZE)
        duration = (
            1
        )  # transit duration in days. Increase for exotic cases
        t = numpy.linspace(
            -duration * 0.5,
            duration * 0.5,
            SUPERSAMPLE_SIZE,
        )
        ma = batman.TransitParams()
        ma.t0 = 0  # time of inferior conjunction
        ma.per = (
            per
        )  # orbital period, use Earth as a reference
        ma.rp = (
            rp
        )  # planet radius (in units of stellar radii)
        ma.a = (
            a
        )  # semi-major axis (in units of stellar radii)
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
        intransit_flux = flux[idx_first : -idx_first + 1]
        intransit_time = t[idx_first : -idx_first + 1]

        # Downsample (bin) to target sample size
        f = scipy.interpolate.interp1d(
            intransit_time,
            intransit_flux,
            assume_sorted=True,
        )
        xnew = numpy.linspace(
            t[idx_first], t[-idx_first - 1], samples
        )
        downsampled_intransit_flux = f(xnew)

        # Rescale to height [0..1]
        rescaled = (
            numpy.min(downsampled_intransit_flux)
            - downsampled_intransit_flux
        ) / (numpy.min(downsampled_intransit_flux) - 1)

        return rescaled

    def _get_cache(
        self,
        durations,
        maxwidth_in_samples,
        per,
        rp,
        a,
        inc,
        ecc,
        w,
        u,
        limb_dark,
    ):
        """Fetches (size(durations)*size(depths)) light curves of length 
        maxwidth_in_samples and returns these LCs in a 2D array, together with 
        their metadata in a separate array."""

        print(
            "Creating model cache for",
            str(len(durations)),
            "durations",
        )
        lc_arr = []
        rows = numpy.size(durations)
        lc_cache_overview = numpy.zeros(
            rows,
            dtype=[
                ("duration", "f8"),
                ("width_in_samples", "i8"),
                ("overshoot", "f8"),
            ],
        )
        cached_reference_transit = self.reference_transit(
            samples=maxwidth_in_samples,
            per=per,
            rp=rp,
            a=a,
            inc=inc,
            ecc=ecc,
            w=w,
            u=u,
            limb_dark=limb_dark,
        )

        row = 0
        for duration in durations:
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
                cached_reference_transit=cached_reference_transit,
            )
            lc_cache_overview["duration"][row] = duration
            used_samples = int(
                (duration / numpy.max(durations))
                * maxwidth_in_samples
            )
            lc_cache_overview["width_in_samples"][
                row
            ] = used_samples
            cutoff = (
                0.01 * 10 ** -6
            )  # 0.01 ppm tolerance for numerical stability
            full_values = numpy.where(
                scaled_transit < (1 - cutoff)
            )
            first_sample = numpy.min(full_values)
            last_sample = numpy.max(full_values) + 1
            signal = scaled_transit[
                first_sample:last_sample
            ]
            lc_arr.append(signal)

            # Overshoot: Fraction of transit bottom and mean flux
            overshoot = numpy.mean(signal) / numpy.min(
                signal
            )

            # Later, we multiply the inverse fraction ==> convert to inverse percentage
            lc_cache_overview["overshoot"][row] = 1 / (
                2 - overshoot
            )

            row += +1

        lc_arr = numpy.array(lc_arr)
        return lc_cache_overview, lc_arr

    def _search_period(
        self,
        period,
        t,
        y,
        dy,
        transit_depth_min,
        R_star_min,
        R_star_max,
        M_star_min,
        M_star_max,
        lc_arr,
        lc_cache_overview,
        T0_fit_margin,
    ):
        """Core routine to search the flux data set 'injected' over all 'periods'"""

        # duration (in samples) of widest transit in lc_cache (axis 0: rows; axis 1: columns)
        durations = numpy.unique(
            lc_cache_overview["width_in_samples"]
        )
        maxwidth_in_samples = int(
            max(durations)
        )  # * numpy.size(y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1

        # Phase fold
        phases = foldfast(t, period)
        sort_index = numpy.argsort(
            phases, kind="mergesort"
        )  # 8% faster than Quicksort
        phases = phases[sort_index]
        flux = y[sort_index]
        dy = dy[sort_index]

        # faster to multiply than divide
        # SQUARE THESE HERE ALREADY?
        patched_dy = numpy.append(
            dy, dy[:maxwidth_in_samples]
        )
        inverse_squared_patched_dy = 1 / patched_dy ** 2

        # Due to phase folding, the signal could start near the end of the data
        # and continue at the beginning. To avoid (slow) rolling,
        # we patch the beginning again to the end of the data
        patched_data = numpy.append(
            flux, flux[:maxwidth_in_samples]
        )

        # Edge effect correction (numba speedup 40x)
        edge_effect_correction = get_edge_effect_correction(
            flux,
            patched_data,
            dy,
            inverse_squared_patched_dy,
        )
        # Strangely, this second part doesn't work with numba
        summed_edge_effect_correction = numpy.sum(
            edge_effect_correction
        )

        # Set "best of" counters to max, in order to find smaller residuals
        smallest_residuals_in_period = float("inf")
        summed_residual_in_rows = float("inf")

        # Make unique to avoid duplicates in dense grids
        duration_max = T14(
            R_s=R_star_max,
            M_s=M_star_max,
            P=period,
            small=False,
        )
        duration_min = T14(
            R_s=R_star_min,
            M_s=M_star_min,
            P=period,
            small=True,
        )

        # Fractional transit duration can be longer than this.
        # Example: Data length 11 days, 2 transits at 0.5 days and 10.5 days
        length = max(t) - min(t)
        no_of_transits_naive = length / period
        no_of_transits_worst = no_of_transits_naive + 1
        correction_factor = (
            no_of_transits_worst / no_of_transits_naive
        )

        # Minimum can be (x-times) 1 cadence: grazing
        duration_min_in_samples = int(
            floor(duration_min * len(y))
        )  # 1
        duration_max_in_samples = int(
            ceil(duration_max * len(y) * correction_factor)
        )
        durations = durations[
            durations >= duration_min_in_samples
        ]
        durations = durations[
            durations <= duration_max_in_samples
        ]

        skipped_all = True
        best_row = 0  # shortest and shallowest transit
        best_depth = 0

        for duration in durations:
            ootr = ootr_efficient(
                patched_data,
                duration,
                inverse_squared_patched_dy,
            )
            mean = 1 - running_mean(patched_data, duration)

            # Get the row with matching duration
            chosen_transit_row = 0
            while (
                lc_cache_overview["width_in_samples"][
                    chosen_transit_row
                ]
                != duration
            ):
                chosen_transit_row += 1

            overshoot = lc_cache_overview["overshoot"][
                chosen_transit_row
            ]

            this_residual, this_row, this_depth = get_lowest_residuals_in_this_duration(
                mean=mean,
                transit_depth_min=transit_depth_min,
                patched_data_arr=patched_data,
                duration=duration,
                signal=lc_arr[chosen_transit_row],
                inverse_squared_patched_dy_arr=inverse_squared_patched_dy,
                overshoot=overshoot,
                ootr=ootr,
                summed_edge_effect_correction=summed_edge_effect_correction,
                chosen_transit_row=chosen_transit_row,
                datapoints=len(flux),
                T0_fit_margin=T0_fit_margin,
            )

            if this_residual < summed_residual_in_rows:
                summed_residual_in_rows = this_residual
                best_row = chosen_transit_row
                best_depth = this_depth

        return [
            period,
            summed_residual_in_rows,
            best_row,
            best_depth,
        ]

    def power(self, **kwargs):
        """Compute the periodogram for a set of user-defined parameters"""

        print(TLS_VERSION)

        # Validate **kwargs and set to defaults where missing
        self.transit_depth_min = kwargs.get(
            "transit_depth_min", TRANSIT_DEPTH_MIN
        )
        self.R_star = kwargs.get("R_star", R_STAR)
        self.M_star = kwargs.get("M_star", M_STAR)
        self.oversampling_factor = kwargs.get(
            "oversampling_factor", OVERSAMPLING_FACTOR
        )
        self.period_max = kwargs.get(
            "period_max", float("inf")
        )
        self.period_min = kwargs.get("period_min", 0)
        self.n_transits_min = kwargs.get(
            "n_transits_min", N_TRANSITS_MIN
        )

        self.R_star_min = kwargs.get(
            "R_star_min", R_STAR_MIN
        )
        self.R_star_max = kwargs.get(
            "R_star_max", R_STAR_MAX
        )
        self.M_star_min = kwargs.get(
            "M_star_min", M_STAR_MIN
        )
        self.M_star_max = kwargs.get(
            "M_star_max", M_STAR_MAX
        )
        self.duration_grid_step = kwargs.get(
            "duration_grid_step", DURATION_GRID_STEP
        )

        self.per = kwargs.get("per", 13.4)
        self.rp = kwargs.get("rp", (1.42 * R_earth) / R_sun)
        self.a = kwargs.get("a", 23.1)

        self.T0_fit_margin = kwargs.get(
            "T0_fit_margin", T0_FIT_MARGIN
        )

        # If an impact parameter is given, it overrules the supplied inclination
        if "b" in kwargs:
            self.b = kwargs.get("b")
            self.inc = self._impact_to_inclination(
                b=self.b, semimajor_axis=self.a
            )
        else:
            self.inc = kwargs.get("inc", 90)

        self.ecc = kwargs.get("ecc", ECC)
        self.w = kwargs.get("w", W)
        self.u = kwargs.get("u", U)
        self.limb_dark = kwargs.get("limb_dark", LIMB_DARK)

        self.transit_template = kwargs.get(
            "transit_template", "default"
        )
        if self.transit_template == "default":
            self.per = 12.9  # orbital period (in days)
            self.rp = (
                0.03
            )  # planet radius (in units of stellar radii)
            self.a = (
                23.1
            )  # semi-major axis (in units of stellar radii)
            self.inc = 89.21

        elif self.transit_template == "grazing":
            self.b = 0.99  # impact parameter
            self.inc = degrees(arccos(self.b / self.a))

        elif self.transit_template == "box":
            self.per = 29  # orbital period (in days)
            self.rp = (
                11 * R_earth
            ) / R_sun  # planet radius (in units of stellar radii)
            self.a = (
                26.9
            )  # semi-major axis (in units of stellar radii)
            self.b = 0  # impact parameter
            self.inc = 90
            self.u = [0]
            self.limb_dark = "linear"

        else:
            raise ValueError(
                'Unknown transit_template. Known values: \
                "default", "grazing", "box"'
            )

        """Validations to avoid (garbage in ==> garbage out)"""

        # Stellar radius
        # 0 < R_star < inf
        if self.R_star <= 0 or self.R_star >= float("inf"):
            raise ValueError("R_star must be positive")

        # Assert (0 < R_star_min <= R_star)
        if self.R_star_min > self.R_star:
            raise ValueError(
                "R_star_min <= R_star is required"
            )
        if (
            self.R_star_min <= 0
            or self.R_star_min >= float("inf")
        ):
            raise ValueError("R_star_min must be positive")

        # Assert (R_star <= R_star_max < inf)
        if self.R_star_max < self.R_star:
            raise ValueError(
                "R_star_max >= R_star is required"
            )
        if (
            self.R_star_max <= 0
            or self.R_star_max >= float("inf")
        ):
            raise ValueError("R_star_max must be positive")

        # Stellar mass
        # Assert (0 < M_star < inf)
        if self.M_star <= 0 or self.M_star >= float("inf"):
            raise ValueError("M_star must be positive")

        # Assert (0 < M_star_min <= M_star)
        if self.M_star_min > self.M_star:
            raise ValueError(
                "M_star_min <= M_star is required"
            )
        if (
            self.M_star_min <= 0
            or self.M_star_min >= float("inf")
        ):
            raise ValueError("M_star_min must be positive")

        # Assert (M_star <= M_star_max < inf)
        if self.M_star_max < self.M_star:
            raise ValueError(
                "M_star_max >= M_star required"
            )
        if (
            self.M_star_max <= 0
            or self.M_star_max >= float("inf")
        ):
            raise ValueError("M_star_max must be positive")

        # Period grid
        if self.period_min < 0:
            raise ValueError("period_min >= 0 required")
        if self.period_min >= self.period_max:
            raise ValueError(
                "period_min < period_max required"
            )
        if not isinstance(self.n_transits_min, int):
            raise ValueError(
                "n_transits_min must be an integer value"
            )
        if self.n_transits_min < 1:
            raise ValueError(
                "n_transits_min must be an integer value >= 1"
            )

        # Assert 0 < T0_fit_margin < 0.1
        if self.T0_fit_margin < 0:
            self.T0_fit_margin = 0
        elif (
            self.T0_fit_margin > 0.1
        ):  # Sensible limit 10% of transit duration
            self.T0_fit_margin = 0.1

        periods = period_grid(
            R_star=self.R_star,
            M_star=self.M_star,
            time_span=numpy.max(self.t) - numpy.min(self.t),
            period_min=self.period_min,
            period_max=self.period_max,
            oversampling_factor=self.oversampling_factor,
            n_transits_min=self.n_transits_min,
        )

        durations = get_duration_grid(
            periods,
            shortest=1 / len(self.t),
            log_step=self.duration_grid_step,
        )

        maxwidth_in_samples = int(
            numpy.max(durations) * numpy.size(self.y)
        )
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        lc_cache_overview, lc_arr = self._get_cache(
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
        )

        # Result lists now (faster), convert to numpy array later
        test_statistic_periods = []
        test_statistic_residuals = []
        test_statistic_rolls = []
        test_statistic_rows = []
        test_statistic_depths = []

        print(
            "Searching "
            + str(len(self.y))
            + " data points, "
            + str(len(periods))
            + " periods from "
            + str(round(min(periods), 3))
            + " to "
            + str(round(max(periods), 3))
            + " days, using all "
            + str(multiprocessing.cpu_count())
            + " CPU threads"
        )
        p = multiprocessing.Pool(
            processes=multiprocessing.cpu_count()
        )
        params = partial(
            self._search_period,
            t=self.t,
            y=self.y,
            dy=self.dy,
            transit_depth_min=self.transit_depth_min,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=self.M_star_min,
            M_star_max=self.M_star_max,
            lc_arr=lc_arr,
            lc_cache_overview=lc_cache_overview,
            T0_fit_margin=self.T0_fit_margin,
        )
        bar_format = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} periods | {elapsed}<{remaining}"
        pbar = tqdm(
            total=numpy.size(periods),
            smoothing=0.3,
            #mininterval=1,
            bar_format=bar_format,
        )

        if PERIODS_SEARCH_ORDER == "ascending":
            periods = reversed(periods)
        elif PERIODS_SEARCH_ORDER == "descending":
            pass  # it already is
        elif PERIODS_SEARCH_ORDER == "shuffled":
            periods = numpy.random.permutation(periods)
        else:
            raise ValueError("Unknown PERIODS_SEARCH_ORDER")

        for data in p.imap_unordered(params, periods):
            test_statistic_periods.append(data[0])
            test_statistic_residuals.append(data[1])
            test_statistic_rows.append(data[2])
            test_statistic_depths.append(data[3])
            pbar.update(1)
        p.close()
        pbar.close()

        # imap_unordered delivers results in unsorted order ==> sort
        test_statistic_periods = numpy.array(
            test_statistic_periods
        )
        sort_index = numpy.argsort(test_statistic_periods)
        test_statistic_periods = test_statistic_periods[
            sort_index
        ]
        test_statistic_residuals = numpy.array(
            test_statistic_residuals
        )[sort_index]
        test_statistic_rows = numpy.array(
            test_statistic_rows
        )[sort_index]
        test_statistic_depths = numpy.array(
            test_statistic_depths
        )[sort_index]

        idx_best = numpy.argmin(test_statistic_residuals)
        best_row = test_statistic_rows[idx_best]
        duration = lc_cache_overview["duration"][best_row]
        maxwidth_in_samples = int(
            numpy.max(durations) * numpy.size(self.t)
        )

        if max(test_statistic_residuals) == min(
            test_statistic_residuals
        ):
            raise ValueError(
                'No transit were fitted. Try smaller "transit_depth_min"'
            )

        # Power spectra variants
        chi2 = test_statistic_residuals
        chi2red = test_statistic_residuals
        chi2red = chi2red / (len(self.t) - 4)
        chi2_min = numpy.min(chi2)
        chi2red_min = numpy.min(chi2red)
        SR = numpy.min(chi2) / chi2
        SDE_raw = (1 - numpy.mean(SR)) / numpy.std(SR)

        # Scale SDE_power from 0 to SDE_raw
        power_raw = SR - numpy.mean(
            SR
        )  # shift down to the mean being zero
        scale = SDE_raw / numpy.max(
            power_raw
        )  # scale factor to touch max=SDE_raw
        power_raw = power_raw * scale

        # Detrended SDE, named "power"
        kernel = (
            self.oversampling_factor
            * SDE_MEDIAN_KERNEL_SIZE
        )
        if kernel % 2 == 0:
            kernel = kernel + 1
        if len(power_raw) > 2 * kernel:
            my_median = running_median(power_raw, kernel)
            power = power_raw - my_median
            # Re-normalize to range between median = 0 and peak = SDE
            # shift down to the mean being zero
            power = power - numpy.mean(power)
            SDE = numpy.max(power / numpy.std(power))
            # scale factor to touch max=SDE
            scale = SDE / numpy.max(power)
            power = power * scale
        else:
            power = power_raw
            SDE = SDE_raw

        index_highest_power = numpy.argmax(power)
        period = test_statistic_periods[index_highest_power]
        depth = test_statistic_depths[index_highest_power]

        # Determine estimate for uncertainty in period
        # Method: Full width at half maximum
        try:
            # Upper limit
            idx = index_highest_power
            while True:
                idx += 1
                if (
                    power[idx]
                    <= 0.5 * power[index_highest_power]
                ):
                    idx_upper = idx
                    break
            # Lower limit
            idx = index_highest_power
            while True:
                idx -= 1
                if (
                    power[idx]
                    <= 0.5 * power[index_highest_power]
                ):
                    idx_lower = idx
                    break
            period_uncertainty = 0.5 * (
                test_statistic_periods[idx_upper]
                - test_statistic_periods[idx_lower]
            )
        except:
            period_uncertainty = float("inf")

        # Now we know the best period, width and duration. But T0 was not preserved
        # due to speed optimizations. Thus, iterate over T0s using the given parameters
        # Fold to all T0s so that the transit is expected at phase = 0
        signal = lc_arr[best_row]
        dur = len(signal)
        scale = SIGNAL_DEPTH / (1 - depth)
        signal = 1 - ((1 - signal) / scale)
        samples_per_period = numpy.size(self.y)

        if self.T0_fit_margin == 0:
            points = samples_per_period
        else:
            step_factor = self.T0_fit_margin * dur
            points = int(samples_per_period / step_factor)
        if points > samples_per_period:
            points = samples_per_period

        # Create all possible T0s from the start of [t] to [t+period] in [samples] steps
        T0_array = numpy.linspace(
            start=numpy.min(self.t),
            stop=numpy.min(self.t) + period,
            num=points,  # samples_per_period
        )

        # Avoid showing progress bar when expected runtime is short
        if points < PROGRESSBAR_THRESHOLD:
            show_progress_info = False
        else:
            show_progress_info = True

        residuals_lowest = float("inf")
        T0 = 0

        if show_progress_info:
            print(
                "Searching for best T0 for period",
                format(period, ".5f"),
                "days"
                )
            pbar2 = tqdm(total=numpy.size(T0_array))
        signal_ootr = numpy.ones(len(self.y[dur:]))

        # Future speed improvement possible: Add multiprocessing. Will be slower for
        # short data and T0_FIT_MARGIN > 0.01, but faster for large data with dense
        # sampling (T0_FIT_MARGIN=0)
        for Tx in T0_array:
            phases = fold(time=self.t, period=period, T0=Tx)
            sort_index = numpy.argsort(
                phases, kind="mergesort"
            )  # 75% of CPU time
            phases = phases[sort_index]
            flux = self.y[sort_index]
            dy = self.dy[sort_index]

            # Roll so that the signal starts at index 0
            # Numpy roll is slow, so we replace it with less elegant concatenate
            # flux = numpy.roll(flux, roll_cadences)
            # dy = numpy.roll(dy, roll_cadences)
            roll_cadences = int(dur / 2) + 1
            flux = numpy.concatenate(
                [
                    flux[-roll_cadences:],
                    flux[:-roll_cadences],
                ]
            )
            dy = numpy.concatenate(
                [
                    flux[-roll_cadences:],
                    flux[:-roll_cadences],
                ]
            )

            residuals_intransit = numpy.sum(
                (flux[:dur] - signal) ** 2 / dy[:dur] ** 2
            )
            residuals_ootr = numpy.sum(
                (flux[dur:] - signal_ootr) ** 2
                / dy[dur:] ** 2
            )
            residuals_total = (
                residuals_intransit + residuals_ootr
            )

            if show_progress_info:
                pbar2.update(1)
            if residuals_total < residuals_lowest:
                residuals_lowest = residuals_total
                T0 = Tx
        if show_progress_info:
            pbar2.close()

        # Calculate all mid-transit times
        if T0 < min(self.t):
            transit_times = [T0 + period]
        else:
            transit_times = [T0]
        previous_transit_time = transit_times[0]
        transit_number = 0
        while True:
            transit_number = transit_number + 1
            next_transit_time = (
                previous_transit_time + period
            )
            if next_transit_time < (
                numpy.min(self.t)
                + (numpy.max(self.t) - numpy.min(self.t))
            ):
                transit_times.append(next_transit_time)
                previous_transit_time = next_transit_time
            else:
                break

        # Calculate transit duration in days
        duration_timeseries = (
            numpy.max(self.t) - numpy.min(self.t)
        ) / period
        epochs = len(transit_times)
        stretch = duration_timeseries / epochs
        transit_duration_in_days = (
            duration * stretch * period
        )

        # Correct duration for gaps in the data:
        average_cadence = numpy.median(numpy.diff(self.t))
        span = max(self.t) - min(self.t)
        theoretical_cadences = span / average_cadence
        fill_factor = (
            len(self.t) - 1
        ) / theoretical_cadences
        transit_duration_in_days = (
            transit_duration_in_days * fill_factor
        )

        # Folded model / data
        phases = fold(self.t, period, T0=T0 + period / 2)
        sort_index = numpy.argsort(phases)
        folded_phase = phases[sort_index]
        folded_y = self.y[sort_index]
        folded_dy = self.dy[sort_index]

        # Folded model / model curve
        # Data phase 0.5 is not always at the midpoint (not at cadence: len(y)/2),
        # so we need to roll the model to match the model so that its mid-transit
        # is at phase=0.5
        fill_half = 1 - ((1 - fill_factor) * 0.5)

        # Model phase, shifted by half a cadence so that mid-transit is at phase=0.5
        model_folded_phase = numpy.linspace(
            0 + 1 / numpy.size(phases) / 2,
            1 + 1 / numpy.size(phases) / 2,
            numpy.size(phases),
        )

        # Model flux
        model_folded_model = self.fractional_transit(
            duration=duration
            * maxwidth_in_samples
            * fill_half,
            maxwidth=maxwidth_in_samples / stretch,
            depth=1 - depth,
            samples=int(len(self.t / epochs)),
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
        )

        # Light curve model
        oversample = (
            5
        )  # more model data points than real data points
        internal_samples = (
            int(len(self.y) / len(transit_times))
        ) * oversample

        # Append one more transit after and before end of nominal time series
        # to fully cover beginning and end with out of transit calculations
        earlier_tt = transit_times[0] - period
        extended_transit_times = numpy.append(
            earlier_tt, transit_times
        )
        next_tt = transit_times[-1] + period
        extended_transit_times = numpy.append(
            extended_transit_times, next_tt
        )
        full_x_array = numpy.array([])
        full_y_array = numpy.array([])
        rounds = len(extended_transit_times)

        # The model for one period
        y_array = self.fractional_transit(
            duration=(duration * maxwidth_in_samples),
            maxwidth=maxwidth_in_samples / stretch,
            depth=1 - depth,
            samples=internal_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
        )

        # Append all periods
        for i in range(rounds):
            xmin = extended_transit_times[i] - period / 2
            xmax = extended_transit_times[i] + period / 2
            x_array = numpy.linspace(
                xmin, xmax, internal_samples
            )
            full_x_array = numpy.append(
                full_x_array, x_array
            )
            full_y_array = numpy.append(
                full_y_array, y_array
            )

        # Determine start and end of relevant time series, and crop it
        start_cadence = numpy.argmax(
            full_x_array > min(self.t)
        )
        stop_cadence = numpy.argmax(
            full_x_array > max(self.t)
        )
        full_x_array = full_x_array[
            start_cadence:stop_cadence
        ]
        full_y_array = full_y_array[
            start_cadence:stop_cadence
        ]
        model_lightcurve_model = full_y_array
        model_lightcurve_time = full_x_array

        # Get transit depth, standard deviation and SNR per transit
        per_transit_count = numpy.zeros(
            [len(transit_times)]
        )
        transit_depths = numpy.zeros([len(transit_times)])
        snr_per_transit = numpy.zeros([len(transit_times)])
        snr_pink_per_transit = numpy.zeros(
            [len(transit_times)]
        )
        all_flux_intransit = numpy.array([])
        all_idx_intransit = numpy.array([])

        # Depth mean odd and even
        all_flux_intransit_odd = numpy.array([])
        all_flux_intransit_even = numpy.array([])

        for i in range(len(transit_times)):
            mid_transit = transit_times[i]
            tmin = (
                mid_transit - 0.5 * transit_duration_in_days
            )
            tmax = (
                mid_transit + 0.5 * transit_duration_in_days
            )
            idx_intransit = numpy.where(
                numpy.logical_and(
                    self.t > tmin, self.t < tmax
                )
            )
            all_idx_intransit = numpy.append(
                all_idx_intransit, idx_intransit
            )
            flux_intransit = self.y[idx_intransit]
            all_flux_intransit = numpy.append(
                all_flux_intransit, flux_intransit
            )
            mean_flux = numpy.mean(self.y[idx_intransit])
            intransit_points = numpy.size(
                self.y[idx_intransit]
            )
            transit_depths[i] = mean_flux
            per_transit_count[i] = intransit_points
            # Check if transit odd/even to collect the flux for the mean calculations
            if i % 2 == 0:  # even
                all_flux_intransit_even = numpy.append(
                    all_flux_intransit_even, flux_intransit
                )
            else:
                all_flux_intransit_odd = numpy.append(
                    all_flux_intransit_odd, flux_intransit
                )

        flux_ootr = numpy.delete(
            self.y, all_idx_intransit.astype(int)
        )

        # Estimate SNR and pink SNR
        # Second run because now the out of transit points are known
        std = numpy.std(flux_ootr)
        for i in range(
            len(transit_times)
        ):  # REFACTOR for mid_transit in transit_times
            mid_transit = transit_times[i]
            tmin = (
                mid_transit - 0.5 * transit_duration_in_days
            )
            tmax = (
                mid_transit + 0.5 * transit_duration_in_days
            )
            idx_intransit = numpy.where(
                numpy.logical_and(
                    self.t > tmin, self.t < tmax
                )
            )
            all_idx_intransit = numpy.append(
                all_idx_intransit, idx_intransit
            )
            flux_intransit = self.y[idx_intransit]
            all_flux_intransit = numpy.append(
                all_flux_intransit, flux_intransit
            )
            mean_flux = numpy.mean(self.y[idx_intransit])
            intransit_points = numpy.size(
                self.y[idx_intransit]
            )
            try:
                pinknoise = pink_noise(
                    flux_ootr,
                    int(numpy.mean(per_transit_count)),
                )
                snr_pink_per_transit[i] = (
                    1 - mean_flux
                ) / pinknoise
                std_binned = std / intransit_points ** 0.5
                snr_per_transit[i] = (
                    1 - mean_flux
                ) / std_binned
            except:
                snr_per_transit[i] = 0
                snr_pink_per_transit[i] = 0

        depth_mean = numpy.mean(all_flux_intransit)
        depth_mean_std = numpy.std(
            all_flux_intransit
        ) / numpy.sum(per_transit_count) ** (0.5)
        snr = (
            (1 - depth_mean) / numpy.std(flux_ootr)
        ) * len(all_flux_intransit) ** (0.5)

        if self.limb_dark == "quadratic":
            rp_rs = rp_rs_from_depth(
                depth=depth, a=self.u[0], b=self.u[1]
            )
        else:
            rp_rs = None

        depth_mean_odd = numpy.mean(all_flux_intransit_odd)
        depth_mean_even = numpy.mean(
            all_flux_intransit_even
        )
        depth_mean_odd_std = numpy.std(
            all_flux_intransit_odd
        ) / numpy.sum(len(all_flux_intransit_odd)) ** (0.5)
        depth_mean_even_std = numpy.std(
            all_flux_intransit_even
        ) / numpy.sum(len(all_flux_intransit_even)) ** (0.5)

        # Odd even mismatch in standard deviations
        odd_even_difference = abs(
            depth_mean_odd - depth_mean_even
        )
        odd_even_std_sum = (
            depth_mean_odd_std + depth_mean_even_std
        )
        odd_even_mismatch = (
            odd_even_difference / odd_even_std_sum
        )

        transit_count = len(transit_times)
        empty_transit_count = numpy.count_nonzero(
            per_transit_count == 0
        )
        distinct_transit_count = (
            transit_count - empty_transit_count
        )

        duration = transit_duration_in_days

        if empty_transit_count / transit_count >= 0.33:
            text = (
                str(empty_transit_count)
                + " of "
                + str(transit_count)
                + " transits without data. The true period may be twice the given period."
            )
            warnings.warn(text)

        return transitleastsquaresresults(
            SDE,
            SDE_raw,
            chi2_min,
            chi2red_min,
            period,
            period_uncertainty,
            T0,
            duration,
            depth,
            (depth_mean, depth_mean_std),
            (depth_mean_even, depth_mean_even_std),
            (depth_mean_odd, depth_mean_odd_std),
            transit_depths,
            rp_rs,
            snr,
            snr_per_transit,
            snr_pink_per_transit,
            odd_even_mismatch,
            transit_times,
            per_transit_count,
            transit_count,
            distinct_transit_count,
            empty_transit_count,
            FAP(SDE),
            test_statistic_periods,
            power,
            power_raw,
            SR,
            chi2,
            chi2red,
            model_lightcurve_time,
            model_lightcurve_model,
            model_folded_phase,
            folded_y,
            folded_dy,
            folded_phase,
            model_folded_model,
        )


class transitleastsquaresresults(dict):
    """The results of a TransitLeastSquares search"""

    def __init__(self, *args):
        super(transitleastsquaresresults, self).__init__(
            zip(
                (
                    "SDE",
                    "SDE_raw",
                    "chi2_min",
                    "chi2red_min",
                    "period",
                    "period_uncertainty",
                    "T0",
                    "duration",
                    "depth",
                    "depth_mean",
                    "depth_mean_even",
                    "depth_mean_odd",
                    "transit_depths",
                    "rp_rs",
                    "snr",
                    "snr_per_transit",
                    "snr_pink_per_transit",
                    "odd_even_mismatch",
                    "transit_times",
                    "per_transit_count",
                    "transit_count",
                    "distinct_transit_count",
                    "empty_transit_count",
                    "FAP",
                    "periods",
                    "power",
                    "power_raw",
                    "SR",
                    "chi2",
                    "chi2red",
                    "model_lightcurve_time",
                    "model_lightcurve_model",
                    "model_folded_phase",
                    "folded_y",
                    "folded_dy",
                    "folded_phase",
                    "model_folded_model",
                ),
                args,
            )
        )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# This is the command line interface
if __name__ == "__main__":
    print(TLS_VERSION)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lightcurve", help="path to lightcurve file"
    )
    parser.add_argument(
        "-o", "--output", help="path to output directory"
    )
    parser.add_argument(
        "-c", "--config", help="path to configuration file"
    )
    args = parser.parse_args()

    use_config_file = False
    if args.config is not None:
        try:
            config = configparser.ConfigParser()
            config.read(args.config)
            R_star = float(config["Grid"]["R_star"])
            R_star_min = float(config["Grid"]["R_star_min"])
            R_star_max = float(config["Grid"]["R_star_max"])
            M_star = float(config["Grid"]["M_star"])
            M_star_min = float(config["Grid"]["M_star_min"])
            M_star_max = float(config["Grid"]["M_star_max"])
            period_min = float(config["Grid"]["period_min"])
            period_max = float(config["Grid"]["period_max"])
            n_transits_min = int(
                config["Grid"]["n_transits_min"]
            )
            transit_template = config["Template"][
                "transit_template"
            ]
            duration_grid_step = float(
                config["Speed"]["duration_grid_step"]
            )
            transit_depth_min = float(
                config["Speed"]["transit_depth_min"]
            )
            oversampling_factor = int(
                config["Speed"]["oversampling_factor"]
            )
            T0_fit_margin = int(
                config["Speed"]["T0_fit_margin"]
            )
            use_config_file = True
            print(
                "Using TLS configuration from config file",
                args.config,
            )
        except:
            print(
                "Using default values because of broken or missing configuration file",
                args.config,
            )
    else:
        print("No config file given. Using default values")

    data = numpy.genfromtxt(args.lightcurve)
    time = data[:, 0]
    flux = data[:, 1]

    try:
        dy = data[:, 2]
    except:
        dy = numpy.full(len(flux), numpy.std(flux))

    time, flux, dy = cleaned_array(time, flux, dy)

    model = transitleastlquares(time, flux, dy)

    if use_config_file:
        results = model.power(
            R_star=R_star,
            R_star_min=R_star_min,
            R_star_max=R_star_max,
            M_star=M_star,
            M_star_min=M_star_min,
            M_star_max=M_star_max,
            period_min=period_min,
            period_max=period_max,
            n_transits_min=n_transits_min,
            transit_template=transit_template,
            duration_grid_step=duration_grid_step,
            transit_depth_min=transit_depth_min,
            oversampling_factor=oversampling_factor,
            T0_fit_margin=T0_fit_margin,
        )
    else:
        results = model.power()

    if args.output is None:
        target_path_file = "TLS_results.csv"
    else:
        target_path_file = args.output

    try:
        with open(target_path_file, "w") as f:
            for key in results.keys():
                f.write("%s %s\n" % (key, results[key]))
        print("Results saved to", target_path_file)
    except IOError:
        print("Error saving result file")

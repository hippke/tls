"""Numerical core of the TLS period search (reference implementation, numba).

The functions in this module define the TLS test statistic. Alternative search
backends (see ``transitleastsquares.backends``) must reproduce
``search_period`` (validated by tests/test_core_oracle.py).
"""

import numba
import numpy

from transitleastsquares import tls_constants
from transitleastsquares.grid import T14
from transitleastsquares.helpers import running_mean


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def fold(time, period, T0):
    """Normal phase folding"""
    return (time - T0) / period - numpy.floor((time - T0) / period)


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def foldfast(time, period):
    """Fast phase folding with T0=0 hardcoded"""
    return time / period - numpy.floor(time / period)


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def edge_effect_correction(flux, patched_data, dy, inverse_squared_patched_dy):
    """chi2 contribution of the points appended for wrap-around (to be removed)."""
    regular = numpy.sum(((1 - flux) ** 2) * 1 / dy**2)
    patched = numpy.sum(((1 - patched_data) ** 2) * inverse_squared_patched_dy)
    return patched - regular


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def lowest_residuals_in_this_duration(
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
    constant_residual,
    T0_fit_margin,
):
    """Slide the template of one duration over the phase-folded data and return
    (lowest chi2, template row, depth) for this duration.

    For templates wider than 1/T0_fit_margin samples only every
    int(duration * T0_fit_margin)-th phase shift is tested.
    """
    # If nothing is fit, we fit a straight line: signal=1.
    # This gives a chi2 of value constant_residual
    summed_residual_in_rows = constant_residual
    best_row = 0
    best_depth = 0

    xth_point = 1  # How many cadences the template shifts forward in each step
    if T0_fit_margin > 0 and duration > T0_fit_margin:
        T0_fit_margin = 1 / T0_fit_margin
        xth_point = int(duration / T0_fit_margin)
        if xth_point < 1:
            xth_point = 1

    for i in range(len(mean)):
        if (mean[i] > transit_depth_min) and (i % xth_point == 0):
            data = patched_data_arr[i : i + duration]
            dy = inverse_squared_patched_dy_arr[i : i + duration]
            target_depth = mean[i] * overshoot
            scale = tls_constants.SIGNAL_DEPTH / target_depth
            reverse_scale = 1 / scale  # speed: one division now, many mults later

            # Scale model and calculate residuals
            intransit_residual = 0
            for j in range(len(signal)):
                sigi = (1 - signal[j]) * reverse_scale
                intransit_residual += ((data[j] - (1 - sigi)) ** 2) * dy[j]
            current_stat = intransit_residual + ootr[i] - summed_edge_effect_correction
            if current_stat < summed_residual_in_rows:
                summed_residual_in_rows = current_stat
                best_row = chosen_transit_row
                best_depth = 1 - target_depth

    return summed_residual_in_rows, best_row, best_depth


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def out_of_transit_residuals(data, width_signal, dy):
    """chi2 of all points outside a sliding window of width_signal samples"""
    chi2 = numpy.zeros(len(data) - width_signal + 1)
    fullsum = numpy.sum(((1 - data) ** 2) * dy)
    window = numpy.sum(((1 - data[:width_signal]) ** 2) * dy[:width_signal])
    chi2[0] = fullsum - window
    for i in range(1, len(data) - width_signal + 1):
        becomes_visible = i - 1
        becomes_invisible = i - 1 + width_signal
        add_visible_left = (1 - data[becomes_visible]) ** 2 * dy[becomes_visible]
        remove_invisible_right = (1 - data[becomes_invisible]) ** 2 * dy[
            becomes_invisible
        ]
        chi2[i] = chi2[i - 1] + add_visible_left - remove_invisible_right
    return chi2


def search_period(
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
    """Search one trial period: return [period, chi2_min, template row, depth].

    ``T0_fit_margin`` is the phase-shift margin used *during the search*
    (``T0_search_margin`` in ``power()``).
    """
    # Width (in samples) of the widest transit template in the cache
    durations = numpy.unique(lc_cache_overview["width_in_samples"])
    maxwidth_in_samples = int(max(durations))
    if maxwidth_in_samples % 2 != 0:
        maxwidth_in_samples = maxwidth_in_samples + 1

    # Phase fold
    phases = foldfast(t, period)
    sort_index = numpy.argsort(phases, kind="mergesort")  # 8% faster than Quicksort
    flux = y[sort_index]
    dy = dy[sort_index]

    # faster to multiply than divide
    patched_dy = numpy.append(dy, dy[:maxwidth_in_samples])
    inverse_squared_patched_dy = 1 / patched_dy**2

    # Due to phase folding, the signal could start near the end of the data
    # and continue at the beginning. To avoid (slow) rolling,
    # we patch the beginning again to the end of the data
    patched_data = numpy.append(flux, flux[:maxwidth_in_samples])

    this_edge_effect_correction = edge_effect_correction(
        flux, patched_data, dy, inverse_squared_patched_dy
    )

    # Set "best of" counters to max, in order to find smaller residuals
    summed_residual_in_rows = float("inf")

    # Physically plausible duration range for this period
    # M_star_min / M_star_max are the masses paired with R_star_min (shortest
    # duration) and R_star_max (longest), see grid.duration_limit_masses
    duration_max = T14(R_s=R_star_max, M_s=M_star_max, P=period, small=False)
    duration_min = T14(R_s=R_star_min, M_s=M_star_min, P=period, small=True)

    # Fractional transit duration can be longer than this.
    # Example: Data length 11 days, 2 transits at 0.5 days and 10.5 days
    length = numpy.max(t) - numpy.min(t)
    no_of_transits_naive = length / period
    no_of_transits_worst = no_of_transits_naive + 1
    correction_factor = no_of_transits_worst / no_of_transits_naive

    duration_min_in_samples = int(numpy.floor(duration_min * len(y)))
    duration_max_in_samples = int(numpy.ceil(duration_max * len(y) * correction_factor))
    durations = durations[durations >= duration_min_in_samples]
    durations = durations[durations <= duration_max_in_samples]

    best_row = 0  # shortest and shallowest transit
    best_depth = 0
    constant_residual = numpy.sum((flux - 1) ** 2 / dy**2)

    for duration in durations:
        chosen_transit_row = 0
        while lc_cache_overview["width_in_samples"][chosen_transit_row] != duration:
            chosen_transit_row += 1
        this_residual, this_row, this_depth = lowest_residuals_in_this_duration(
            mean=1 - running_mean(patched_data, duration),
            transit_depth_min=transit_depth_min,
            patched_data_arr=patched_data,
            duration=duration,
            signal=lc_arr[chosen_transit_row],
            inverse_squared_patched_dy_arr=inverse_squared_patched_dy,
            overshoot=lc_cache_overview["overshoot"][chosen_transit_row],
            # chi2 outside of the template. The template can be shorter than
            # the window (trimmed near-unity samples of some custom shapes);
            # then the window points beyond the template are out of transit.
            # (TLS <= 1.33 used the window width: these points were dropped.)
            ootr=out_of_transit_residuals(
                patched_data,
                len(lc_arr[chosen_transit_row]),
                inverse_squared_patched_dy,
            ),
            summed_edge_effect_correction=this_edge_effect_correction,
            chosen_transit_row=chosen_transit_row,
            constant_residual=constant_residual,
            T0_fit_margin=T0_fit_margin,
        )

        if this_residual < summed_residual_in_rows:
            summed_residual_in_rows = this_residual
            best_row = chosen_transit_row
            best_depth = this_depth

    return [period, summed_residual_in_rows, best_row, best_depth]

"""Spectra (SR, power, SDE), final T0 fit and post-detection statistics."""

import functools
from os import path

import numba
import numpy
from tqdm import tqdm

from transitleastsquares import tls_constants
from transitleastsquares.core import fold
from transitleastsquares.helpers import running_median, transit_mask


@functools.lru_cache(maxsize=1)
def _fap_table():
    return numpy.genfromtxt(
        path.join(tls_constants.resources_dir, "fap.csv"),
        dtype="f8, f8",
        delimiter=",",
        names=["FAP", "SDE"],
    )


def FAP(SDE):
    """Returns FAP (False Alarm Probability) for a given SDE"""
    data = _fap_table()
    return data["FAP"][numpy.argmax(data["SDE"] > SDE)]


def rp_rs_from_depth(depth, law, params):
    """Takes the maximum transit depth, limb-darkening law and parameters
    Returns R_P / R_S (ratio of planetary to stellar radius)
    Source: Heller 2019, https://arxiv.org/abs/1901.01730"""

    if len(params) == 1:
        params = float(params[0])

    if not isinstance(params, (float, int)) and not all(
        isinstance(x, (float, int)) for x in params
    ):
        raise ValueError("All limb-darkening parameters must be numbers")

    laws = "linear, quadratic, squareroot, logarithmic, nonlinear"
    if law not in laws:
        raise ValueError("Please provide a supported limb-darkening law:", laws)

    if law == "linear" and not isinstance(params, float):
        raise ValueError("Please provide exactly one parameter")

    if law in "quadratic, logarithmic, squareroot" and len(params) != 2:
        raise ValueError("Please provide exactly two limb-darkening parameters")

    if law == "nonlinear" and len(params) != 4:
        raise ValueError("Please provide exactly four limb-darkening parameters")

    if law == "linear":
        return (depth * (1 - params / 3)) ** (1 / 2)
    if law == "quadratic":
        return (depth * (1 - params[0] / 3 - params[1] / 6)) ** (1 / 2)
    if law == "squareroot":
        return (depth * (1 - params[0] / 3 - params[1] / 5)) ** (1 / 2)
    if law == "logarithmic":
        return (depth * (1 + 2 * params[1] / 9 - params[0] / 3)) ** (1 / 2)
    if law == "nonlinear":
        return (
            depth
            * (1 - params[0] / 5 - params[1] / 3 - 3 * params[2] / 7 - params[3] / 2)
        ) ** (1 / 2)


@numba.njit(cache=True)
def _pink_noise(data, width):
    total = 0.0
    datapoints = len(data) - width + 1
    for i in range(datapoints):
        mean = 0.0
        for j in range(i, i + width):
            mean += data[j]
        mean /= width
        var = 0.0
        for j in range(i, i + width):
            var += (data[j] - mean) ** 2
        total += numpy.sqrt(var / width) / width**0.5
    return total / datapoints


def pink_noise(data, width):
    """Mean standard deviation of the mean in sliding windows of `width` points"""
    if width < 1 or len(data) - width + 1 < 1:
        raise ValueError("pink_noise: invalid window")
    return _pink_noise(numpy.ascontiguousarray(data, dtype=float), int(width))


def period_uncertainty(periods, power):
    """Half width at half maximum of the highest peak (inf if not bracketed)"""
    try:
        index_highest_power = numpy.argmax(power)
        half = 0.5 * power[index_highest_power]
        # Upper limit
        idx = index_highest_power
        while True:
            idx += 1
            if power[idx] <= half:
                idx_upper = idx
                break
        # Lower limit (negative indices must not wrap around)
        idx = index_highest_power
        while True:
            idx -= 1
            if idx < 0:
                raise IndexError
            if power[idx] <= half:
                idx_lower = idx
                break
        return 0.5 * (periods[idx_upper] - periods[idx_lower])
    except (IndexError, ValueError):
        return float("inf")


def spectra(chi2, oversampling_factor):
    """SR, power_raw, power (median-detrended), SDE_raw and SDE from chi2"""
    SR = numpy.min(chi2) / chi2
    SDE_raw = (1 - numpy.mean(SR)) / numpy.std(SR)

    # Scale SDE_power from 0 to SDE_raw
    power_raw = SR - numpy.mean(SR)  # shift down to the mean being zero
    scale = SDE_raw / numpy.max(power_raw)  # scale factor to touch max=SDE_raw
    power_raw = power_raw * scale

    # Detrended SDE, named "power"
    kernel = oversampling_factor * tls_constants.SDE_MEDIAN_KERNEL_SIZE
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

    return SR, power_raw, power, SDE_raw, SDE


@numba.njit(cache=True)
def _t0_scan(flux_p, w_p, signal, starts, total):
    """Index of the start with the lowest chi2 (first one in case of ties).
    flux_p/w_p: phase-sorted flux and weights, patched with their first
    len(signal) values (cyclic windows without modulo)."""
    best = numpy.inf
    best_n = 0
    dur = len(signal)
    for n in range(len(starts)):
        g = starts[n]
        acc = 0.0
        for k in range(dur):
            f = flux_p[g + k]
            acc += ((f - signal[k]) ** 2 - (f - 1.0) ** 2) * w_p[g + k]
        value = total + acc
        if value < best:
            best = value
            best_n = n
    return best_n


def _T0_trials(t, n_samples, dur, period, T0_fit_margin):
    if T0_fit_margin == 0:
        points = n_samples
    else:
        points = int(n_samples / (T0_fit_margin * dur))
    points = min(points, n_samples)
    # All trial T0s from the start of [t] to [t+period]
    return numpy.linspace(start=numpy.min(t), stop=numpy.min(t) + period, num=points)


def final_T0_fit(
    signal, depth, t, y, dy, period, T0_fit_margin, show_progress_bar, verbose
):
    """After the search, we know the best period, width and duration.
    But T0 was not preserved due to speed optimizations.
    Thus, iterate over T0s using the given parameters.

    Shifting T0 only rotates the phase circle, so the data are folded and
    sorted once; every trial T0 is a cyclic rotation of that order. The chi2
    outside of the template window follows from the total chi2 of a flat
    model, so each trial costs O(duration) instead of O(N log N).
    """
    dur = len(signal)
    scale = tls_constants.SIGNAL_DEPTH / (1 - depth)
    signal = 1 - ((1 - signal) / scale)
    n = numpy.size(y)
    T0_array = _T0_trials(t, n, dur, period, T0_fit_margin)

    if verbose:
        print("Searching for best T0 for period", format(period, ".5f"), "days")

    # Phases relative to the first trial T0 = min(t); trial Tx = min(t) + c * P
    t_min = numpy.min(t)
    phases = fold(time=t, period=period, T0=t_min)
    order = numpy.argsort(phases, kind="mergesort")
    phases_sorted = phases[order]
    flux = y[order]
    weights = 1 / dy[order] ** 2
    c = (T0_array - t_min) / period
    # Trial Tx: the sorted sequence starts at the first phase >= c (cyclic) and
    # is rolled by int(dur/2)+1 so that the template window starts at index 0
    starts = numpy.searchsorted(phases_sorted, c, side="left")
    starts = (starts - (int(dur / 2) + 1)) % n
    flux_p = numpy.concatenate([flux, flux[:dur]])
    w_p = numpy.concatenate([weights, weights[:dur]])
    total = numpy.sum((flux - 1) ** 2 * weights)
    best = _t0_scan(flux_p, w_p, signal, starts.astype(numpy.int64), total)
    return T0_array[best]


def final_T0_fit_sorting(
    signal, depth, t, y, dy, period, T0_fit_margin, show_progress_bar, verbose
):
    """Reference implementation of the final T0 fit (TLS <= 1.33): re-fold and
    re-sort the data for every trial T0. O(points * N log N). Kept for tests;
    final_T0_fit gives the same result in O(N log N + points * duration)."""

    dur = len(signal)
    scale = tls_constants.SIGNAL_DEPTH / (1 - depth)
    signal = 1 - ((1 - signal) / scale)
    samples_per_period = numpy.size(y)

    if T0_fit_margin == 0:
        points = samples_per_period
    else:
        step_factor = T0_fit_margin * dur
        points = int(samples_per_period / step_factor)
    if points > samples_per_period:
        points = samples_per_period

    # Create all possible T0s from the start of [t] to [t+period] in [samples] steps
    T0_array = numpy.linspace(
        start=numpy.min(t), stop=numpy.min(t) + period, num=points
    )

    # Avoid showing progress bar when expected runtime is short
    show_progress_info = (
        points > tls_constants.PROGRESSBAR_THRESHOLD and show_progress_bar
    )

    residuals_lowest = float("inf")
    T0 = 0

    if verbose:
        print("Searching for best T0 for period", format(period, ".5f"), "days")

    if show_progress_info:
        pbar2 = tqdm(total=numpy.size(T0_array))
    signal_ootr = numpy.ones(len(y[dur:]))
    roll_cadences = int(dur / 2) + 1

    for Tx in T0_array:
        phases = fold(time=t, period=period, T0=Tx)
        sort_index = numpy.argsort(phases, kind="mergesort")  # 75% of CPU time
        flux = y[sort_index]
        dy_sorted = dy[sort_index]

        # Roll so that the signal starts at index 0
        # (numpy.roll is slow, so we use concatenate)
        flux = numpy.concatenate([flux[-roll_cadences:], flux[:-roll_cadences]])
        dy_sorted = numpy.concatenate(
            [dy_sorted[-roll_cadences:], dy_sorted[:-roll_cadences]]
        )

        residuals_intransit = numpy.sum(
            (flux[:dur] - signal) ** 2 / dy_sorted[:dur] ** 2
        )
        residuals_ootr = numpy.sum(
            (flux[dur:] - signal_ootr) ** 2 / dy_sorted[dur:] ** 2
        )
        residuals_total = residuals_intransit + residuals_ootr

        if show_progress_info:
            pbar2.update(1)
        if residuals_total < residuals_lowest:
            residuals_lowest = residuals_total
            T0 = Tx
    if show_progress_info:
        pbar2.close()
    return T0


def model_lightcurve(transit_times, period, t, model_transit_single):
    """Creates the model light curve for the full unfolded dataset"""

    # Append one more transit after and before end of nominal time series
    # to fully cover beginning and end with out of transit calculations
    extended_transit_times = numpy.concatenate(
        [[transit_times[0] - period], transit_times, [transit_times[-1] + period]]
    )
    internal_samples = (
        int(len(t) / len(transit_times))
    ) * tls_constants.OVERSAMPLE_MODEL_LIGHT_CURVE

    full_x_array = numpy.concatenate(
        [
            numpy.linspace(tt - period / 2, tt + period / 2, internal_samples)
            for tt in extended_transit_times
        ]
    )
    full_y_array = numpy.tile(model_transit_single, len(extended_transit_times))

    if numpy.all(numpy.isnan(full_x_array)):
        return None, None
    # Determine start and end of relevant time series, and crop it
    start_cadence = numpy.nanargmax(full_x_array > numpy.min(t))
    stop_cadence = numpy.nanargmax(full_x_array > numpy.max(t))
    return (
        full_y_array[start_cadence:stop_cadence],
        full_x_array[start_cadence:stop_cadence],
    )


def all_transit_times(T0, t, period):
    """Return all mid-transit times within t"""
    t_min, t_max = numpy.min(t), numpy.max(t)
    transit_times = [T0 + period] if T0 < t_min else [T0]
    next_transit_time = transit_times[0] + period
    while next_transit_time < (t_min + (t_max - t_min)):
        transit_times.append(next_transit_time)
        next_transit_time = next_transit_time + period
    return transit_times


def calculate_transit_duration_in_days(t, period, transit_times, duration):
    """Return estimate for transit duration in days"""

    # Difference between (time series duration / period) and epochs
    transit_duration_in_days_raw = (
        duration * calculate_stretch(t, period, transit_times) * period
    )

    # Correct the duration for gaps in the data
    return transit_duration_in_days_raw * calculate_fill_factor(t)


def calculate_stretch(t, period, transit_times):
    """Return difference between (time series duration / period) and epochs
    Example:
    - Time series duration = 100 days
    - Period = 40 days
    - Epochs = 2 at t0s = [30, 70] days
    ==> stretch = (100 / 40) / 2 = 1.25"""

    duration_timeseries = (numpy.max(t) - numpy.min(t)) / period
    return duration_timeseries / len(transit_times)


def calculate_fill_factor(t):
    """Return the fraction of existing cadences, assuming constant cadences"""

    average_cadence = numpy.median(numpy.diff(t))
    span = numpy.max(t) - numpy.min(t)
    theoretical_cadences = span / average_cadence
    return (len(t) - 1) / theoretical_cadences


def count_stats(t, y, transit_times, transit_duration_in_days):
    """Return:
    * in_transit_count:     Number of data points in transit (phase-folded)
    * after_transit_count:  Number of data points in a bin of transit duration,
                            after transit (phase-folded)
    * before_transit_count: Number of data points in a bin of transit duration,
                            before transit (phase-folded)
    """
    in_transit_count = 0
    after_transit_count = 0
    before_transit_count = 0
    t_min, t_max = numpy.min(t), numpy.max(t)

    for mid_transit in transit_times:
        T0 = mid_transit - 1.5 * transit_duration_in_days  # 1 duration before ingress
        T1 = mid_transit - 0.5 * transit_duration_in_days  # start of ingress
        T4 = mid_transit + 0.5 * transit_duration_in_days  # end of egress
        T5 = mid_transit + 1.5 * transit_duration_in_days  # 1 duration after egress

        if T0 > t_min and T5 < t_max:  # inside time
            in_transit_count += numpy.count_nonzero((t > T1) & (t < T4))
            before_transit_count += numpy.count_nonzero((t > T0) & (t < T1))
            after_transit_count += numpy.count_nonzero((t > T4) & (t < T5))

    return in_transit_count, after_transit_count, before_transit_count


def intransit_stats(t, y, transit_times, transit_duration_in_days):
    """Return all intransit odd and even flux points"""

    all_flux_intransit_odd = numpy.array([])
    all_flux_intransit_even = numpy.array([])
    per_transit_count = numpy.zeros([len(transit_times)])
    transit_depths = numpy.zeros([len(transit_times)])
    transit_depths_uncertainties = numpy.zeros([len(transit_times)])
    depth_mean_odd = numpy.nan
    depth_mean_even = numpy.nan
    depth_mean_odd_std = numpy.nan
    depth_mean_even_std = numpy.nan

    for i, mid_transit in enumerate(transit_times):
        tmin = mid_transit - 0.5 * transit_duration_in_days
        tmax = mid_transit + 0.5 * transit_duration_in_days
        if numpy.isnan(tmin) or numpy.isnan(tmax):
            flux_intransit = numpy.array([])
        else:
            flux_intransit = y[(t > tmin) & (t < tmax)]
        intransit_points = numpy.size(flux_intransit)
        if intransit_points > 0:
            transit_depths[i] = numpy.mean(flux_intransit)
            transit_depths_uncertainties[i] = numpy.std(flux_intransit) / numpy.sqrt(
                intransit_points
            )
        else:
            transit_depths[i] = numpy.nan
            transit_depths_uncertainties[i] = numpy.nan
        per_transit_count[i] = intransit_points

        # Odd/even transits: collect the flux for the mean calculations
        if i % 2 == 0:  # even
            all_flux_intransit_even = numpy.append(
                all_flux_intransit_even, flux_intransit
            )
        else:  # odd
            all_flux_intransit_odd = numpy.append(
                all_flux_intransit_odd, flux_intransit
            )

    if len(all_flux_intransit_odd) > 0:
        depth_mean_odd = numpy.mean(all_flux_intransit_odd)
        depth_mean_odd_std = numpy.std(all_flux_intransit_odd) / numpy.sum(
            len(all_flux_intransit_odd)
        ) ** (0.5)
    if len(all_flux_intransit_even) > 0:
        depth_mean_even = numpy.mean(all_flux_intransit_even)
        depth_mean_even_std = numpy.std(all_flux_intransit_even) / numpy.sum(
            len(all_flux_intransit_even)
        ) ** (0.5)

    return (
        depth_mean_odd,
        depth_mean_even,
        depth_mean_odd_std,
        depth_mean_even_std,
        all_flux_intransit_odd,
        all_flux_intransit_even,
        per_transit_count,
        transit_depths,
        transit_depths_uncertainties,
    )


def snr_stats(
    t,
    y,
    period,
    duration,
    T0,
    transit_times,
    transit_duration_in_days,
    per_transit_count,
):
    """Return snr_per_transit and snr_pink_per_transit.
    `duration` is the transit duration in days (used for the out-of-transit mask)."""

    snr_per_transit = numpy.zeros([len(transit_times)])
    snr_pink_per_transit = numpy.zeros([len(transit_times)])
    intransit = transit_mask(t, period, 2 * duration, T0)
    flux_ootr = y[~intransit]

    try:
        pinknoise = pink_noise(flux_ootr, int(numpy.mean(per_transit_count)))
    except Exception:  # e.g. no in-transit points -> int(nan)
        pinknoise = numpy.nan

    # Estimate SNR and pink SNR
    # Second run because now the out of transit points are known
    std = numpy.std(flux_ootr) if len(flux_ootr) > 0 else numpy.nan
    for i, mid_transit in enumerate(transit_times):
        tmin = mid_transit - 0.5 * transit_duration_in_days
        tmax = mid_transit + 0.5 * transit_duration_in_days
        if numpy.isnan(tmin) or numpy.isnan(tmax):
            intransit_points = 0
            mean_flux = numpy.nan
        else:
            flux_in = y[(t > tmin) & (t < tmax)]
            intransit_points = numpy.size(flux_in)
            mean_flux = numpy.mean(flux_in) if intransit_points > 0 else numpy.nan

        try:
            if intransit_points > 0 and not numpy.isnan(std):
                snr_pink_per_transit[i] = (1 - mean_flux) / pinknoise
                std_binned = std / intransit_points**0.5
                snr_per_transit[i] = (1 - mean_flux) / std_binned
            else:
                snr_per_transit[i] = 0
                snr_pink_per_transit[i] = 0
        except ZeroDivisionError:
            snr_per_transit[i] = 0
            snr_pink_per_transit[i] = 0

    return snr_per_transit, snr_pink_per_transit

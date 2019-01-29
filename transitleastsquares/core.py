import numpy
import numba
import transitleastsquares.tls_constants as tls_constants


@numba.jit(fastmath=True, parallel=False, nopython=True)
def fold(time, period, T0):
    """Normal phase folding"""
    return (time - T0) / period - numpy.floor((time - T0) / period)


@numba.jit(fastmath=True, parallel=False, nopython=True)
def foldfast(time, period):
    """Fast phase folding with T0=0 hardcoded"""
    return time / period - numpy.floor(time / period)


@numba.jit(fastmath=True, parallel=False, nopython=True)
def get_edge_effect_correction(flux, patched_data, dy, inverse_squared_patched_dy):
    regular = numpy.sum(((1 - flux) ** 2) * 1 / dy ** 2)
    patched = numpy.sum(((1 - patched_data) ** 2) * inverse_squared_patched_dy)
    return patched - regular


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


@numba.jit(fastmath=True, parallel=False, nopython=True)
def out_of_transit_residuals(data, width_signal, dy):
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

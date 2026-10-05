"""Helper functions (cleaning, masking, running statistics)."""

import numpy
from numpy import arccos, degrees

from transitleastsquares.interpolation import interp1d


def resample(time, flux, factor):
    """Resample (time, flux) to len(flux)/factor equidistant points (linear)."""
    time_grid = int(len(flux) / factor)
    time_resampled = numpy.linspace(min(time), max(time), time_grid)
    f = interp1d(time_resampled, time)
    flux_resampled = f(flux)
    return time_resampled, flux_resampled


def _to_float_array(values):
    """Convert lists/object arrays/masked arrays to float, invalid -> NaN."""
    if numpy.ma.isMaskedArray(values):
        values = numpy.ma.filled(values.astype(object), numpy.nan)
    values = numpy.asarray(values)
    if values.dtype == object:
        values = numpy.array(
            [numpy.nan if v is None else v for v in values.ravel()], dtype=float
        ).reshape(values.shape)
    return numpy.asarray(values, dtype=float)


def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays.

    Rows are removed if t is not finite, or if y (or dy, if given) is not
    finite or not positive. Masked values count as invalid.
    """
    t = _to_float_array(t)
    y = _to_float_array(y)
    valid = numpy.isfinite(t) & numpy.isfinite(y) & (y > 0)
    if dy is None:
        return t[valid], y[valid]
    dy = _to_float_array(dy)
    valid &= numpy.isfinite(dy) & (dy > 0)
    return t[valid], y[valid], dy[valid]


def transit_mask(t, period, duration, T0):
    """Boolean mask, True for points within +-duration/2 of a mid-transit time"""
    return numpy.abs((t - T0 + 0.5 * period) % period - 0.5 * period) < 0.5 * duration


def running_mean(data, width_signal):
    """Returns the running mean in a given window"""
    cumsum = numpy.cumsum(numpy.insert(data, 0, 0))
    return (cumsum[width_signal:] - cumsum[:-width_signal]) / float(width_signal)


def _pad_to_length(med, length):
    """Append the first/last value at the beginning/end to reach `length`"""
    missing_values = length - len(med)
    values_front = int(missing_values * 0.5)
    values_end = missing_values - values_front
    med = numpy.append(numpy.full(values_front, med[0]), med)
    return numpy.append(med, numpy.full(values_end, med[-1]))


def running_mean_equal_length(data, width_signal):
    """Returns the running mean in a given window, same length as data"""
    cumsum = numpy.cumsum(numpy.insert(data, 0, 0))
    med = (cumsum[width_signal:] - cumsum[:-width_signal]) / float(width_signal)
    return _pad_to_length(med, len(data))


def running_median(data, kernel):
    """Returns sliding median of width 'kernel' and same length as data"""
    idx = numpy.arange(kernel) + numpy.arange(len(data) - kernel + 1)[:, None]
    idx = idx.astype(numpy.int64)  # needed if oversampling_factor is not int
    med = numpy.median(data[idx], axis=1)
    return _pad_to_length(med, len(data))


def impact_to_inclination(b, semimajor_axis):
    """Converts planet impact parameter b = [0..1.x] to inclination [deg]"""
    return degrees(arccos(b / semimajor_axis))

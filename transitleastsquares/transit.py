"""Transit templates: reference shape, scaled templates and the template cache."""

import numpy

from transitleastsquares import tls_constants
from transitleastsquares.interpolation import interp1d
from transitleastsquares.transit_model import light_curve


def model_flux(t, per, rp, a, inc, ecc, w, u, limb_dark):
    """Limb-darkened transit light curve at times t (mid-transit at t=0).

    Uses TLS' own transit model (transit_model.py). For comparisons, batman can
    be selected with tls_constants.TRANSIT_MODEL = "batman" (if installed).
    """
    if tls_constants.TRANSIT_MODEL == "batman":
        import batman

        params = batman.TransitParams()
        params.t0, params.per, params.rp, params.a, params.inc = 0, per, rp, a, inc
        params.ecc, params.w, params.u, params.limb_dark = ecc, w, u, limb_dark
        return batman.TransitModel(params, t).light_curve(params)
    if tls_constants.TRANSIT_MODEL != "tls":
        raise ValueError(f"Unknown TRANSIT_MODEL {tls_constants.TRANSIT_MODEL!r}")
    return light_curve(t, 0, per, rp, a, inc, ecc, w, u, limb_dark)


_SUPERSAMPLED = {}  # (parameters) -> (t, flux), see _supersampled_transit
_SUPERSAMPLED_MAX = 8


def _supersampled_transit(per, rp, a, inc, ecc, w, u, limb_dark):
    """The supersampled model transit behind reference_transit (independent
    of `samples`). One power() call needs it for the template cache, each
    pre-binned copy and twice for the statistics; the values are identical,
    so they are computed once per parameter set (small LRU-like dict; the
    arrays are read-only)."""
    try:
        key = (
            per, rp, a, inc, ecc, w,
            tuple(numpy.ravel(numpy.asarray(u, dtype=float)).tolist()),
            limb_dark, tls_constants.TRANSIT_MODEL, tls_constants.SUPERSAMPLE_SIZE,
        )
        hash(key)
    except (TypeError, ValueError):
        key = None
    if key is not None and key in _SUPERSAMPLED:
        return _SUPERSAMPLED[key]
    duration = 1  # time window in days, widened for long transits
    while True:
        t = numpy.linspace(
            -duration * 0.5, duration * 0.5, tls_constants.SUPERSAMPLE_SIZE
        )
        flux = model_flux(t, per, rp, a, inc, ecc, w, u, limb_dark)
        # The transit must not fill the whole window (T14 > window)
        if flux[0] < 1 and duration < 1000:
            duration *= 2
            continue
        break
    if key is not None:
        t.flags.writeable = False
        flux = numpy.asarray(flux)
        flux.flags.writeable = False
        if len(_SUPERSAMPLED) >= _SUPERSAMPLED_MAX:
            _SUPERSAMPLED.pop(next(iter(_SUPERSAMPLED)))
        _SUPERSAMPLED[key] = (t, flux)
    return t, flux


def reference_transit(samples, per, rp, a, inc, ecc, w, u, limb_dark):
    """Returns a transit template of width 1 (first to last contact) and depth 1,
    sampled with `samples` points (1 = nominal flux, 0 = transit bottom)."""
    t, flux = _supersampled_transit(per, rp, a, inc, ecc, w, u, limb_dark)
    if numpy.all(flux >= 1):
        raise ValueError("Transit template parameters yield no transit")

    # Determine start of transit (first value < 1); symmetric in-transit slice
    idx_first = numpy.argmax(flux < 1)
    idx_last = len(flux) - idx_first  # exclusive
    intransit_flux = flux[idx_first:idx_last]
    intransit_time = t[idx_first:idx_last]

    # Downsample (bin) to target sample size
    x_new = numpy.linspace(t[idx_first], t[-idx_first - 1], samples)
    f = interp1d(x_new, intransit_time)
    downsampled_intransit_flux = f(intransit_flux)

    # Rescale to height [0..1]
    rescaled = (numpy.min(downsampled_intransit_flux) - downsampled_intransit_flux) / (
        numpy.min(downsampled_intransit_flux) - 1
    )

    return rescaled


def fractional_transit(
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

    if cached_reference_transit is None:
        reference_flux = reference_transit(
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

    # Interpolate to shorter interval - new method without scipy
    reference_time = numpy.linspace(-0.5, 0.5, samples)
    occupied_samples = int((duration / maxwidth) * samples)
    x_new = numpy.linspace(-0.5, 0.5, occupied_samples)
    f = interp1d(x_new, reference_time)
    y_new = f(reference_flux)

    # Patch ends with ones ("1")
    missing_samples = samples - occupied_samples
    emtpy_segment = numpy.ones(int(missing_samples * 0.5))
    result = numpy.append(emtpy_segment, y_new)
    result = numpy.append(result, emtpy_segment)
    if numpy.size(result) < samples:  # If odd number of samples
        result = numpy.append(result, numpy.ones(1))

    # Depth rescaling
    result = 1 - ((1 - result) * depth)

    return result


def get_cache(
    durations, maxwidth_in_samples, per, rp, a, inc, ecc, w, u, limb_dark, verbose=True
):
    """Create one template per trial duration.

    Returns (lc_cache_overview, lc_arr): a structured array with the fields
    duration, width_in_samples and overshoot per row, and a 1-D object array
    with the (variable length) in-transit part of each template.
    """

    if verbose:
        print("Creating model cache for", str(len(durations)), "durations")
    lc_arr = []
    rows = numpy.size(durations)
    lc_cache_overview = numpy.zeros(
        rows,
        dtype=[("duration", "f8"), ("width_in_samples", "i8"), ("overshoot", "f8")],
    )
    cached_reference_transit = reference_transit(
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
        scaled_transit = fractional_transit(
            duration=duration,
            maxwidth=numpy.max(durations),
            depth=tls_constants.SIGNAL_DEPTH,
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
        used_samples = int((duration / numpy.max(durations)) * maxwidth_in_samples)
        full_values = numpy.where(
            scaled_transit < (1 - tls_constants.NUMERICAL_STABILITY_CUTOFF)
        )
        # Short data sets: the shortest trial durations can be < 1 sample wide
        if used_samples < 1 or numpy.size(full_values) == 0:
            continue
        lc_cache_overview["duration"][row] = duration
        lc_cache_overview["width_in_samples"][row] = used_samples
        first_sample = numpy.min(full_values)
        last_sample = numpy.max(full_values) + 1
        signal = scaled_transit[first_sample:last_sample]
        lc_arr.append(signal)

        # Fraction of transit bottom and mean flux
        overshoot = numpy.mean(signal) / numpy.min(signal)

        # Later, we multiply the inverse fraction ==> convert to inverse percentage
        lc_cache_overview["overshoot"][row] = 1 / (2 - overshoot)
        row += 1

    if row == 0:
        raise ValueError("Too few data points to create any transit template")
    lc_cache_overview = lc_cache_overview[:row]
    # Always a 1-D object array: one (variable length) template per row
    lc_arr_obj = numpy.empty(len(lc_arr), dtype=object)
    for i, signal in enumerate(lc_arr):
        lc_arr_obj[i] = signal
    lc_arr = lc_arr_obj
    return lc_cache_overview, lc_arr

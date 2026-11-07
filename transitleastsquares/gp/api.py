"""Public entry point of the GP transit search."""

import warnings

import numba
import numpy as np

from transitleastsquares import tls_constants
from transitleastsquares.gp.baselines import (
    FIT_BIN_WIDTH,
    Baseline,
    RotationBaseline,
    SplineBaseline,
    fit_with_flare_clipping,
    robust_errors,
)
from transitleastsquares.gp.core import _search
from transitleastsquares.helpers import cleaned_array

BASELINES = {"rotation": RotationBaseline, "spline": SplineBaseline}


class GPSearchResults(dict):
    """The results of gp.search: a dict whose items are also attributes.

    Best signal: period, T0 (first transit time on the grid), duration
    (days), depth (template depth at the bottom, relative flux), depth_err,
    snr (depth / depth_err), transit_times.
    Spectrum: periods, power (SDE spectrum), SDE, SDE_raw, delta_chi2 (best
    chi2 improvement per period), chi2 (= chi2_0 - delta_chi2), chi2_0
    (whitened data, no transit), durations (trial durations).
    Bayes factors (bayes_factor=True, else None): log_bf (per period, summed
    over T0 and durations) and log_bf_global (over the whole grid).
    Baseline: baseline (the fitted or given Baseline), baseline_log_likelihood
    (None for a given baseline), trend (posterior mean of the baseline,
    conditioned on the used points outside the best signal's transits; None
    with trend=False).
    Data (cleaned, sorted by time, aligned with trend): time, flux,
    flux_err, use (False: flares removed by clip_flares, or points of too
    short segments); n_points (used). With segment_gap: flux and flux_err
    normalised per segment and divided by its polynomial, and the trend of
    that flux.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _prepare(t, y, dy):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if dy is None:
        t, y = cleaned_array(t, y)
    else:
        t, y, dy = cleaned_array(t, y, np.asarray(dy, float))
    if t.ndim != 1 or len(t) < 10:
        raise ValueError("Need at least 10 finite data points in 1-dimensional arrays")
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    if np.mean(y) > 1.01 or np.mean(y) < 0.99:
        warnings.warn(f"The mean flux should be normalized to 1, but it was found to be {np.mean(y)}")
    if dy is None:
        dy = np.full(len(y), robust_errors(y))
    else:
        dy = dy[order]
    if np.any(dy <= 0):
        raise ValueError("Flux uncertainties dy must be positive")
    return t, y, dy


SEGMENT_POINTS_MIN = 10  # shorter segments are not searched


def _segments(t, gap):
    """Index ranges (a, b) of the parts of t separated by gaps longer than gap (days)."""
    idx = np.r_[0, np.nonzero(np.diff(t) > gap)[0] + 1, len(t)]
    return list(zip(idx[:-1].tolist(), idx[1:].tolist()))


def _polynomial_flatten(t, y, degree, nsigma=3.0, iters=3):
    """y divided by its Chebyshev polynomial of this degree, fitted with sigma clipping."""
    if len(t) <= degree + 1:
        return y
    x = (t - t.mean()) / (t[-1] - t[0]) * 2
    use = np.ones(len(t), bool)
    for _ in range(iters):
        c = np.polynomial.chebyshev.chebfit(x[use], y[use], degree)
        r = y - np.polynomial.chebyshev.chebval(x, c)
        keep = np.abs(r) < nsigma * 1.4826 * np.median(np.abs(r[use]))
        if keep.sum() <= degree + 1:
            break
        use = keep
    return y / np.polynomial.chebyshev.chebval(x, c)


def _flatten_segments(t, y, dy, segments, degree):
    """Flux and errors, per segment divided by the median and (degree not None) by a polynomial."""
    y, dy = y.copy(), dy.copy()
    for a, b in segments:
        m = np.median(y[a:b])
        y[a:b] /= m
        dy[a:b] /= m
        if degree is not None:
            y[a:b] = _polynomial_flatten(t[a:b], y[a:b], degree)
    return y, dy


def search(
    t,
    y,
    dy=None,
    baseline="rotation",
    *,
    clip_flares=True,
    rotation_period=None,
    n_starts=1,
    fit_method="gradient",
    fit_device=None,
    fit_bin_width=FIT_BIN_WIDTH,
    flare_bin_width=FIT_BIN_WIDTH,
    period_min=0,
    period_max=float("inf"),
    oversampling_factor=tls_constants.OVERSAMPLING_FACTOR,
    duration_grid_step=tls_constants.DURATION_GRID_STEP,
    n_transits_min=tls_constants.N_TRANSITS_MIN,
    grid_transits_min=None,
    R_star_min=tls_constants.R_STAR_MIN,
    R_star_max=tls_constants.R_STAR_MAX,
    M_star_min=tls_constants.M_STAR_MIN,
    M_star_max=tls_constants.M_STAR_MAX,
    duration_min_scale=1.0,
    segment_gap=None,
    segment_degree=4,
    t0_step=None,
    t0_step_duration=0.1,
    bayes_factor=False,
    depth_prior=0.02,
    fold="blocked",
    device="cpu",
    bin_fraction=None,
    trend=True,
    use_threads=None,
    timings=None,
):
    """Periodic transit search with a Gaussian-process baseline marginalised.

    t, y, dy: time (days), flux normalised to ~1, absolute flux uncertainties
        (default: estimated from the point-to-point scatter).
    baseline: "rotation" or "spline" (fitted by maximum likelihood), or a
        Baseline instance (used as given, no fit and no clipping).
    clip_flares: when fitting, remove positive outliers iteratively;
        flare_bin_width: also remove flares found in the residuals binned to
        this width (days, default 10 min) if the cadence is shorter than half
        of it, as most points of a flare are hidden in the noise of a single
        point; None: single points only.
    rotation_period: starting guess of the rotation fit (default:
        Lomb-Scargle peak); n_starts: rotation fit starting points (default
        1; 1-3 with the gradient fit, 1-4 with Nelder-Mead).
    fit_method: optimizer of the rotation fit: "gradient" (default: bounded
        quasi-Newton with the exact gradient of the likelihood; with
        device="gpu" the gradient on the GPU) or "nelder-mead" (the default
        of TLS 2.1: derivative-free, ~30x more likelihood evaluations, and
        mostly lower likelihoods; the results differ slightly). The spline
        baseline (two parameters) is always fitted by Nelder-Mead.
    fit_device: device of the baseline fit, "cpu" or "gpu" (default: device).
        With several searches sharing a consumer GPU (FP64 at 1/64 of its
        FP32 rate), fit_method="gradient" with fit_device="cpu" gives the
        highest throughput without NVIDIA MPS (RTX 5060 and 16 cores, 14
        processes, BLAS limited to one thread each: 0.32 s per Talens light
        curve, against 0.43 s with the fit on the GPU); with MPS, 8 processes
        fitting on the CPU and 8 on the GPU are fastest (0.28 s). A single
        search is fastest with the fit on the GPU (0.62 s against 2.95 s).
    fit_bin_width: the rotation fit uses data binned to this width (days,
        default 10 min) if the cadence is shorter than half of it; the
        search always uses the unbinned data. None: never bin.
    period_min, period_max, oversampling_factor, duration_grid_step,
        n_transits_min, R_star_min, R_star_max, M_star_min, M_star_max: search
        grid, as in TLS.
    grid_transits_min: the longest trial period is the time span divided by
        this (default n_transits_min). Two transits fit into the time span at
        all periods below it, so n_transits_min=2 with grid_transits_min=1
        searches the periods up to the span with at least two transits.
    duration_min_scale: factor on the shortest trial duration of each period.
        The limit of the grid belongs to a central transit; sqrt(1 - b^2)
        extends it to the impact parameter b (0.44: b = 0.9).
    segment_gap: None (default: one filter over the whole light curve), or
        gaps longer than this (days) separate segments, e.g. the quarters of
        a mission with offsets or drifts between them. Each segment is
        normalised by its median and divided by a polynomial of degree
        segment_degree (fitted with sigma clipping; None: no polynomial).
        The baseline is fitted once, on all segments; in the search each
        segment has its own filter and free mean. Not with bayes_factor,
        bin_fraction or t0_step_duration=None.
    t0_step: transit-time grid step (default cadence / 4); t0_step_duration:
        coarser step max(t0_step, t0_step_duration * duration) for long
        durations (default 0.1; None: uniform grid).
    bayes_factor: also compute log Bayes factors (slower); depth_prior: scale
        of the half-normal depth prior (relative flux).
    fold: the period fold of the single events: "blocked" (default, every
        epoch of every period), "ffa" (FFA + polish, approximate: faster on
        long light curves, every value an exact "blocked" value, never above
        it; see gp.fold_ffa), with "-screen" (duration screen) and/or
        "-coarse" (coarse epochs): e.g. "ffa-screen-coarse", or the same
        options without the FFA: "blocked-screen-coarse" (the fast choice
        on the GPU, where the FFA does not pay).
        Bayes factors always use the full fold.
    device: "cpu" (default) or "gpu" (CUDA, FP32 single events and fold, and
        the rotation fit's likelihood on long data; needs numba-cuda; see
        transitleastsquares.gpu).
    bin_fraction: None (default) or e.g. 1/30: the single events of a
        duration D use data binned to <= bin_fraction D (approximate;
        faster for short cadences).
    trend: compute the trend of the result (posterior mean of the baseline
        outside the best signal's transits: one more Kalman pass); False:
        trend = None.
    use_threads: numba threads for the fit and the search (default: numba's
        current setting).
    timings: optional dict, filled with the wall time per stage.

    Returns GPSearchResults.
    """
    t, y, dy = _prepare(t, y, dy)
    segments = None
    if segment_gap is not None:
        segments = _segments(t, segment_gap)
        y, dy = _flatten_segments(t, y, dy, segments, segment_degree)
    old = numba.get_num_threads()
    if use_threads is not None:
        numba.set_num_threads(max(1, min(int(use_threads), numba.config.NUMBA_NUM_THREADS)))
    try:
        if isinstance(baseline, str):
            if baseline not in BASELINES:
                raise ValueError(f"baseline must be one of {sorted(BASELINES)} or a Baseline instance")
            kw = dict(period_guess=rotation_period, n_starts=n_starts, bin_width=fit_bin_width, method=fit_method) \
                if baseline == "rotation" else {}
            if clip_flares:
                b, ll, use = fit_with_flare_clipping(baseline, t, y, dy, device=device, flare_bin=flare_bin_width,
                                                     fit_device=fit_device, **kw)
            else:
                b, ll = BASELINES[baseline].fit(t, y, dy, device=fit_device or device, **kw)
                use = np.ones(len(t), bool)
        elif isinstance(baseline, Baseline):
            b, ll, use = baseline, None, np.ones(len(t), bool)
        else:
            raise TypeError("baseline must be a string or a Baseline instance")
        if segments is not None:
            for a, c in segments:  # (not searched: also not part of the results' used points)
                if use[a:c].sum() < SEGMENT_POINTS_MIN:
                    use[a:c] = False
            if not use.any():
                raise ValueError(f"No segment has at least {SEGMENT_POINTS_MIN} points")
        tu, yu, du = t[use], y[use], dy[use]
        used = None
        if segments is not None:
            n = np.r_[0, np.cumsum(use)]
            used = [(int(n[a]), int(n[c])) for a, c in segments if n[c] > n[a]]
        r = _search(
            tu, yu, du, b._model(tu, du) if used is None else [b._model(tu[a:c], du[a:c]) for a, c in used],
            period_min=period_min, period_max=period_max, oversampling_factor=oversampling_factor,
            duration_grid_step=duration_grid_step, t0_step=t0_step, t0_step_duration=t0_step_duration,
            depth_prior=depth_prior, n_transits_min=n_transits_min, bayes_factor=bayes_factor,
            fold=fold, device=device, bin_fraction=bin_fraction, timings=timings,
            R_star_min=R_star_min, R_star_max=R_star_max, M_star_min=M_star_min, M_star_max=M_star_max,
            duration_min_scale=duration_min_scale, grid_transits_min=grid_transits_min, segments=used,
        )
    finally:
        numba.set_num_threads(old)
    P, T0, D = r["period"], r["T0"], r["duration"]
    transit_times = T0 + P * np.arange(int((t.max() - T0) / P) + 1)
    phase = (t - T0 + 0.5 * P) % P - 0.5 * P
    out = use & (np.abs(phase) > 0.75 * D)
    if not trend:
        trend = None
    elif segments is None:
        trend = b.trend(t, y, dy, out, device=device)
    else:
        trend = np.ones(len(t))
        for a, c in segments:
            if use[a:c].any():
                trend[a:c] = b.trend(t[a:c], y[a:c], dy[a:c], out[a:c], device=device)
    return GPSearchResults(
        period=P,
        T0=T0,
        duration=D,
        depth=r["depth"],
        depth_err=r["depth_err"],
        snr=r["snr"],
        transit_times=transit_times,
        periods=r["periods"],
        power=r["power"],
        SDE=r["SDE"],
        SDE_raw=r["SDE_raw"],
        delta_chi2=r["dchi2"],
        chi2=r["chi2"],
        chi2_0=r["chi2_0"],
        durations=r["durations"],
        log_bf=r["log_bf_per_period"],
        log_bf_global=r["log_bf_global"],
        baseline=b,
        baseline_log_likelihood=ll,
        trend=trend,
        time=t,
        flux=y,
        flux_err=dy,
        use=use,
        n_points=int(use.sum()),
    )

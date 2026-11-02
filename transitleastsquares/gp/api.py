"""Public entry point of the GP transit search."""

import warnings

import numba
import numpy as np

from transitleastsquares import tls_constants
from transitleastsquares.gp.baselines import (
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
    conditioned on the used points outside the best signal's transits).
    Data (cleaned, sorted by time, aligned with trend): time, flux,
    flux_err, use (False: flares removed by clip_flares); n_points (used).
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


def search(
    t,
    y,
    dy=None,
    baseline="rotation",
    *,
    clip_flares=True,
    rotation_period=None,
    n_starts=4,
    period_min=0,
    period_max=float("inf"),
    oversampling_factor=tls_constants.OVERSAMPLING_FACTOR,
    duration_grid_step=tls_constants.DURATION_GRID_STEP,
    n_transits_min=tls_constants.N_TRANSITS_MIN,
    t0_step=None,
    t0_step_duration=0.1,
    bayes_factor=False,
    depth_prior=0.02,
    use_threads=None,
    timings=None,
):
    """Periodic transit search with a Gaussian-process baseline marginalised.

    t, y, dy: time (days), flux normalised to ~1, absolute flux uncertainties
        (default: estimated from the point-to-point scatter).
    baseline: "rotation" or "spline" (fitted by maximum likelihood), or a
        Baseline instance (used as given, no fit and no clipping).
    clip_flares: when fitting, remove positive outliers iteratively.
    rotation_period: starting guess of the rotation fit (default:
        Lomb-Scargle peak); n_starts: rotation fit starting points (1-4).
    period_min, period_max, oversampling_factor, duration_grid_step,
        n_transits_min: search grid, as in TLS (stellar limits: TLS defaults).
    t0_step: transit-time grid step (default cadence / 4); t0_step_duration:
        coarser step max(t0_step, t0_step_duration * duration) for long
        durations (default 0.1; None: uniform grid).
    bayes_factor: also compute log Bayes factors (slower); depth_prior: scale
        of the half-normal depth prior (relative flux).
    use_threads: numba threads for the fit and the search (default: numba's
        current setting).
    timings: optional dict, filled with the wall time per stage.

    Returns GPSearchResults.
    """
    t, y, dy = _prepare(t, y, dy)
    old = numba.get_num_threads()
    if use_threads is not None:
        numba.set_num_threads(max(1, min(int(use_threads), numba.config.NUMBA_NUM_THREADS)))
    try:
        if isinstance(baseline, str):
            if baseline not in BASELINES:
                raise ValueError(f"baseline must be one of {sorted(BASELINES)} or a Baseline instance")
            kw = dict(period_guess=rotation_period, n_starts=n_starts) if baseline == "rotation" else {}
            if clip_flares:
                b, ll, use = fit_with_flare_clipping(baseline, t, y, dy, **kw)
            else:
                b, ll = BASELINES[baseline].fit(t, y, dy, **kw)
                use = np.ones(len(t), bool)
        elif isinstance(baseline, Baseline):
            b, ll, use = baseline, None, np.ones(len(t), bool)
        else:
            raise TypeError("baseline must be a string or a Baseline instance")
        tu, yu, du = t[use], y[use], dy[use]
        r = _search(
            tu, yu, du, b._model(tu, du),
            period_min=period_min, period_max=period_max, oversampling_factor=oversampling_factor,
            duration_grid_step=duration_grid_step, t0_step=t0_step, t0_step_duration=t0_step_duration,
            depth_prior=depth_prior, n_transits_min=n_transits_min, bayes_factor=bayes_factor,
            timings=timings,
        )
    finally:
        numba.set_num_threads(old)
    P, T0, D = r["period"], r["T0"], r["duration"]
    transit_times = T0 + P * np.arange(int((t.max() - T0) / P) + 1)
    phase = (t - T0 + 0.5 * P) % P - 0.5 * P
    trend = b.trend(t, y, dy, use & (np.abs(phase) > 0.75 * D))
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

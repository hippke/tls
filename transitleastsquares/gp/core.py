"""Periodic transit search with the GP baseline marginalised (internal; the
public entry point is transitleastsquares.gp.search)."""

import time

import numba
import numpy as np

from transitleastsquares import tls_constants
from transitleastsquares.gp.events import (
    FOLD_CHUNK,
    compact_offsets,
    fold_groups,
    multiple_events_blocked,
    multiple_events,
    multiple_events_strided,
    single_events,
    single_events_compact,
    tail_operators,
    template_table,
)
from transitleastsquares.gp.fold_ffa import fold_ffa
from transitleastsquares.gp.statespace import kalman_whiten, n_segments, transition_matrices
from transitleastsquares.grid import T14, duration_limit_masses, period_grid
from transitleastsquares.stats import spectra

FOLDS = ("blocked", "ffa", "ffa-screen")


class _Clock:
    """Accumulates wall time per stage into a dict (no-op without one)."""

    def __init__(self, timings):
        self.timings, self.last = timings, time.perf_counter()

    def __call__(self, stage):
        now = time.perf_counter()
        if self.timings is not None:
            self.timings[stage] = self.timings.get(stage, 0.0) + now - self.last
        self.last = now


def _search(
    t,
    flux,
    flux_err,
    model,
    period_min=0.5,
    period_max=35.0,
    oversampling_factor=tls_constants.OVERSAMPLING_FACTOR,
    duration_grid_step=tls_constants.DURATION_GRID_STEP,
    t0_step=None,
    t0_step_duration=0.1,
    texp=None,
    nsub=7,
    depth_prior=0.02,
    n_transits_min=2,
    bayes_factor=False,
    fold="blocked",
    timings=None,
):
    """Periodic transit search with the GP baseline marginalised.

    flux is normalised (median ~ 1); internally in ppt. Returns a dict with
    the TLS-style SDE spectrum and the best signal; with bayes_factor=True
    also the log Bayes factors (per period and summed over the grid).
    t0_step: transit-time grid step (default cadence / 4).
    t0_step_duration: duration D uses the coarser step
    max(t0_step, t0_step_duration * D), rounded down to a multiple of t0_step
    (default 0.1: 2.2x faster, recovery unchanged in the K2-25 injection
    test); None for the uniform grid.
    fold: "blocked", "ffa" or "ffa-screen" (gp.fold_ffa; not with Bayes
    factors or the uniform T0 grid, which use the full fold).
    timings: optional dict, filled with the wall time of each stage."""
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    clock = _Clock(timings)
    t = np.asarray(t, float)
    y = (np.asarray(flux, float) - 1.0) * 1e3
    yerr = np.asarray(flux_err, float) * 1e3
    cad = np.median(np.diff(t))
    texp = cad if texp is None else texp
    delta = cad / 4 if t0_step is None else t0_step

    kinds, starts, pars, P0 = model.blocks()
    R = yerr**2 + model.jitter**2
    nseg = n_segments(len(t), numba.get_num_threads())
    A = transition_matrices(t, kinds, starts, pars, model.dim)
    clock("matrices")
    K, S, ey = kalman_whiten(t, y, R, kinds, starts, pars, P0, nseg)
    nd = model.n_diffuse
    ey[:nd] = 0.0  # diffuse start: no information
    chi2_0 = float(np.sum(ey**2))
    clock("kalman")

    span = t.max() - t.min()
    periods = np.sort(period_grid(1.0, 1.0, span, period_min, period_max, oversampling_factor))
    m_short, m_long = duration_limit_masses(
        tls_constants.R_STAR_MIN, tls_constants.R_STAR_MAX,
        tls_constants.M_STAR_MIN, tls_constants.M_STAR_MAX,
    )
    dmin = np.array([T14(tls_constants.R_STAR_MIN, m_short, P, small=True) * P for P in periods])
    dmax = np.array([T14(tls_constants.R_STAR_MAX, m_long, P, small=False) * P for P in periods])
    durs = [dmin.min()]
    while durs[-1] < dmax.max():
        durs.append(durs[-1] * duration_grid_step)
    durs = np.array(durs)
    jlo = np.clip(np.searchsorted(durs, dmin, side="right") - 1, 0, len(durs) - 1)
    jhi = np.clip(np.searchsorted(durs, dmax, side="left"), 0, len(durs) - 1)

    tc = t.min() + delta * np.arange(int(span / delta) + 1)
    table = template_table()
    clock("grids")
    W, w = tail_operators(A, K, S, model.H, ey, nseg)
    clock("tail_operators")
    strided = t0_step_duration is not None
    if strided:
        mstep = np.maximum(1, np.floor(t0_step_duration * durs / delta)).astype(np.int64)
    else:
        mstep = np.ones(len(durs), np.int64)
    if strided:  # compact layout: per duration only the epochs it uses
        off = compact_offsets(len(tc), mstep)
        ab = single_events_compact(t, ey, A, K, S, model.H, W, w, tc, durs, table, texp, nsub, mstep, off)
    else:
        ab = single_events(t, ey, A, K, S, model.H, W, w, tc, durs, table, texp, nsub)
    clock("single_events")

    s2 = (depth_prior * 1e3) ** 2
    if not strided:
        dchi2, k0, jb, bA, bB, lse, ntrials = multiple_events(
            ab, periods, delta, jlo.astype(np.int64), jhi.astype(np.int64), n_transits_min, s2,
            bayes_factor,
        )
    elif bayes_factor:
        dchi2, k0, jb, bA, bB, lse, ntrials = multiple_events_strided(
            ab, off, len(tc), periods, delta, mstep, jlo.astype(np.int64), jhi.astype(np.int64), n_transits_min, s2,
            bayes_factor,
        )
    else:  # identical results, cache-blocked (see multiple_events_blocked); or FFA + polish
        fold_fn = multiple_events_blocked if fold == "blocked" else fold_ffa
        kw = {} if fold == "blocked" else dict(screen=fold == "ffa-screen")
        dchi2, k0, jb, bA, bB, ntrials = fold_fn(
            ab, off, len(tc), periods, delta, mstep, jlo.astype(np.int64), jhi.astype(np.int64), n_transits_min,
            fold_groups(periods, delta), FOLD_CHUNK, **kw,
        )
        lse = None
    clock("multiple_events")
    chi2 = chi2_0 - dchi2
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor, periods=periods)
    clock("spectra")
    ib = int(np.argmax(power))
    if bayes_factor:
        ok = ntrials > 0
        lse_mean = np.where(ok, lse - np.log(np.maximum(ntrials, 1)), -np.inf)
        log_bf_global = float(np.logaddexp.reduce(lse_mean[ok]) - np.log(ok.sum()))
    else:
        lse_mean, log_bf_global = None, None
    return dict(
        periods=periods,
        power=power,
        dchi2=dchi2,
        chi2=chi2,
        chi2_0=chi2_0,
        n_points=len(t),
        SDE=float(SDE),
        SDE_raw=float(SDE_raw),
        period=float(periods[ib]),
        T0=float(tc[k0[ib]]),
        duration=float(durs[jb[ib]]),
        depth=float(bA[ib] / bB[ib] / 1e3),  # template depth (fraction)
        depth_err=float(1 / np.sqrt(bB[ib]) / 1e3),
        snr=float(bA[ib] / np.sqrt(bB[ib])),
        log_bf_per_period=lse_mean,
        log_bf_global=log_bf_global,
        durations=durs,
        n_events=int(sum(len(tc[::m]) for m in mstep)),
    )

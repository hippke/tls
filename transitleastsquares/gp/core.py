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
    single_events_split,
    numerator_weights,
    tail_operators,
    template_table,
)
from transitleastsquares.gp.fold_ffa import FOLD_NAMES, fold_ffa, fold_options
from transitleastsquares.gp.statespace import kalman_whiten, n_segments, transition_matrices
from transitleastsquares.grid import T14_durations, duration_limit_masses, period_grid
from transitleastsquares.stats import spectra

FOLDS = ("blocked",) + FOLD_NAMES


def _single_strided(single, t, y, R, ey, eyd, A, K, S, W, w, model, nseg, tc, durs, table, texp, nsub, mstep, off):
    """Compact single events of the durations durs on the series t (Kalman or split)."""
    if single == "split":
        z = numerator_weights(ey, S, K, w)
        return single_events_split(t, z, np.zeros(0), A, K, S, model.H, W, tc, durs, table, texp, nsub, mstep, off)
    return single_events_compact(t, ey, A, K, S, model.H, W, w, tc, durs, table, texp, nsub, mstep, off)


def _binned_passes(t, y, yerr, model, width, nseg):
    """The light curve in bins of width (days) and its search passes: (t, y,
    R, ey, eyd, A, K, S, W, w) as in _search. Inverse-variance weighted means
    (weights 1 / (yerr^2 + jitter^2)), bin variance 1 / sum of the weights:
    no information is lost for white noise and a template constant within
    the bin (an unweighted mean loses several % for unequal errors)."""
    wt = 1.0 / (yerr**2 + model.jitter**2)
    k = np.floor((t - t[0]) / width).astype(np.int64)
    _, idx = np.unique(k, return_index=True)
    sw = np.add.reduceat(wt, idx)
    tb = np.add.reduceat(wt * t, idx) / sw
    yb = np.add.reduceat(wt * y, idx) / sw
    Rb = 1.0 / sw
    kinds, starts, pars, P0 = model.blocks()
    Ab = transition_matrices(tb, kinds, starts, pars, model.dim)
    Kb, Sb, eyb = kalman_whiten(tb, yb, Rb, kinds, starts, pars, P0, nseg)
    eydb = eyb[:model.n_diffuse].copy()
    eyb[:model.n_diffuse] = 0.0
    Wb, wb = tail_operators(Ab, Kb, Sb, model.H, eyb, nseg)
    return tb, yb, Rb, eyb, eydb, Ab, Kb, Sb, Wb, wb


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
    single="kalman",
    device="cpu",
    bin_fraction=None,
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
    fold: "blocked" (full fold), or a name of gp.fold_ffa.FOLD_NAMES: "ffa"
    (FFA + polish) or "blocked" with "-screen" (duration screen) and/or
    "-coarse" (coarse epochs in the brute-force part), e.g.
    "ffa-screen-coarse" (not with Bayes factors or the uniform T0 grid,
    which use the full fold).
    timings: optional dict, filled with the wall time of each stage.
    single: single events by the Kalman filter over each template ("kalman")
    or split ("split": the numerator as sum s z with z = C^-1 y from the
    filter and the tail recursion (gp.events.numerator_weights), the
    denominator from a data-free filter; equal up to rounding, only with the
    strided T0 grid; gp.events.single_events_split).
    device: "cpu", or "gpu" (transitleastsquares.gpu: the split single
    events and the fold ("blocked" or "ffa") in FP32 on a CUDA device; the
    Kalman passes and the smoother stay on the CPU in FP64; not with Bayes
    factors or the uniform T0 grid).
    bin_fraction: None (default), or the single events of duration D use the
    data binned to the largest power-of-two multiple k of the cadence with
    k cadence <= bin_fraction D (approximate: the GP on bin means, the
    template averaged over the bin; e.g. 1/30). Strided T0 grid, CPU only."""
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    if single not in ("kalman", "split"):
        raise ValueError('single must be "kalman" or "split"')
    if device not in ("cpu", "gpu"):
        raise ValueError('device must be "cpu" or "gpu"')
    if device == "gpu" and (bayes_factor or t0_step_duration is None):
        raise ValueError('device="gpu" needs the strided T0 grid and no Bayes factors')
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
    if device == "gpu":  # the gains stay on the device (FP32)
        from transitleastsquares.gpu import driver

        dev, ey = driver.kalman_gpu(t, y, R, model, nseg)
    else:
        A = transition_matrices(t, kinds, starts, pars, model.dim)
        clock("matrices")
        K, S, ey = kalman_whiten(t, y, R, kinds, starts, pars, P0, nseg)
    nd = model.n_diffuse
    eyd = ey[:nd].copy()  # (single="split" removes their terms itself)
    ey[:nd] = 0.0  # diffuse start: no information
    chi2_0 = float(np.sum(ey**2))
    clock("kalman")

    span = t.max() - t.min()
    periods = np.sort(period_grid(1.0, 1.0, span, period_min, period_max, oversampling_factor))
    m_short, m_long = duration_limit_masses(
        tls_constants.R_STAR_MIN, tls_constants.R_STAR_MAX,
        tls_constants.M_STAR_MIN, tls_constants.M_STAR_MAX,
    )
    dmin = T14_durations(tls_constants.R_STAR_MIN, m_short, periods, True)
    dmax = T14_durations(tls_constants.R_STAR_MAX, m_long, periods, False)
    durs = [dmin.min()]
    while durs[-1] < dmax.max():
        durs.append(durs[-1] * duration_grid_step)
    durs = np.array(durs)
    jlo = np.clip(np.searchsorted(durs, dmin, side="right") - 1, 0, len(durs) - 1)
    jhi = np.clip(np.searchsorted(durs, dmax, side="left"), 0, len(durs) - 1)

    tc = t.min() + delta * np.arange(int(span / delta) + 1)
    table = template_table()
    clock("grids")
    if device != "gpu":  # (the GPU path computes them on the device)
        W, w = tail_operators(A, K, S, model.H, ey, nseg)
    clock("tail_operators")
    strided = t0_step_duration is not None
    if strided:
        mstep = np.maximum(1, np.floor(t0_step_duration * durs / delta)).astype(np.int64)
    else:
        mstep = np.ones(len(durs), np.int64)
    if device == "gpu":  # split single events and the fold on the device (FP32)
        off = compact_offsets(len(tc), mstep)
        W_d, _, z_d = driver.tail_operators_gpu(dev, ey)  # FP32, on the device
        clock("tail_operators")
        d_ab = driver.single_events_gp_dev(t, z_d, dev, W_d, tc[0], delta, durs, nsub, table, texp, mstep, off)
    elif strided:  # compact layout: per duration only the epochs it uses
        off = compact_offsets(len(tc), mstep)
        kbin = np.ones(len(durs), np.int64)
        if bin_fraction is not None:
            kmax = np.floor(bin_fraction * durs / cad)
            kbin = np.where(kmax >= 2, 2 ** np.floor(np.log2(np.maximum(kmax, 1))), 1).astype(np.int64)
        if np.all(kbin == 1):
            ab = _single_strided(single, t, y, R, ey, eyd, A, K, S, W, w, model, nseg, tc, durs, table, texp, nsub,
                                 mstep, off)
        else:
            ab = np.zeros((off[-1], 2))
            for k in np.unique(kbin):
                js = np.nonzero(kbin == k)[0]
                if k == 1:
                    args = (t, y, R, ey, eyd, A, K, S, W, w, texp)
                else:
                    args = _binned_passes(t, y, yerr, model, k * cad, nseg) + (k * cad,)
                sub_off = compact_offsets(len(tc), mstep[js])
                tb, yb, Rb, eyb, eydb, Ab, Kb, Sb, Wb, wb, texpb = args
                sub = _single_strided(single, tb, yb, Rb, eyb, eydb, Ab, Kb, Sb, Wb, wb, model, nseg, tc, durs[js],
                                      table, texpb, nsub, mstep[js], sub_off)
                for q, j in enumerate(js):
                    ab[off[j]:off[j + 1]] = sub[sub_off[q]:sub_off[q + 1]]
    else:
        ab = single_events(t, ey, A, K, S, model.H, W, w, tc, durs, table, texp, nsub)
    clock("single_events")

    s2 = (depth_prior * 1e3) ** 2
    if device == "gpu":
        dchi2, k0, jb, bA, bB, ntrials = driver.fold(
            d_ab, off, len(tc), periods, delta, mstep, jlo.astype(np.int64), jhi.astype(np.int64), n_transits_min,
            0.0, **fold_options(fold))
        lse = None
    elif not strided:
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
        kw = {} if fold == "blocked" else fold_options(fold)
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

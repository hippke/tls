"""Type-II maximum likelihood fits of the baseline hyperparameters and
iterative flare clipping (internal; data in ppt)."""

from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np

from transitleastsquares.gp.nelder_mead import nelder_mead
from transitleastsquares.gp.statespace import gp_mean, log_likelihood, rotation_model, spline_model

ROT_KEYS = ["sigma", "period", "Q0", "dQ", "f", "sigma_slow", "rho_slow", "jitter"]


# generic starting points (log-parameter shifts of the base guess); the first
# is the original single start, the others are from bench/optimizer_study.py
ROT_START_SHIFTS = [
    np.zeros(8),
    np.log([0.5, 1, 5, 20, 2, 2, 0.5, 2]),
    np.log([2, 1, 0.3, 100, 0.5, 0.5, 3, 0.5]),
    np.log([1, 1, 20, 5, 1, 1, 10, 1]),
]


def _thread_map(fn, items, use_threads):
    """map() over items in up to use_threads Python threads (None: numba's
    current thread count). The work is in GIL-free numba kernels
    (kalman_loglik), so the threads run in parallel."""
    if use_threads is None:
        use_threads = numba.get_num_threads()
    n = min(int(use_threads), len(items))
    if n <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items))


def bin_series(t, y, yerr, width):
    """Mean of t and y in bins of the given width (days); errors of the mean
    sqrt(sum yerr^2) / n; n = points per bin. Empty bins are dropped."""
    k = np.floor((t - t[0]) / width).astype(np.int64)
    _, idx, n = np.unique(k, return_index=True, return_counts=True)
    tb = np.add.reduceat(t, idx) / n
    yb = np.add.reduceat(y, idx) / n
    eb = np.sqrt(np.add.reduceat(yerr**2, idx)) / n
    return tb, yb, eb, n


def fit_rotation(t, y, yerr, period_guess, mask=None, start=None, n_starts=1, use_threads=None,
                 bin_width=None):
    """Type-II maximum likelihood for the rotation model.

    Without start: n_starts (1-4, default 1) generic starting points, each
    with 3 Nelder-Mead passes, run in up to use_threads threads (default:
    numba's thread count); the best optimum wins (the likelihood is
    multimodal, see PERF_PLAN.md). With start (a
    previous solution, e.g. from the last flare-clipping round): a single
    pass from there.

    bin_width (days): fit the likelihood of the data binned to this width
    when the cadence is shorter than half of it (the kernel timescales are
    >= 0.5 d, so the GP is smooth within a bin). The jitter is fitted per bin
    and converted to per point (x sqrt(median points per bin)); the returned
    log likelihood is that of the unbinned data."""
    m = np.ones(len(t), bool) if mask is None else mask
    tm, ym, em = t[m], y[m], yerr[m]
    std = np.std(ym)
    binned = bin_width is not None and np.median(np.diff(tm)) < 0.5 * bin_width
    if binned:
        t_all, y_all, e_all = tm, ym, em
        tm, ym, em, nb = bin_series(tm, ym, em, bin_width)
        jit_scale = np.sqrt(np.median(nb))  # per-point jitter = per-bin jitter x jit_scale
        if start is not None:
            start = dict(start, jitter=start["jitter"] / jit_scale)
    amp_max = 10 * std  # amplitudes and jitter: at most 10x the scatter of the data
    if start is None:
        if not 1 <= n_starts <= len(ROT_START_SHIFTS):
            raise ValueError(f"n_starts must be 1-{len(ROT_START_SHIFTS)}")
        base = np.log([std, period_guess, 1.0, 1.0, 0.5, 0.3 * std, 5.0, np.median(em) * 0.5])
        x0s, passes = [base + sh for sh in ROT_START_SHIFTS[:n_starts]], 3
    else:
        x0s, passes = [np.log([start[k] for k in ROT_KEYS])], 1

    def nll(lx):
        with np.errstate(over="ignore"):  # far-out simplex vertices: penalised below
            p = dict(zip(ROT_KEYS, np.exp(lx)))
        if not (0.3 * period_guess < p["period"] < 3 * period_guess and 0.5 < p["rho_slow"] < 100):
            return 1e25
        # physical bounds: without them a fit can run away (seen once in 434
        # validation light curves: sigma = 378 x the flux, dQ = 1e121)
        if max(p["sigma"], p["sigma_slow"], p["jitter"]) > amp_max or p["Q0"] > 1e4 or p["dQ"] > 1e5 \
                or p["f"] > 100:
            return 1e25
        try:
            v = -log_likelihood(rotation_model(p), tm, ym, em)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return 1e25
        return v if np.isfinite(v) else 1e25

    def run(x0):
        best = None
        for _ in range(passes):  # restarts from the last optimum
            r = nelder_mead(nll, x0 if best is None else best.x, maxiter=4000, xatol=1e-4, fatol=1e-4)
            if best is None or r.fun < best.fun:
                best = r
        return best

    best = min(_thread_map(run, x0s, use_threads), key=lambda r: r.fun)
    p = dict(zip(ROT_KEYS, np.exp(best.x)))
    if not binned:
        return p, -best.fun
    p["jitter"] *= jit_scale
    return p, log_likelihood(rotation_model(p), t_all, y_all, e_all)


def fit_spline(t, y, yerr, mask=None, h_range=(0.05, 5.0), start=None, use_threads=None):
    """Type-II maximum likelihood for the smoothing length h (and jitter):
    3 starts (in up to use_threads threads), or a single pass from start
    (a previous solution)."""
    m = np.ones(len(t), bool) if mask is None else mask
    tm, ym, em = t[m], y[m], yerr[m]
    sig, cad = np.median(yerr), np.median(np.diff(t))

    def nll(lx):
        h, jit = np.exp(lx)
        if not h_range[0] <= h <= h_range[1]:
            return 1e25
        v = -log_likelihood(spline_model(h, sig, cad, jit), tm, ym, em)
        return v if np.isfinite(v) else 1e25

    starts = [np.log([h0, sig]) for h0 in (0.1, 0.3, 1.0)] if start is None else \
        [np.log([start["h"], max(start["jitter"], 1e-6)])]

    def run(x0):
        return nelder_mead(nll, x0, maxiter=2000, xatol=1e-5, fatol=1e-5)

    best = min(_thread_map(run, starts, use_threads), key=lambda r: r.fun)
    h, jit = np.exp(best.x)
    return dict(h=h, jitter=jit, sigma=sig, cadence=cad), -best.fun


def clip_flares(make_model, fit, t, y, yerr, nsigma=4.0, max_iter=5):
    """Iteratively fit the GP and remove positive outliers (flares) of the
    smoothed residuals. Dips (transits) are never clipped. fit(mask, start)
    is warm-started from the previous round's solution."""
    mask = np.ones(len(t), bool)
    params = None
    for _ in range(max_iter):
        params, _ = fit(mask, params)
        model = make_model(params)
        resid = y - gp_mean(model, t, y, yerr)
        mad = 1.4826 * np.median(np.abs(resid[mask] - np.median(resid[mask])))
        new = mask & (resid < nsigma * mad)
        if new.sum() == mask.sum():
            break
        mask = new
    return params, mask

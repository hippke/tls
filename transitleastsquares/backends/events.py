"""Backend "events": the period search as a fold of single transit events.

For white noise, the chi2 gain of a transit template s with a free depth is
a^2 / b, with a = sum s r w and b = sum s^2 w (r = flux - 1, w = 1/dy^2),
summed over the in-transit points. Both sums are additive over the transits
of a periodic signal. So instead of folding all data points for every trial
period (cost ~ N per period), this backend computes a and b once for every
trial duration (in days) and mid-transit time on a time grid, and the period
search only adds them up along T0 + n P (the marginalised GP search,
transitleastsquares.gp, with the GP replaced by white noise). The cost per
period no longer depends on the number of data points, only on the time span
and the T0 grid step: cadence / 4, coarsened to t0_step * duration for long
durations (default 0.1).

Differences to the TLS statistic (approximate backend):
* the depth is the least-squares depth of every trial (TLS: box depth);
* trial durations are a log grid in days (step duration_grid_step = 1.1)
  within the same stellar limits, mapped to the nearest TLS template row for
  the result; T0 on the grid above (TLS: every folded point, coarsened);
* a period needs >= N_TRANSITS_MIN transits with data;
* depths below transit_depth_min are not fit (as in TLS).
The final T0 fit and all statistics are TLS's own, on the unbinned data.
"""

import math
import os

import numba
import numpy

from transitleastsquares import tls_constants
from transitleastsquares.backends import SearchBackend, SearchResult
from transitleastsquares.grid import T14

T0_STEP = 0.1  # T0 grid step as a fraction of the duration (long durations)
TASKS = 256  # parallel tasks of the single events (balanced, see below)
ROWS = 64  # rows per block


@numba.njit(cache=True)
def _template(tt, tc, D, table, texp, nsub):
    """Transit template (negative, depth 1 at the bottom), averaged over the
    exposure time with nsub samples (as gp.events._template)."""
    L = table.shape[0]
    acc = 0.0
    for k in range(nsub):
        off = (k + 0.5) / nsub - 0.5
        u = (tt + off * texp - tc) / D
        if -0.5 < u < 0.5:
            x = (u + 0.5) * (L - 1)
            i = min(int(x), L - 2)
            f = x - i
            acc += table[i] * (1 - f) + table[i + 1] * f
    return -acc / nsub


@numba.njit(cache=True, parallel=True)
def single_events_white(t, r, w, tc, durs, nsubs, table, texp, mstep, off):
    """a = sum s r w and b = sum s^2 w of one transit at tc[kk * mstep[j]]
    with duration durs[j], in the compact layout of gp.events (duration j in
    rows off[j]..off[j+1]). Rows go in blocks round-robin to the tasks: the
    cost of a row grows with its duration."""
    J = len(durs)
    total = off[J]
    ab = numpy.zeros((total, 2))
    nblocks = (total + ROWS - 1) // ROWS
    n = len(t)
    for task in numba.prange(TASKS):
        for blk in range(task, nblocks, TASKS):
            i0 = blk * ROWS
            j = numpy.searchsorted(off, i0, side="right") - 1
            for row in range(i0, min(i0 + ROWS, total)):
                while row >= off[j + 1]:
                    j += 1
                tcm = tc[(row - off[j]) * mstep[j]]
                D = durs[j]
                half = 0.5 * D + 0.5 * texp
                i = numpy.searchsorted(t, tcm - half)
                aa = 0.0
                bb = 0.0
                while i < n and t[i] < tcm + half:
                    s = _template(t[i], tcm, D, table, texp, nsubs[j])
                    aa += s * r[i] * w[i]
                    bb += s * s * w[i]
                    i += 1
                ab[row, 0] = aa
                ab[row, 1] = bb
    return ab


@numba.njit(cache=True, parallel=True)
def fold_events(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, depth_min):
    """Best periodic signal per period: chi2 gain (sum a)^2 / sum b over the
    transits T0 + n P, for depths (sum a / sum b) >= depth_min and >=
    ntr_min transits with data; duration j uses every mstep[j]-th epoch
    (gp.events.multiple_events_strided without the Bayes factor). Returns
    the gain, the best duration index, sum a and sum b per period."""
    NP = len(periods)
    gain = numpy.zeros(NP)
    best_j = numpy.zeros(NP, numpy.int64)
    best_A = numpy.zeros(NP)
    best_B = numpy.ones(NP)
    for p in numba.prange(NP):
        P = periods[p]
        toff = numpy.empty(int(M * delta / P) + 2, numpy.int64)
        for j in range(jlo[p], jhi[p] + 1):
            m = mstep[j]
            dj = m * delta
            Mj = off[j + 1] - off[j]
            o = off[j]
            nph = max(1, int(P / dj))
            ntr = int(Mj * dj / P) + 2
            for nn in range(ntr):
                toff[nn] = int(round(nn * P / dj))
            for kk0 in range(nph):
                sA = 0.0
                sB = 0.0
                sN = 0
                for nn in range(ntr):
                    kk = kk0 + toff[nn]
                    if kk >= Mj:
                        break
                    bb = ab[o + kk, 1]
                    sA += ab[o + kk, 0]
                    sB += bb
                    sN += bb > 0
                if sN < ntr_min or sB <= 0 or sA < depth_min * sB:
                    continue
                g = sA * sA / sB
                if g > gain[p]:
                    gain[p] = g
                    best_j[p] = j
                    best_A[p] = sA
                    best_B[p] = sB
    return gain, best_j, best_A, best_B


def _env_float(name, default):
    return float(os.environ.get(name, default))


class EventsBackend(SearchBackend):
    """Fold of single transit events (approximate; see the module docstring)."""

    name = "events"
    exact = False
    same_statistic = False  # least-squares depth on its own grids, not TLS's statistic

    def search(self, problem, periods, use_threads=1, progress=None):
        from transitleastsquares.gp.events import compact_offsets, template_table

        p = problem
        periods = numpy.ascontiguousarray(periods, dtype=float)
        t = numpy.ascontiguousarray(p.t, dtype=float)
        r = numpy.ascontiguousarray(p.y, dtype=float) - 1.0
        w = 1.0 / numpy.ascontiguousarray(p.dy, dtype=float) ** 2
        chi2_0 = float(numpy.sum(r * r * w))
        cad = float(numpy.median(numpy.diff(t)))
        texp = cad
        delta = cad / float(os.environ.get("TLS_EVENTS_DIV", 4))
        span = t[-1] - t[0]
        dmin = numpy.array([T14(p.R_star_min, p.M_star_min, P, small=True) * P for P in periods])
        dmax = numpy.array([T14(p.R_star_max, p.M_star_max, P, small=False) * P for P in periods])
        step = tls_constants.DURATION_GRID_STEP
        durs = [dmin.min()]
        while durs[-1] < dmax.max():
            durs.append(durs[-1] * step)
        durs = numpy.array(durs)
        jlo = numpy.clip(numpy.searchsorted(durs, dmin, side="right") - 1, 0, len(durs) - 1).astype(numpy.int64)
        jhi = numpy.clip(numpy.searchsorted(durs, dmax, side="left"), 0, len(durs) - 1).astype(numpy.int64)
        f = _env_float("TLS_EVENTS_T0_STEP", T0_STEP)
        mstep = numpy.maximum(1, numpy.floor(f * durs / delta)).astype(numpy.int64)
        # exposure averaging only where the exposure is not short against the duration
        nsubs = numpy.where(texp * float(os.environ.get("TLS_EVENTS_NSUB_RATIO", 30)) > durs, 7, 1).astype(numpy.int64)
        tc = t[0] + delta * numpy.arange(int(span / delta) + 1)
        off = compact_offsets(len(tc), mstep)
        old = numba.get_num_threads()
        numba.set_num_threads(max(1, min(int(use_threads), numba.config.NUMBA_NUM_THREADS)))
        try:
            ab = single_events_white(t, r, w, tc, durs, nsubs, template_table(), texp, mstep, off)
            dchi2, jb, bA, bB = fold_events(
                ab, off, len(tc), periods, delta, mstep, jlo, jhi,
                tls_constants.N_TRANSITS_MIN, max(float(p.transit_depth_min), 0.0),
            )
        finally:
            numba.set_num_threads(old)
        fit = dchi2 > 0
        depth = numpy.where(fit, bA / numpy.where(fit, bB, 1.0), 0.0)
        # nearest TLS template row (durations as fractions of the period)
        fractions = numpy.asarray(p.lc_cache_overview["duration"], float)
        lf = numpy.log(fractions)
        x = numpy.log(durs[jb] / periods)
        k = numpy.clip(numpy.searchsorted(lf, x), 1, len(lf) - 1)
        rows = numpy.where(numpy.abs(x - lf[k - 1]) <= numpy.abs(lf[k] - x), k - 1, k)
        if progress is not None:
            progress(len(periods))
        return SearchResult.from_unsorted(
            periods, chi2_0 - dchi2, rows.astype(numpy.int64), numpy.where(fit, 1.0 - depth, 0.0)
        )

"""Fused, allocation-free period search (numba). Same statistic as core.py.

Differences to the reference implementation (core.search_period), all exact
up to floating point rounding:

* One compiled function per period: fold, sort, patch and all trial durations.
  There is no per-duration numpy work and no per-period Python overhead.
* The running mean (box depth) comes from one cumulative sum per period,
  instead of one `insert` + `cumsum` per duration.
* The out-of-transit chi2 is not computed at all. With a_j = 1 - s_j (template
  profile), k = target_depth / SIGNAL_DEPTH, r = 1 - flux and w = 1/dy^2:
      chi2_i = sum_all r^2 w + k^2 A2_i - 2 k AR_i,
      AR_i = sum_j a_j r_{i+j} w_{i+j},   A2_i = sum_j a_j^2 w_{i+j}.
  The best shift maximises gain_i = 2 k AR_i - k^2 A2_i. If all weights are
  equal (no dy given), A2_i is constant, so a single dot product per shift is
  needed.
* Shifts i >= N repeat shifts i - N exactly when every shift is tested
  (xth_point == 1). They are skipped in that case.
"""

import numba
import numpy

from transitleastsquares import tls_constants
from transitleastsquares.core import foldfast
from transitleastsquares.grid import T14


def flatten_templates(lc_arr, lc_cache_overview):
    """Templates as flat arrays for numba.

    Returns (widths, rows, offsets, lengths, profile, overshoot, sum_a2), where
    widths are the unique template widths (ascending), rows the first cache row
    with each width (as in core.search_period), and profile = 1 - signal.
    """
    width_all = numpy.asarray(lc_cache_overview["width_in_samples"], dtype=numpy.int64)
    widths = numpy.unique(width_all)
    rows = numpy.array(
        [numpy.argmax(width_all == w) for w in widths], dtype=numpy.int64
    )
    lengths = numpy.array([len(lc_arr[r]) for r in rows], dtype=numpy.int64)
    offsets = numpy.zeros(len(rows), dtype=numpy.int64)
    offsets[1:] = numpy.cumsum(lengths)[:-1]
    profile = numpy.concatenate(
        [1 - numpy.asarray(lc_arr[r], dtype=float) for r in rows]
    )
    overshoot = numpy.asarray(lc_cache_overview["overshoot"], dtype=float)[rows]
    sum_a2 = numpy.array(
        [numpy.sum(profile[o : o + n] ** 2) for o, n in zip(offsets, lengths)]
    )
    return widths, rows, offsets, lengths, profile, overshoot, sum_a2


@numba.njit(cache=True)
def fold_sort(t, y, inv_dy2, period):
    """Phase-fold and return (flux, weights) sorted by phase (stable order).

    Bucket sort: phases are near-uniform in [0, 1), so N buckets hold about
    one point each. A stable counting scatter followed by an insertion sort
    gives exactly the stable (mergesort) order in O(N); 2-4x faster than
    numpy/numba mergesort (scratch/natsort.py). Falls back to mergesort if the
    phases are strongly clustered (insertion work > 8 N).
    """
    n = len(t)
    # Same (fastmath) folding as the reference core.search_period
    phases = foldfast(t, period)
    counts = numpy.zeros(n + 1, dtype=numpy.int64)
    bucket = numpy.empty(n, dtype=numpy.int64)
    for i in range(n):
        k = int(phases[i] * n)
        if k >= n:
            k = n - 1
        elif k < 0:
            k = 0
        bucket[i] = k
        counts[k + 1] += 1
    for k in range(n):
        counts[k + 1] += counts[k]
    keys = numpy.empty(n)
    flux = numpy.empty(n)
    w = numpy.empty(n)
    for i in range(n):
        pos = counts[bucket[i]]
        counts[bucket[i]] = pos + 1
        keys[pos] = phases[i]
        flux[pos] = y[i]
        w[pos] = inv_dy2[i]
    work = 0
    for i in range(1, n):
        kv = keys[i]
        if kv < keys[i - 1]:
            fv = flux[i]
            wv = w[i]
            j = i - 1
            while j >= 0 and keys[j] > kv:
                keys[j + 1] = keys[j]
                flux[j + 1] = flux[j]
                w[j + 1] = w[j]
                j -= 1
            keys[j + 1] = kv
            flux[j + 1] = fv
            w[j + 1] = wv
            work += i - 1 - j
            if work > 8 * n:  # clustered phases: use the O(N log N) sort
                order = numpy.argsort(phases, kind="mergesort")
                for q in range(n):
                    flux[q] = y[order[q]]
                    w[q] = inv_dy2[order[q]]
                return flux, w
    return flux, w


# The dot products take array *views* (slices). Indexing the full arrays with
# rw[i + j] inside a loop over i with a variable step prevents LLVM from
# vectorising the inner loop (4x slower, measured).
@numba.njit(fastmath=True, cache=True)
def _dot1(a, r):
    acc = 0.0
    for j in range(len(a)):
        acc += a[j] * r[j]
    return acc


@numba.njit(fastmath=True, cache=True)
def _dot2(a, r, w):
    ar = 0.0
    a2 = 0.0
    for j in range(len(a)):
        aj = a[j]
        ar += aj * r[j]
        a2 += aj * aj * w[j]
    return ar, a2


@numba.njit(fastmath=True, cache=True)
def search_period_fused(
    period,
    t,
    y,
    inv_dy2,
    uniform_weights,
    time_span,
    transit_depth_min,
    R_star_min,
    R_star_max,
    M_star_min,
    M_star_max,
    widths,
    rows,
    offsets,
    lengths,
    profile,
    overshoot,
    sum_a2,
    T0_search_margin,
    signal_depth,
    prune=True,
):
    """Returns (chi2_min, template row, depth) for one trial period.

    prune: skip shifts whose gain provably cannot beat the best gain so far.
    With R2 = sum r^2 w over the window and Cauchy-Schwarz AR <= sqrt(A2 R2),
    gain <= 2 k sqrt(A2 R2) - k^2 A2 <= R2. (Exact: never changes the result.)
    """
    n = len(t)
    maxw = widths[-1]
    if maxw % 2 != 0:
        maxw += 1

    # Phase fold and sort (stable order, as in the reference)
    flux_sorted, w_sorted = fold_sort(t, y, inv_dy2, period)

    m = n + maxw
    flux = numpy.empty(m)
    w = numpy.empty(m)
    rw = numpy.empty(m)
    cum = numpy.empty(m + 1)
    cum_r2 = numpy.empty(m + 1)
    cum_w = numpy.empty(m + 1)
    cum[0] = 0.0
    cum_r2[0] = 0.0
    cum_w[0] = 0.0
    total = 0.0
    for k in range(m):
        src = k if k < n else k - n
        f = flux_sorted[src]
        wk = w_sorted[src]
        flux[k] = f
        w[k] = wk
        rw[k] = (1.0 - f) * wk
        cum[k + 1] = cum[k] + f
        cum_r2[k + 1] = cum_r2[k] + (1.0 - f) * (1.0 - f) * wk
        cum_w[k + 1] = cum_w[k] + wk
        if k < n:
            total += (f - 1.0) ** 2 * wk

    # Physically plausible template widths for this period
    duration_max = T14(R_s=R_star_max, M_s=M_star_max, P=period, small=False)
    duration_min = T14(R_s=R_star_min, M_s=M_star_min, P=period, small=True)
    transits_naive = time_span / period
    correction_factor = (transits_naive + 1) / transits_naive
    width_min = int(numpy.floor(duration_min * n))
    width_max = int(numpy.ceil(duration_max * n * correction_factor))

    best_gain = 0.0
    best_row = 0
    best_depth = 0.0
    w0 = inv_dy2[0]

    # Seed for pruning: the gain at the deepest box position of each duration.
    # These shifts are evaluated again in the main loop, so the result is
    # unchanged (barring exact ties); the seed only lets pruning start early.
    seed = 0.0
    if prune:
        for u in range(len(widths)):
            d = widths[u]
            if d < width_min or d > width_max:
                continue
            xth = 1
            if T0_search_margin > 0 and d > T0_search_margin:
                xth = max(1, int(d / (1 / T0_search_margin)))
            n_shifts = min(m - d + 1, n) if xth == 1 else m - d + 1
            best_i = -1
            best_mean = transit_depth_min
            for i in range(0, n_shifts, xth):
                mean = 1 - (cum[i + d] - cum[i]) / d
                if mean > best_mean:
                    best_mean = mean
                    best_i = i
            if best_i < 0:
                continue
            length = lengths[u]
            prof = profile[offsets[u] : offsets[u] + length]
            k = 1 / (signal_depth / (best_mean * overshoot[u]))
            ar, a2 = _dot2(
                prof, rw[best_i : best_i + length], w[best_i : best_i + length]
            )
            gain = 2 * k * ar - k * k * a2
            if gain > seed:
                seed = gain
    # Shifts with gain <= seed (minus rounding margin) cannot be the result
    threshold = seed * (1 - 1e-9)

    for u in range(len(widths)):
        d = widths[u]
        if d < width_min or d > width_max:
            continue
        row = rows[u]
        off = offsets[u]
        length = lengths[u]
        ov = overshoot[u]
        a2_const = sum_a2[u] * w0
        prof = profile[off : off + length]
        amax2 = 0.0
        for j in range(length):
            if prof[j] * prof[j] > amax2:
                amax2 = prof[j] * prof[j]

        xth = 1
        if T0_search_margin > 0 and d > T0_search_margin:
            xth = int(d / (1 / T0_search_margin))
            if xth < 1:
                xth = 1
        n_shifts = m - d + 1
        if xth == 1 and n_shifts > n:
            n_shifts = n  # shifts >= n repeat shifts already tested

        for i in range(0, n_shifts, xth):
            mean = 1 - (cum[i + d] - cum[i]) / d
            if mean > transit_depth_min:
                target_depth = mean * ov
                k = 1 / (signal_depth / target_depth)
                if prune:
                    r2 = cum_r2[i + length] - cum_r2[i]
                    if uniform_weights:  # A2 known exactly
                        bound = 2 * k * numpy.sqrt(a2_const * r2) - k * k * a2_const
                    else:  # 0 <= A2 <= a2b; maximum of the concave bound
                        a2b = amax2 * (cum_w[i + length] - cum_w[i])
                        if a2b * k * k >= r2:
                            bound = r2
                        else:
                            bound = 2 * k * numpy.sqrt(a2b * r2) - k * k * a2b
                    if bound * (1 + 1e-9) <= best_gain or bound < threshold:
                        continue
                if uniform_weights:
                    ar = _dot1(prof, rw[i : i + length])
                    a2 = a2_const
                else:
                    ar, a2 = _dot2(prof, rw[i : i + length], w[i : i + length])
                gain = 2 * k * ar - k * k * a2
                if gain > best_gain:
                    best_gain = gain
                    best_row = row
                    best_depth = 1 - target_depth

    return total - best_gain, best_row, best_depth


class FusedProblem:
    """Precomputed, numba-friendly view of a backends.SearchProblem."""

    def __init__(self, problem):
        self.t = numpy.ascontiguousarray(problem.t, dtype=float)
        self.y = numpy.ascontiguousarray(problem.y, dtype=float)
        dy = numpy.asarray(problem.dy, dtype=float)
        self.inv_dy2 = 1 / dy**2
        self.uniform_weights = bool(numpy.all(self.inv_dy2 == self.inv_dy2[0]))
        self.time_span = float(numpy.max(self.t) - numpy.min(self.t))
        self.templates = flatten_templates(problem.lc_arr, problem.lc_cache_overview)
        self.problem = problem
        self.prune = True

    def search(self, period):
        p = self.problem
        chi2, row, depth = search_period_fused(
            float(period),
            self.t,
            self.y,
            self.inv_dy2,
            self.uniform_weights,
            self.time_span,
            p.transit_depth_min,
            p.R_star_min,
            p.R_star_max,
            p.M_star_min,
            p.M_star_max,
            *self.templates,
            float(p.T0_search_margin),
            float(tls_constants.SIGNAL_DEPTH),
            self.prune,
        )
        return period, chi2, row, depth


@numba.njit(parallel=True, cache=True)
def search_periods_fused_parallel(
    periods,
    t,
    y,
    inv_dy2,
    uniform_weights,
    time_span,
    transit_depth_min,
    R_star_min,
    R_star_max,
    M_star_min,
    M_star_max,
    widths,
    rows,
    offsets,
    lengths,
    profile,
    overshoot,
    sum_a2,
    T0_search_margin,
    signal_depth,
    prune,
):
    """search_period_fused for many periods, numba threads (prange)."""
    n_p = len(periods)
    chi2 = numpy.empty(n_p)
    row = numpy.empty(n_p, dtype=numpy.int64)
    depth = numpy.empty(n_p)
    for p in numba.prange(n_p):
        c, r, d = search_period_fused(
            periods[p],
            t,
            y,
            inv_dy2,
            uniform_weights,
            time_span,
            transit_depth_min,
            R_star_min,
            R_star_max,
            M_star_min,
            M_star_max,
            widths,
            rows,
            offsets,
            lengths,
            profile,
            overshoot,
            sum_a2,
            T0_search_margin,
            signal_depth,
            prune,
        )
        chi2[p] = c
        row[p] = r
        depth[p] = d
    return chi2, row, depth

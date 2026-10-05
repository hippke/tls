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


@numba.njit(fastmath=True, cache=True)
def _foldfast_into(t, period, out):
    """Same (fastmath) expression as core.foldfast, without allocation."""
    for i in range(len(t)):
        out[i] = t[i] / period - numpy.floor(t[i] / period)


@numba.njit(cache=True)
def fold_sort_into(t, y, inv_dy2, period, phases, counts, bucket, keys, flux, w):
    """Phase-fold and write flux and weights sorted by phase (stable order)
    into `flux` and `w` (work arrays: phases, keys: n; counts: n+1; bucket: n).

    Bucket sort: phases are near-uniform in [0, 1), so N buckets hold about
    one point each. A stable counting scatter followed by an insertion sort
    gives exactly the stable (mergesort) order in O(N); 2-4x faster than
    numpy/numba mergesort (scratch/natsort.py). Falls back to mergesort if the
    phases are strongly clustered (insertion work > 8 N).
    """
    n = len(t)
    _foldfast_into(t, period, phases)  # same folding as core.search_period
    for k in range(n + 1):
        counts[k] = 0
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
                order = numpy.argsort(phases[:n], kind="mergesort")
                for q in range(n):
                    flux[q] = y[order[q]]
                    w[q] = inv_dy2[order[q]]
                return


@numba.njit(cache=True)
def fold_sort(t, y, inv_dy2, period):
    """Phase-fold; return (flux, weights) sorted by phase (stable order)."""
    n = len(t)
    flux = numpy.empty(n)
    w = numpy.empty(n)
    fold_sort_into(
        t,
        y,
        inv_dy2,
        period,
        numpy.empty(n),
        numpy.empty(n + 1, dtype=numpy.int64),
        numpy.empty(n, dtype=numpy.int64),
        numpy.empty(n),
        flux,
        w,
    )
    return flux, w


def workspace_size(n, maxw):
    """Sizes (float64, int64, value buffer) of the work buffers of
    search_period_fused."""
    m = n + maxw + 1
    return 4 * n + 4 * m + 8, 2 * n + 2, 6 * m


# Pruning pays off only if the dot product is much more expensive than the bound
PRUNE_MIN_LENGTH = 48


# The dot products take array *views* (slices). Indexing the full arrays with
# rw[i + j] inside a loop over i with a variable step prevents LLVM from
# vectorising the inner loop (4x slower, measured).
# The accumulators have the dtype of the inputs (a[0] * 0): float32 inputs are
# summed in float32 (2x SIMD width, ~1e-7 relative error), float64 in float64.
@numba.njit(fastmath=True, cache=True)
def _dot1(a, r):
    if len(a) == 0:
        return 0.0
    acc = a[0] * 0
    for j in range(len(a)):
        acc += a[j] * r[j]
    return float(acc)


@numba.njit(fastmath=True, cache=True)
def _dot2(a, r, w):
    if len(a) == 0:
        return 0.0, 0.0
    ar = a[0] * 0
    a2 = a[0] * 0
    for j in range(len(a)):
        aj = a[j]
        ar += aj * r[j]
        a2 += aj * aj * w[j]
    return float(ar), float(a2)


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
    prune,
    nbins,
    bsize,
    boff,
    a_bin,
    a2_bin,
    fbuf,
    ibuf,
    vbuf,
):
    """Returns (chi2_min, template row, depth) for one trial period.

    fbuf/ibuf: preallocated work buffers (see workspace_size). Reusing them
    across periods avoids ~1 ms of page faults per period for N ~ 1e5.

    nbins/boff/a_bin/a2_bin: stride-binned templates (see bin_templates). With
    nbins[u] == 0 (default, exact) the full-resolution correlation is used;
    otherwise the approximate stride-binned correlation (idea B1).

    prune: skip shifts whose gain provably cannot beat the best gain so far.
    With R2 = sum r^2 w over the window and Cauchy-Schwarz AR <= sqrt(A2 R2),
    gain <= 2 k sqrt(A2 R2) - k^2 A2 <= R2. (Exact: never changes the result.)
    """
    n = len(t)
    maxw = widths[-1]
    if maxw % 2 != 0:
        maxw += 1

    m = n + maxw
    # Work arrays: views into the preallocated buffers
    o = 0
    phases = fbuf[o : o + n]
    o += n
    keys = fbuf[o : o + n]
    o += n
    flux_sorted = fbuf[o : o + n]
    o += n
    w_sorted = fbuf[o : o + n]
    o += n
    cum = fbuf[o : o + m + 1]
    o += m + 1
    cum_r2 = fbuf[o : o + m + 1]
    o += m + 1
    cum_w = fbuf[o : o + m + 1]
    o += m + 1
    cum_rw = fbuf[o : o + m + 1]
    o += m + 1
    # values entering the dot products: dtype of vbuf (float64 or float32)
    w = vbuf[:m]
    rw = vbuf[m : 2 * m]
    brw_buf = vbuf[2 * m : 4 * m]
    bw_buf = vbuf[4 * m : 6 * m]
    counts = ibuf[: n + 1]
    bucket = ibuf[n + 1 : 2 * n + 1]

    # Phase fold and sort (stable order, as in the reference)
    fold_sort_into(
        t, y, inv_dy2, period, phases, counts, bucket, keys, flux_sorted, w_sorted
    )
    cum_rw[0] = 0.0
    cum[0] = 0.0
    cum_r2[0] = 0.0
    cum_w[0] = 0.0
    total = 0.0
    for k in range(m):
        src = k if k < n else k - n
        f = flux_sorted[src]
        wk = w_sorted[src]
        w[k] = wk
        rw[k] = (1.0 - f) * wk
        cum[k + 1] = cum[k] + f
        cum_r2[k + 1] = cum_r2[k] + (1.0 - f) * (1.0 - f) * wk
        cum_w[k + 1] = cum_w[k] + wk
        cum_rw[k + 1] = cum_rw[k] + (1.0 - f) * wk
        if k < n:
            total += (f - 1.0) ** 2 * wk

    # Physically plausible template widths for this period
    # M_star_min / M_star_max are the masses paired with R_star_min (shortest
    # duration) and R_star_max (longest), see grid.duration_limit_masses
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
        # a few durations suffice (evenly spaced among the allowed long ones)
        n_long = 0
        for u in range(len(widths)):
            if width_min <= widths[u] <= width_max and lengths[u] >= PRUNE_MIN_LENGTH:
                n_long += 1
        every = max(1, n_long // 8)
        c = -1
        for u in range(len(widths)):
            d = widths[u]
            if d < width_min or d > width_max or lengths[u] < PRUNE_MIN_LENGTH:
                continue
            c += 1
            if c % every != 0:
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

        # Stride-binned correlation (approximate, only if nbins[u] > 0)
        nb = nbins[u]
        brw = rw[:0]
        bw = w[:0]
        ab = a_bin[:0]
        a2bin = a2_bin[:0]
        rem0 = 0
        prem = prof
        bs = bsize[u]
        q_row = 0  # bins per row of the (decimated) data bin table
        multirow = False
        if nb > 0:
            # Data bin sums of size bs. Row r holds the bins starting at
            # r, r + bs, r + 2 bs, ... If all shifts are multiples of bs
            # (aligned case, B1) only row 0 is needed.
            multirow = xth % bs != 0
            n_rows = bs if multirow else 1
            q_row = m // bs + 1
            for r in range(n_rows):
                base = r * q_row
                for kb in range(q_row):
                    lo = r + kb * bs
                    hi = lo + bs
                    if hi > m:
                        break
                    brw_buf[base + kb] = cum_rw[hi] - cum_rw[lo]
                    bw_buf[base + kb] = cum_w[hi] - cum_w[lo]
            brw = brw_buf
            bw = bw_buf
            ab = a_bin[boff[u] : boff[u] + nb]
            a2bin = a2_bin[boff[u] : boff[u] + nb]
            rem0 = nb * bs
            prem = prof[rem0:]

        for i in range(0, n_shifts, xth):
            mean = 1 - (cum[i + d] - cum[i]) / d
            if mean > transit_depth_min:
                target_depth = mean * ov
                k = 1 / (signal_depth / target_depth)
                if prune and length >= PRUNE_MIN_LENGTH:
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
                if nb > 0:
                    if multirow:
                        q = (i % bs) * q_row + i // bs
                    else:
                        q = i // bs
                    ar = _dot1(ab, brw[q : q + nb])
                    if uniform_weights:
                        ar += _dot1(prem, rw[i + rem0 : i + length])
                        a2 = a2_const
                    else:
                        ar2, a22 = _dot2(
                            prem, rw[i + rem0 : i + length], w[i + rem0 : i + length]
                        )
                        ar += ar2
                        a2 = _dot1(a2bin, bw[q : q + nb]) + a22
                elif uniform_weights:
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


def shift_stride(width, margin):
    """Phase-shift stride of a template (as in core.lowest_residuals_...)."""
    if margin > 0 and width > margin:
        return max(1, int(width / (1 / margin)))
    return 1


def bin_templates(templates, margin, min_stride, pieces=0):
    """Binned templates for the approximate correlation (ideas B1, B1b).

    Per template width, the bin size is
      * the shift stride xth, if xth >= min_stride (min_stride <= 0: never);
        bins are then aligned with the shift grid (B1);
      * else length // pieces, if pieces > 0 and length >= 2 * pieces (B1b,
        piecewise-constant template with about `pieces` pieces);
      * else no binning (exact).
    Returns (nbins, bsize, boff, a_bin, a2_bin) with the bin means of a, a^2.
    """
    widths, rows, offsets, lengths, profile, overshoot, sum_a2 = templates
    nbins = numpy.zeros(len(widths), dtype=numpy.int64)
    bsize = numpy.ones(len(widths), dtype=numpy.int64)
    boff = numpy.zeros(len(widths), dtype=numpy.int64)
    a_b, a2_b = [numpy.zeros(0)], [numpy.zeros(0)]
    pos = 0
    for u in range(len(widths)):
        xth = shift_stride(widths[u], margin)
        size = 0
        if min_stride > 0 and xth >= min_stride:
            size = xth
        elif pieces > 0 and lengths[u] >= 2 * pieces:
            size = lengths[u] // pieces
        if size >= 2:
            nb = lengths[u] // size
            a = profile[offsets[u] : offsets[u] + nb * size].reshape(nb, size)
            a_b.append(a.mean(axis=1))
            a2_b.append((a * a).mean(axis=1))
            nbins[u] = nb
            bsize[u] = size
        boff[u] = pos
        pos += nbins[u]
    return nbins, bsize, boff, numpy.concatenate(a_b), numpy.concatenate(a2_b)


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
        self.dtype = numpy.float64
        self.set_binning(0)

    def set_precision(self, dtype):
        """float64 (default) or float32 for the dot products (approximate)."""
        self.dtype = numpy.dtype(dtype).type
        self._ws = None
        self.set_binning(self.min_stride, getattr(self, "pieces", 0))

    def set_binning(self, min_stride, pieces=None):
        """Approximate binned correlation (see bin_templates): min_stride > 0
        for stride-aligned bins, pieces > 0 for piecewise-constant templates
        elsewhere; both 0: exact."""
        self.min_stride = min_stride
        if pieces is not None:
            self.pieces = pieces
        nbins, bsize, boff, a_bin, a2_bin = bin_templates(
            self.templates,
            float(self.problem.T0_search_margin),
            min_stride,
            getattr(self, "pieces", 0),
        )
        self.binning = (
            nbins,
            bsize,
            boff,
            a_bin.astype(self.dtype),
            a2_bin.astype(self.dtype),
        )
        widths, rows, offsets, lengths, profile, overshoot, sum_a2 = self.templates
        self.templates_kernel = (
            widths,
            rows,
            offsets,
            lengths,
            profile.astype(self.dtype),
            overshoot,
            sum_a2,
        )

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
            *self.templates_kernel,
            float(p.T0_search_margin),
            float(tls_constants.SIGNAL_DEPTH),
            self.prune,
            *self.binning,
            *self.workspace(),
        )
        return period, chi2, row, depth

    def workspace(self):
        if getattr(self, "_ws", None) is None:
            maxw = int(self.templates[0][-1]) + 1
            nf, ni, nv = workspace_size(len(self.t), maxw)
            self._ws = (
                numpy.empty(nf),
                numpy.empty(ni, dtype=numpy.int64),
                numpy.empty(nv, dtype=self.dtype),
            )
        return self._ws

    def __getstate__(self):  # do not pickle work buffers
        state = dict(self.__dict__)
        state["_ws"] = None
        return state


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
    nbins,
    bsize,
    boff,
    a_bin,
    a2_bin,
    n_chunks,
):
    """search_period_fused for many periods, numba threads (prange over
    n_chunks chunks, each with its own work buffers)."""
    n_p = len(periods)
    chi2 = numpy.empty(n_p)
    row = numpy.empty(n_p, dtype=numpy.int64)
    depth = numpy.empty(n_p)
    maxw = widths[-1] + 1
    n = len(t)
    m = n + maxw + 1
    nf = 4 * n + 4 * m + 8
    ni = 2 * n + 2
    for c_idx in numba.prange(n_chunks):
        fbuf = numpy.empty(nf)
        ibuf = numpy.empty(ni, dtype=numpy.int64)
        vbuf = numpy.empty(6 * m, dtype=profile.dtype)
        for p in range(c_idx, n_p, n_chunks):
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
                nbins,
                bsize,
                boff,
                a_bin,
                a2_bin,
                fbuf,
                ibuf,
                vbuf,
            )
            chi2[p] = c
            row[p] = r
            depth[p] = d
    return chi2, row, depth

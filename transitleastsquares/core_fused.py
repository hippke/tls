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

import os

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
def _stable_argsort_into(keys, order, tmp):
    """order = numpy.argsort(keys, kind="mergesort") (stable: equal keys keep
    index order): insertion-sorted runs of 32, then bottom-up merges; tmp:
    work buffer of len(order). Used for the rare clustered-phase fallback
    instead of numpy's mergesort, whose numba implementation costs ~0.6 s of
    first-run compilation (step 52)."""
    n = len(order)
    run = 32
    for lo in range(0, n, run):
        hi = min(lo + run, n)
        order[lo] = lo
        for i in range(lo + 1, hi):
            kv = keys[i]
            j = i - 1
            while j >= lo and keys[order[j]] > kv:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = i
    src = order
    dst = tmp
    in_order = True  # src is `order`
    width = run
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n)
            hi = min(lo + 2 * width, n)
            i = lo
            j = mid
            k = lo
            if mid < hi and keys[src[mid - 1]] <= keys[src[mid]]:
                # already in order: copy
                for q in range(lo, hi):
                    dst[q] = src[q]
                continue
            if i < mid and j < hi:
                ki = keys[src[i]]
                kj = keys[src[j]]
                while True:
                    if kj < ki:
                        dst[k] = src[j]
                        k += 1
                        j += 1
                        if j == hi:
                            break
                        kj = keys[src[j]]
                    else:
                        dst[k] = src[i]
                        k += 1
                        i += 1
                        if i == mid:
                            break
                        ki = keys[src[i]]
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
        src, dst = dst, src
        in_order = not in_order
        width *= 2
    if not in_order:
        for i in range(n):
            order[i] = src[i]


@numba.njit(cache=True)
def fold_sort_into(
    t, y, inv_dy2, period, phases, counts, bucket, keys, flux, w, move_w, folded=False
):
    """Phase-fold and write flux and weights sorted by phase (stable order)
    into `flux` and `w` (work arrays: phases, keys: n; counts: n+1; bucket: n).
    move_w=False: `w` is not written (all weights equal; saves memory traffic).

    Bucket sort: phases are near-uniform in [0, 1), so N buckets hold about
    one point each. A stable counting scatter followed by an insertion sort
    gives exactly the stable (mergesort) order in O(N); 2-4x faster than
    numpy/numba mergesort (scratch/natsort.py). Falls back to mergesort if the
    phases are strongly clustered (insertion work > 8 N).
    """
    n = len(t)
    if not folded:
        _foldfast_into(t, period, phases)
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
        if move_w:
            w[pos] = inv_dy2[i]
    work = 0
    for i in range(1, n):
        kv = keys[i]
        if kv < keys[i - 1]:
            fv = flux[i]
            wv = w[i] if move_w else 0.0
            j = i - 1
            while j >= 0 and keys[j] > kv:
                keys[j + 1] = keys[j]
                flux[j + 1] = flux[j]
                if move_w:
                    w[j + 1] = w[j]
                j -= 1
            keys[j + 1] = kv
            flux[j + 1] = fv
            if move_w:
                w[j + 1] = wv
            work += i - 1 - j
            if work > 8 * n:  # clustered phases: use the O(N log N) sort
                order = numpy.empty(n, dtype=bucket.dtype)
                _stable_argsort_into(phases[:n], order, bucket)
                for q in range(n):
                    flux[q] = y[order[q]]
                    if move_w:
                        w[q] = inv_dy2[order[q]]
                return


@numba.njit(cache=True)
def fold_order_into(t, period, phases, counts, order, keys, folded=False):
    """Stable phase permutation, delaying flux/weight gathers until prefixing.

    Recompute bucket IDs during scatter so the integer buffer can hold output
    indices instead of input bucket IDs. Only the permutation is scattered: no
    keys and no flux/weight payload (PERFORMANCE_LOG step 49); the insertion
    repair compares phases[order[i]] with the same strict ordering, so the
    result is the identical stable order. `keys` is no longer written.
    folded=True reuses phases from an unsuccessful few-run merge attempt.
    Returns the insertion-repair work (moves; used by the walk-fold probes).
    """
    n = len(t)
    if not folded:
        _foldfast_into(t, period, phases)
    for k in range(n + 1):
        counts[k] = 0
    for i in range(n):
        k = min(n - 1, max(0, int(phases[i] * n)))
        counts[k + 1] += 1
    for k in range(n):
        counts[k + 1] += counts[k]
    for i in range(n):
        k = min(n - 1, max(0, int(phases[i] * n)))
        pos = counts[k]
        counts[k] = pos + 1
        order[pos] = i
    work = 0
    prev = phases[order[0]] if n > 0 else 0.0
    for i in range(1, n):
        iv = order[i]
        kv = phases[iv]
        if kv < prev:
            j = i - 1
            while j >= 0 and phases[order[j]] > kv:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = iv
            work += i - 1 - j
        else:
            prev = kv
            if work > 8 * n:  # clustered phases: O(N log N) stable sort
                _stable_argsort_into(phases[:n], order, counts[:n])
                return work
    return work


# --- Three-gap walk fold (idea L12, PERFORMANCE_PLAN §13.2, step 51) --------
#
# For time stamps on a regular grid t = t_ref + g * delta + e (integer slot g,
# 0 <= g < G, empty slots allowed, small deviations e) the grid phases
# frac(phi0 + g * alpha), alpha = delta / P, are ordered by the three-distance
# theorem: the circular successor of slot g is g + a, g - b or g + a - b, with
# a, b the denominators of the Farey neighbours of alpha of order G - 1. The
# walk proposes the phase order without counting or scattering; the insertion
# repair on the actual phases makes it the exact stable order, so the result
# never depends on how good the proposal is (any permutation is repaired).


@numba.njit(cache=True)
def _three_gap_steps(G, alpha):
    """(a, b) = (argmin, argmax) over 1 <= j < G of frac(j * alpha): the
    denominators of the best lower / upper rational approximations of alpha
    with denominator < G (Stern-Brocot descent, batched steps, O(log G))."""
    pl, ql, pu, qu = 0, 1, 1, 1
    while True:
        if pl + pu <= alpha * (ql + qu):  # mediant <= alpha: raise the lower bound
            den = pu - alpha * qu
            k = max(1, int((alpha * ql - pl) / den)) if den > 0 else G
            k = min(k, (G - 1 - ql) // qu)
            if k < 1:
                break
            pl += k * pu
            ql += k * qu
        else:
            den = alpha * ql - pl
            k = max(1, int((pu - alpha * qu) / den)) if den > 0 else G
            k = min(k, (G - 1 - qu) // ql)
            if k < 1:
                break
            pu += k * pl
            qu += k * ql
    return ql, qu


@numba.njit(cache=True)
def _grid_phase(x):
    return x - numpy.floor(x)


# float64 machine epsilon (numpy.finfo(numpy.float64).eps), used in the
# rigorous bound on the foldfast rounding inside fold_walk_into.
WALK_PHASE_EPS = 2.220446049250313e-16


@numba.njit(cache=True)
def _walk_certified(period, walk, a, b):
    """Certified no-scan walk (idea N1, step 54): is the assembled walk order
    provably the stable phase order?

    A step of the walk (+a, -b or +a-b) moves the *grid* phase
    frac(phi0 + g * alpha), alpha = frac(delta / P), by a constant of its
    type, whatever the descent returned for near-rational alpha (the branch
    conditions depend only on the slot, so every slot's step of a type has
    the same displacement):
        A = frac(a * alpha),  B = 1 - frac(b * alpha),  (A + B - 1) mod 1.
    Every actual phase frac(t_i / P) is within
        dev = (emax + 8 * eps * max|t|) / P + eps
    of its grid phase (circularly): emax bounds the stamp deviations
    e_i = t_i - (t_ref + g_i * delta) (regular_grid), 8 * eps * max|t| / P
    covers the foldfast rounding of t/P (incl. fastmath reassociation), eps
    the rotation alpha; max|t| <= |t_ref| + G * delta + emax is rigorous.

    If all three step displacements (as signed minimal displacements) lie in
    (2 * dev, 1/2] - strictly forward, by more than twice the phase
    deviation - and the walk winds exactly once around the phase circle
    (W = (G - a) d_a + (G - b) d_b + (a + b - G) d_ab = 1; W is an integer
    by telescoping over the cycle, so |W - 1| <= 1/4 decides it), the walk
    visits the slots in grid-phase order with consecutive gaps > 2 * dev.
    The actual phases inherit that order strictly - also across the wrapped
    ends, where a misplacement would require two grid phases closer than
    2 * dev < the smallest gap - so the assembled order is the unique stable
    order (no ties) and the insertion-repair scan can be skipped.
    """
    gp, delta, t_ref, emax = walk[0], walk[2], walk[3], walk[4]
    G = len(gp)
    alpha = _grid_phase(delta / period)
    dev = (emax + 8.0 * WALK_PHASE_EPS * (abs(t_ref) + G * delta + emax)) / period
    dev += WALK_PHASE_EPS
    A = a * alpha - numpy.floor(a * alpha)
    B = 1.0 - (b * alpha - numpy.floor(b * alpha))
    d_a = A if A <= 0.5 else A - 1.0
    d_b = B if B <= 0.5 else B - 1.0
    s = A + B - 1.0  # exact displacement of +a-b steps, before the mod
    if s > 0.5:
        d_ab = s - 1.0
    elif s < -0.5:
        d_ab = s + 1.0
    else:
        d_ab = s
    twice = 2.0 * dev
    if not (d_a > twice and d_b > twice and d_ab > twice):
        return False
    W = (G - a) * d_a + (G - b) * d_b + (a + b - G) * d_ab
    return abs(W - 1.0) <= 0.25


@numba.njit(cache=True)
def fold_walk_into(period, phases, walk, buf, order, budget, certify=True):
    """Stable phase permutation of a regular-grid light curve (idea L12).

    phases: actual (foldfast) phases. walk = (slot map gp[G] -> data index or
    -1, slot per point, delta, t_ref, max|e|, P*) from regular_grid. buf: work
    buffer of n + 1 integers; order: output (n). Returns the insertion-repair
    work (>= 0), or -1 if the walk is not usable or the repair exceeds
    `budget` moves; `order` must then be rebuilt by fold_order_into.
    The result is the same permutation as fold_order_into: sorted by phase,
    ties by index (the repair compares (phase, index) lexicographically).
    certify=False always runs the repair scan (testing hook for step 54).
    """
    gp, slot, delta, t_ref, emax = walk[0], walk[1], walk[2], walk[3], walk[4]
    n = len(phases)
    G = len(gp)
    if n < 2 or G < 2:
        return -1
    alpha = _grid_phase(delta / period)
    if not alpha > 0.0:
        return -1
    a, b = _three_gap_steps(G, alpha)
    # a + b >= G with 1 <= a, b < G makes the successor map a bijection
    if a < 1 or b < 1 or a >= G or b >= G or a + b < G:
        return -1
    j = 0
    cnt = 0
    steps = 0
    for _ in range(G):
        k = gp[j]
        buf[cnt] = k  # cnt <= n: buf has n + 1 entries
        if k >= 0:
            cnt += 1
        j1 = j + a
        if j1 < G:
            j = j1
        elif j >= b:
            j = j - b
        else:
            j = j1 - b
        steps += 1
        if j == 0:
            break
    # a single cycle through all G slots, i.e. every point exactly once
    if steps != G or cnt != n:
        return -1
    # rotate to the smallest grid phase (binary search for the wrap of the
    # grid phases, which increase cyclically along the walk)
    phi0 = _grid_phase(t_ref / period)
    g0 = _grid_phase(phi0 + slot[buf[0]] * alpha)
    lo = 1
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if _grid_phase(phi0 + slot[buf[mid]] * alpha) < g0:
            hi = mid
        else:
            lo = mid + 1
    rot = lo if lo < n else 0
    # Points whose actual phase wrapped across 0/1 relative to the grid phase
    # (|phase - grid phase| <= max|e| / P) can only lie within the first hs /
    # last ts walk positions: move them to the other end.
    thr = 2.0 * (emax / period + 1e-9)
    hs = 0
    while hs < n:
        q = rot + hs
        if q >= n:
            q -= n
        ph = phases[buf[q]]
        if thr <= ph <= 1.0 - thr:
            break
        hs += 1
    ts = 0
    while hs + ts < n:
        q = rot + n - 1 - ts
        if q >= n:
            q -= n
        ph = phases[buf[q]]
        if thr <= ph <= 1.0 - thr:
            break
        ts += 1
    out = 0
    for i in range(n - ts, n):  # wrapped to phase ~0: first
        q = rot + i
        if q >= n:
            q -= n
        if phases[buf[q]] < 0.5:
            order[out] = buf[q]
            out += 1
    for i in range(hs):
        q = rot + i
        if q >= n:
            q -= n
        if phases[buf[q]] < 0.5:
            order[out] = buf[q]
            out += 1
    # bulk: walk positions hs .. n - ts - 1 after rotation (two segments)
    s0 = rot + hs
    s1 = rot + n - ts
    if s0 >= n:
        for q in range(s0 - n, s1 - n):
            order[out] = buf[q]
            out += 1
    elif s1 <= n:
        for q in range(s0, s1):
            order[out] = buf[q]
            out += 1
    else:
        for q in range(s0, n):
            order[out] = buf[q]
            out += 1
        for q in range(0, s1 - n):
            order[out] = buf[q]
            out += 1
    for i in range(n - ts, n):
        q = rot + i
        if q >= n:
            q -= n
        if phases[buf[q]] >= 0.5:
            order[out] = buf[q]
            out += 1
    for i in range(hs):  # wrapped to phase ~1: last
        q = rot + i
        if q >= n:
            q -= n
        if phases[buf[q]] >= 0.5:
            order[out] = buf[q]
            out += 1
    # Certified no-scan walk (step 54): skip the repair scan when the
    # assembled order is provably the stable order (_walk_certified); only
    # the two boundary pairs are verified, and any surprise falls back to
    # the scan (the result never depends on the certification).
    if certify and _walk_certified(period, walk, a, b):
        if (
            phases[order[0]] <= phases[order[1]]
            and phases[order[n - 2]] <= phases[order[n - 1]]
        ):
            return 0
    # exact stable order: insertion repair on (phase, index)
    work = 0
    pv = order[0]
    prev = phases[pv]
    for i in range(1, n):
        iv = order[i]
        kv = phases[iv]
        if kv < prev or (kv == prev and iv < pv):
            j = i - 1
            while j >= 0:
                oj = order[j]
                po = phases[oj]
                if po > kv or (po == kv and oj > iv):
                    order[j + 1] = oj
                    j -= 1
                else:
                    break
            order[j + 1] = iv
            work += i - 1 - j
            if work > budget:
                return -1
        else:
            prev = kv
            pv = iv
    return work


def walk_repair_work(t, periods, walk, index_dtype=numpy.int64, budget=None):
    """Per probe period: insertion-repair work of the walk fold (-1: not
    usable or more than `budget` moves) and of the bucket path; used to find
    the crossover period P* (regular_grid_crossover). A Python loop over a
    few periods (no extra compiled function: less first-run compilation)."""
    n = len(t)
    phases = numpy.empty(n)
    buf = numpy.empty(n + 1, dtype=index_dtype)
    order = numpy.empty(n, dtype=index_dtype)
    keys = numpy.empty(0)  # not written by fold_order_into
    if budget is None:
        budget = int(WALK_SEARCH_WORK * n)
    walk = tuple(walk[:5]) + (numpy.inf,)
    out = numpy.empty((len(periods), 2), dtype=numpy.int64)
    for p, period in enumerate(periods):
        period = float(period)
        _foldfast_into(t, period, phases)
        out[p, 0] = fold_walk_into(period, phases, walk, buf, order, budget)
        out[p, 1] = fold_order_into(t, period, phases, buf, order, keys, True)
    return out


def regular_grid(t, index_dtype=numpy.int64, max_fill=2.0):
    """Regular cadence grid of the time stamps for the walk fold (idea L12).

    Slot increments rint(dt / median dt) between sorted time stamps (local
    rounding is robust to a slowly drifting cadence, e.g. barycentric
    corrections), least-squares spacing delta and origin t_ref, deviations
    e = t - t_ref - slot * delta. Returns (gp, slot, delta, t_ref, max|e|) or
    None if the stamps do not form a usable grid: duplicate slots, more than
    max_fill * N slots (sparse data: the walk visits every slot) or
    max|e| >= delta / 4.
    """
    t = numpy.asarray(t, dtype=float)
    n = len(t)
    if n < 16:
        return None
    dt = numpy.diff(t)
    if numpy.all(dt > 0):  # usual case: sorted input
        srt = numpy.arange(n)
        ts = t
    else:
        srt = numpy.argsort(t, kind="mergesort")
        ts = t[srt]
        dt = numpy.diff(ts)
    if not numpy.all(dt > 0):
        return None
    d0 = float(numpy.median(dt))
    inc = numpy.rint(dt / d0)
    if numpy.any(inc < 1):
        return None
    g = numpy.concatenate(([0.0], numpy.cumsum(inc)))
    G = int(g[-1]) + 1
    if G > max_fill * n or G >= numpy.iinfo(index_dtype).max:
        return None
    x = ts - ts[0]
    gm, xm = g.mean(), x.mean()
    delta = float(numpy.sum((g - gm) * (x - xm)) / numpy.sum((g - gm) ** 2))
    origin = xm - delta * gm
    e = x - (origin + g * delta)
    emax = float(numpy.max(numpy.abs(e)))
    if not delta > 0 or emax >= 0.25 * delta:
        return None
    slot = numpy.empty(n, dtype=index_dtype)
    slot[srt] = g.astype(index_dtype)
    gp = numpy.full(G, -1, dtype=index_dtype)
    gp[slot] = numpy.arange(n, dtype=index_dtype)
    return gp, slot, delta, float(ts[0] + origin), emax


# Walk-fold gate (data-driven, no machine constants): a probe period passes
# if the walk's insertion repair needs no more moves than the bucket path's
# own repair at that period, since the walk itself (G slot steps, sequential
# writes) is cheaper than the counting pass plus scatter. In the search, the
# walk gives up after WALK_SEARCH_WORK * N moves and the bucket path takes over.
# Some periods fail sporadically (alpha close to a fraction with a small
# denominator: near-tied grid phases), more of them towards short P. A failed
# period costs about as much as two folds, a successful one saves about half
# a fold, so the walk is used where at least WALK_PASS_FRACTION of the probes
# at and above the period pass.
WALK_SEARCH_WORK = 0.5
WALK_PROBE_RATIO = 1.25  # spacing of the probe periods
WALK_PASS_FRACTION = 0.75


def regular_grid_crossover(
    t, walk, time_span, index_dtype=numpy.int64, period_range=None
):
    """Data-driven crossover period P*: the walk is used for P >= P*.

    Probe periods are log-spaced (ratio WALK_PROBE_RATIO) from time_span / 2
    down to max(16 delta, time_span / 2000), restricted to period_range =
    (min, max) of the searched periods if given; the walk's repair work falls
    with P (it scales like max|e| N / P). A probe passes if the walk's repair
    work is at most the bucket path's. P* is the shortest passing probe at
    which at least WALK_PASS_FRACTION of the probes at and above it pass;
    inf if there is none.
    """
    delta = walk[2]
    p_hi = time_span / 2
    p_lo = max(16 * delta, time_span / 2000)
    if period_range is not None:
        p_hi = min(p_hi, float(period_range[1]))
        p_lo = max(p_lo, float(period_range[0]))
    p_hi = max(p_hi, p_lo)
    count = 1 + int(numpy.ceil(numpy.log(p_hi / p_lo) / numpy.log(WALK_PROBE_RATIO)))
    periods = numpy.geomspace(p_hi, p_lo, count)
    work = walk_repair_work(t, periods, walk, index_dtype)
    passed = (work[:, 0] >= 0) & (work[:, 0] <= work[:, 1])
    fraction = numpy.cumsum(passed) / numpy.arange(1, len(passed) + 1)
    ok = passed & (fraction >= WALK_PASS_FRACTION)
    return float(numpy.min(periods[ok])) if numpy.any(ok) else numpy.inf


def walk_disabled(index_dtype=numpy.int64):
    """Kernel argument for problems without a usable grid (never walks)."""
    empty = numpy.zeros(0, dtype=index_dtype)
    return (empty, empty, 1.0, 0.0, 0.0, numpy.inf)


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
        True,
    )
    return flux, w


def sort_index_dtype(n):
    """Counts include the end offset N; use compact storage only if it fits."""
    return numpy.int32 if n <= numpy.iinfo(numpy.int32).max else numpy.int64


def workspace_size(n, maxw):
    """Sizes (float64, integer, value buffer) of the work buffers of
    search_period_fused."""
    m = n + maxw + 1
    return 4 * n + 6 * m + 16, 2 * n + 2, 6 * m


# Pruning pays off only if the dot product is much more expensive than the bound
PRUNE_MIN_LENGTH = 48

# Scout durations (L8) only for light curves with >= SCOUT_MIN_N points: for
# very short ones the per-period optimum of noise is less coherent across
# durations (noise-only N = 150: 15 % of the periods change, up to 3 %;
# N >= 2000: ~1 %, <= 7e-4; PERFORMANCE_LOG step 28). Small N is fast anyway.
SCOUT_MIN_N = 2000


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
def _pl_eval(c, pos, dd):
    """sum_t c[t] * dd[pos[t]]: correlation of a piecewise-linear template
    with the data whose double prefix sum is dd (idea L3)."""
    acc = 0.0
    for t in range(len(c)):
        acc += c[t] * dd[pos[t]]
    return acc


@numba.njit(fastmath=True, cache=True)
def _input_means(y, inv_dy2):
    """Period-independent centering constants for the double prefix sums."""
    mu_rw = 0.0
    mu_w = 0.0
    for i in range(len(y)):
        mu_rw += (1.0 - y[i]) * inv_dy2[i]
        mu_w += inv_dy2[i]
    return mu_rw / len(y), mu_w / len(y)


@numba.njit(cache=True)
def _dedup_windows(win_lo, win_hi, n_win, xc, pc_lo, pc_hi):
    """Rewrite the shift windows [win_lo, win_hi) of a non-scout duration so
    that shifts covered by an earlier window are skipped (step 58), keeping
    the order of the remaining shifts. Re-evaluating a shift never changes
    the result (gains replace the incumbents only on a strict increase, and
    the pruning incumbents only grow, so a shift pruned once stays pruned),
    so the search result is identical. All windows lie on the grid of
    multiples of xc. pc_lo/pc_hi: work arrays. Returns the new count."""
    n_p = 0
    for wq in range(n_win):
        s0 = n_p
        pc_lo[n_p] = win_lo[wq]
        pc_hi[n_p] = win_hi[wq]
        n_p += 1
        for pq in range(wq):
            a0 = win_lo[pq]
            b0 = win_hi[pq]
            e0 = n_p
            for z in range(s0, e0):
                lz = pc_lo[z]
                hz = pc_hi[z]
                if hz <= lz or hz <= a0 or lz >= b0:
                    continue
                right = ((b0 + xc - 1) // xc) * xc
                pc_hi[z] = a0 if a0 > lz else lz
                if right < hz:
                    pc_lo[n_p] = right
                    pc_hi[n_p] = hz
                    n_p += 1
        for z in range(s0 + 1, n_p):
            zl = pc_lo[z]
            zh = pc_hi[z]
            y = z - 1
            while y >= s0 and pc_lo[y] > zl:
                pc_lo[y + 1] = pc_lo[y]
                pc_hi[y + 1] = pc_hi[y]
                y -= 1
            pc_lo[y + 1] = zl
            pc_hi[y + 1] = zh
    n_win = 0
    for z in range(n_p):
        if pc_hi[z] > pc_lo[z]:
            win_lo[n_win] = pc_lo[z]
            win_hi[n_win] = pc_hi[z]
            n_win += 1
    return n_win


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
    bins,
    pl_n,
    pl_off,
    pl_n2,
    pl_off2,
    pl_pos,
    pl_c,
    pl_sum,
    pl_sum2,
    pl_a2_approx,
    pl_w_max,
    t0_coarsen,
    scout_every,
    ls_depth,
    fbuf,
    ibuf,
    vbuf,
    invariants,
    screen=None,
    walk=None,
):
    """Returns (chi2_min, template row, depth) for one trial period.

    y: input *residuals* r = 1 - flux (float64, or float32 storage for the
    approximate backends; see FusedProblem.set_storage). The kernel uses
    f = 1 - r, which is exact for float32 r, so (1 - f) recovers r.
    inv_dy2: weights 1/dy^2 in the same storage precision.

    fbuf/ibuf: preallocated work buffers (see workspace_size). Reusing them
    across periods avoids ~1 ms of page faults per period for N ~ 1e5.

    invariants: input means (rw, w) and per-template max(a^2), computed once
    per problem/precision rather than once per period.

    screen: optional coarse-PL certificates. Only candidates whose upper gain
    can improve the relevant incumbent receive the original full correlation.
    None specializes this path away; the default PL backend does not use it.

    walk: regular-grid data for the three-gap walk fold (regular_grid plus
    the crossover period P*, see FusedProblem.set_walk; idea L12). For
    P >= P* the stable phase order comes from the walk instead of the
    counting scatter; the permutation is identical. None: never walk.

    bins: None (default and exact backends: the binned code is specialized
    away, which shortens compilation), or (nbins, bsize, boff, a_bin, a2_bin),
    stride-binned templates (see bin_templates): where nbins[u] > 0 the
    approximate stride-binned correlation (idea B1) is used.

    pl_*: piecewise-linear templates (see pl_templates, idea L3). If
    pl_n[u] > 0, AR (and A2 for non-uniform weights) of template u come from
    pl_n[u] (pl_n2[u]) terms on double prefix sums; this takes precedence
    over the bins. Approximate unless every sample is a knot.
    pl_a2_approx (non-uniform weights, PL templates only; idea U2): A2 from
    the window-mean weight, sum(a^2) * mean(w over the window), instead of
    the second PL correlation; pruning then uses A2_true <= sum(a^2) * w_max.
    t0_coarsen = c > 1 (idea L5, approximate): shifts on a c times coarser
    grid (where int(c * margin * width) >= 2), then the c - 1 fine-grid
    shifts on each side of the best coarse shift of every duration.
    scout_every = s > 0 (idea L8, approximate): only every s-th allowed
    duration (and the longest) scans all phases; the others scan windows of
    +- one duration around the best centres found by these scouts.
    ls_depth (idea B6/L9, statistic change; None = off and specialized away,
    True = on): depth by least squares per shift,
    k = AR / A2 and gain = AR^2 / A2 (shifts with AR <= 0 skipped), instead
    of TLS's box mean * overshoot. The box-depth threshold still selects
    the shifts. Pruning: gain <= R2 (Cauchy-Schwarz; scaled by A2_max / A2
    for the U2 approximation).

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
    dd_rw = fbuf[o : o + m + 2]
    o += m + 2
    dd_w = fbuf[o : o + m + 2]
    o += m + 2
    # values entering the dot products: dtype of vbuf (float64 or float32)
    w = vbuf[:m]
    rw = vbuf[m : 2 * m]
    brw_buf = vbuf[2 * m : 4 * m]
    bw_buf = vbuf[4 * m : 6 * m]
    counts = ibuf[: n + 1]
    bucket = ibuf[n + 1 : 2 * n + 1]

    # Try a stable three-run merge only in the long-period regime.
    folded = period * 3 >= time_span
    runs = 4
    h0, h1, h2 = 0, n, n
    e0, e1 = n, n
    if folded:
        _foldfast_into(t, period, phases)
        runs = 1
        for q in range(1, n):
            if phases[q] < phases[q - 1]:
                if runs == 1:
                    e0, h1 = q, q
                elif runs == 2:
                    e1, h2 = q, q
                else:
                    runs = 4
                    break
                runs += 1
    w0 = inv_dy2[0]
    # cum_rw is read only by stride-binned correlations: skip its chain and
    # store otherwise (less memory traffic per period; step 49).
    has_pl = len(pl_c) > 0
    if screen is not None:
        has_pl = True
    # U2 obtains A2 from cum_w, but screening an exact target still needs dd_w
    # even if the U2 flag was set (it only applies to targets with npl > 0).
    has_pl_w = has_pl and (not pl_a2_approx or screen is not None)
    # Double prefix sums for the piecewise-linear templates (in the same
    # pass; mean removed for precision, its contribution mu * sum(p) is added
    # back). The means come from the unsorted data (any constant is exact).
    mu_rw, mu_w, max_a2 = invariants
    cum_rw[0] = 0.0
    cum[0] = 0.0
    cum_r2[0] = 0.0
    cum_w[0] = 0.0
    dd_rw[0] = 0.0
    dd_w[0] = 0.0
    s_rw = 0.0
    a_rw = 0.0
    s_w = 0.0
    a_w = 0.0
    total = 0.0
    if runs <= 3:
        p0 = phases[0]
        p1 = phases[h1] if h1 < e1 else 2.0
        p2 = phases[h2] if h2 < n else 2.0
        if uniform_weights:
            # w = w0 everywhere: no weight arrays (w, cum_w) needed
            for k in range(m):
                if k < n:
                    if p0 <= p1 and p0 <= p2:
                        src = h0
                        h0 += 1
                        p0 = phases[h0] if h0 < e0 else 2.0
                    elif p1 <= p2:
                        src = h1
                        h1 += 1
                        p1 = phases[h1] if h1 < e1 else 2.0
                    else:
                        src = h2
                        h2 += 1
                        p2 = phases[h2] if h2 < n else 2.0
                    f = 1.0 - y[src]
                    if k < maxw:
                        flux_sorted[k] = f
                else:
                    f = flux_sorted[k - n]
                rk = (1.0 - f) * w0
                rw[k] = rk
                cum[k + 1] = cum[k] + f
                cum_r2[k + 1] = cum_r2[k] + (1.0 - f) * (1.0 - f) * w0
                if bins is not None:
                    cum_rw[k + 1] = cum_rw[k] + rk
                if k < n:
                    total += (f - 1.0) ** 2 * w0
                if has_pl:
                    a_rw += s_rw
                    dd_rw[k + 1] = a_rw
                    s_rw += rk - mu_rw
        else:
            for k in range(m):
                if k < n:
                    if p0 <= p1 and p0 <= p2:
                        src = h0
                        h0 += 1
                        p0 = phases[h0] if h0 < e0 else 2.0
                    elif p1 <= p2:
                        src = h1
                        h1 += 1
                        p1 = phases[h1] if h1 < e1 else 2.0
                    else:
                        src = h2
                        h2 += 1
                        p2 = phases[h2] if h2 < n else 2.0
                    f = 1.0 - y[src]
                    if k < maxw:
                        flux_sorted[k] = f
                else:
                    f = flux_sorted[k - n]
                if k < n:
                    wk = inv_dy2[src]
                    if k < maxw:
                        w_sorted[k] = wk
                else:
                    wk = w_sorted[k - n]
                rk = (1.0 - f) * wk
                w[k] = wk
                rw[k] = rk
                cum[k + 1] = cum[k] + f
                cum_r2[k + 1] = cum_r2[k] + (1.0 - f) * (1.0 - f) * wk
                cum_w[k + 1] = cum_w[k] + wk
                if bins is not None:
                    cum_rw[k + 1] = cum_rw[k] + rk
                if k < n:
                    total += (f - 1.0) ** 2 * wk
                if has_pl:
                    a_rw += s_rw
                    dd_rw[k + 1] = a_rw
                    s_rw += rk - mu_rw
                if has_pl_w:
                    a_w += s_w
                    dd_w[k + 1] = a_w
                    s_w += wk - mu_w
    else:
        # Regular time grid and P >= P*: three-gap walk instead of the
        # counting scatter (idea L12; identical permutation, step 51)
        walked = False
        if walk is not None:
            if period >= walk[5]:
                if not folded:
                    _foldfast_into(t, period, phases)
                    folded = True
                budget = int(WALK_SEARCH_WORK * n)
                work = fold_walk_into(period, phases, walk, counts, bucket, budget)
                walked = work >= 0
        if not walked:
            fold_order_into(t, period, phases, counts, bucket, keys, folded)
        if uniform_weights:
            # w = w0 everywhere: no weight arrays (w, cum_w) needed
            for k in range(m):
                src = bucket[k if k < n else k - n]
                f = 1.0 - y[src]
                rk = (1.0 - f) * w0
                rw[k] = rk
                cum[k + 1] = cum[k] + f
                cum_r2[k + 1] = cum_r2[k] + (1.0 - f) * (1.0 - f) * w0
                if bins is not None:
                    cum_rw[k + 1] = cum_rw[k] + rk
                if k < n:
                    total += (f - 1.0) ** 2 * w0
                if has_pl:
                    a_rw += s_rw
                    dd_rw[k + 1] = a_rw
                    s_rw += rk - mu_rw
        else:
            for k in range(m):
                src = bucket[k if k < n else k - n]
                f = 1.0 - y[src]
                wk = inv_dy2[src]
                rk = (1.0 - f) * wk
                w[k] = wk
                rw[k] = rk
                cum[k + 1] = cum[k] + f
                cum_r2[k + 1] = cum_r2[k] + (1.0 - f) * (1.0 - f) * wk
                cum_w[k + 1] = cum_w[k] + wk
                if bins is not None:
                    cum_rw[k + 1] = cum_rw[k] + rk
                if k < n:
                    total += (f - 1.0) ** 2 * wk
                if has_pl:
                    a_rw += s_rw
                    dd_rw[k + 1] = a_rw
                    s_rw += rk - mu_rw
                if has_pl_w:
                    a_w += s_w
                    dd_w[k + 1] = a_w
                    s_w += wk - mu_w
    dd_rw[m + 1] = a_rw + s_rw
    dd_w[m + 1] = a_w + s_w

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
            xth = _coarse_stride(d, xth, T0_search_margin, t0_coarsen)
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
            if uniform_weights:
                ar = _dot1(prof, rw[best_i : best_i + length])
                a2 = sum_a2[u] * w0
            else:
                ar, a2 = _dot2(
                    prof, rw[best_i : best_i + length], w[best_i : best_i + length]
                )
            gain = 2 * k * ar - k * k * a2
            if gain > seed:
                seed = gain
    # Shifts with gain <= seed (minus rounding margin) cannot be the result
    threshold = seed * (1 - 1e-9)

    # Scout durations (idea L8): allowed durations with ordinal % scout_every
    # == 0, and the longest, scan all phases first (upass 0); the others
    # (upass 1) only windows around the scouts' best centres.
    n_allowed = 0
    for u in range(len(widths)):
        if width_min <= widths[u] <= width_max:
            n_allowed += 1
    use_scouts = scout_every > 1 and n_allowed > scout_every and n >= SCOUT_MIN_N
    cand_c = numpy.empty(len(widths) + 1, dtype=numpy.int64)
    n_cand = 0
    win_lo = numpy.empty((2 * len(widths) + 2) ** 2, dtype=numpy.int64)
    win_hi = numpy.empty((2 * len(widths) + 2) ** 2, dtype=numpy.int64)
    pc_lo = numpy.empty((2 * len(widths) + 2) ** 2, dtype=numpy.int64)
    pc_hi = numpy.empty((2 * len(widths) + 2) ** 2, dtype=numpy.int64)
    for upass in range(2 if use_scouts else 1):
        ordinal = -1
        for u in range(len(widths)):
            d = widths[u]
            if d < width_min or d > width_max:
                continue
            ordinal += 1
            is_scout = (
                not use_scouts or ordinal % scout_every == 0 or ordinal == n_allowed - 1
            )
            if is_scout != (upass == 0):
                continue
            row = rows[u]
            off = offsets[u]
            length = lengths[u]
            ov = overshoot[u]
            a2_const = sum_a2[u] * w0
            prof = profile[off : off + length]
            amax2 = max_a2[u]

            xth = 1
            if T0_search_margin > 0 and d > T0_search_margin:
                xth = int(d / (1 / T0_search_margin))
                if xth < 1:
                    xth = 1
            n_shifts = m - d + 1
            if xth == 1 and n_shifts > n:
                n_shifts = n  # shifts >= n repeat shifts already tested

            npl = pl_n[u]
            pc = pl_c[pl_off[u] : pl_off[u] + npl]
            pp = pl_pos[pl_off[u] : pl_off[u] + npl]
            pc2 = pl_c[pl_off2[u] : pl_off2[u] + pl_n2[u]]
            pp2 = pl_pos[pl_off2[u] : pl_off2[u] + pl_n2[u]]
            psum = pl_sum[u] * mu_rw
            psum2 = pl_sum2[u] * mu_w
            nb = 0
            a2_win = pl_a2_approx and npl > 0 and not uniform_weights
            a2_unit = sum_a2[u] / length
            a2_wmax = sum_a2[u] * pl_w_max
            # Stride-binned correlation (fused-binned / fused-pieces only;
            # bins=None specializes this code away, PERFORMANCE_LOG step 52)
            if bins is not None:
                nbins, bsize, boff, a_bin, a2_bin = bins
                nb = nbins[u] if npl == 0 else 0
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
                    # Data bin sums of size bs. Row r holds the bins starting
                    # at r, r + bs, r + 2 bs, ... If all shifts are multiples
                    # of bs (aligned case, B1) only row 0 is needed.
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
                            if not uniform_weights:
                                bw_buf[base + kb] = cum_w[hi] - cum_w[lo]
                    brw = brw_buf
                    bw = bw_buf
                    ab = a_bin[boff[u] : boff[u] + nb]
                    a2bin = a2_bin[boff[u] : boff[u] + nb]
                    rem0 = nb * bs
                    prem = prof[rem0:]

            # coarse grid (stride xc) plus refinement around the best coarse shift
            xc = _coarse_stride(d, xth, T0_search_margin, t0_coarsen)
            sn = 0
            if screen is not None:
                sn, so = screen[0][u], screen[1][u]
                sn2, so2 = screen[2][u], screen[3][u]
                sc = screen[5][so : so + sn]
                sp = screen[4][so : so + sn]
                sc2 = screen[5][so2 : so2 + sn2]
                sp2 = screen[4][so2 : so2 + sn2]
                ss, ss2 = screen[6][u] * mu_rw, screen[7][u] * mu_w
                se, se2 = screen[8][u] * pl_w_max, screen[9][u]
                sr, sr2, srr = screen[10][u], screen[11][u], screen[12]
            # A duration's best phase can guide later searches even when its
            # gain is below the global incumbent. Preserve these anchors too.
            preserve_anchor = (is_scout and use_scouts) or xc > xth
            dur_gain = -1e300
            dur_i = -1
            i_star = -1
            # windows of pass 0: all shifts, or (non-scout) +- d around the
            # scouts' best centres (wrapped at the phase boundary)
            n_win = 1
            win_lo[0] = 0
            win_hi[0] = n_shifts
            if not is_scout:
                n_win = 0
                period_n = n  # shift i and i + n are the same phase
                for q in range(n_cand):
                    lo = cand_c[q] - d // 2 - d
                    hi = cand_c[q] - d // 2 + d + 1
                    lo = (lo // xc) * xc
                    if lo < 0:
                        win_lo[n_win] = max(0, ((lo + period_n) // xc) * xc)
                        win_hi[n_win] = min(n_shifts, period_n)
                        n_win += 1
                        lo = 0
                    if hi > n_shifts:
                        win_lo[n_win] = 0
                        win_hi[n_win] = min(n_shifts, hi - period_n)
                        n_win += 1
                        hi = n_shifts
                    if hi > lo:
                        win_lo[n_win] = lo
                        win_hi[n_win] = hi
                        n_win += 1
                n_win = _dedup_windows(win_lo, win_hi, n_win, xc, pc_lo, pc_hi)
            for pass_ in range(3 if xc > xth else 1):
                for wdx in range(n_win if pass_ == 0 else 1):
                    lo, hi, st = win_lo[wdx], win_hi[wdx], xc
                    if pass_ > 0:  # fine shifts left (1) and right (2) of i_star
                        if dur_i < 0:
                            break
                        if pass_ == 1:
                            i_star = dur_i
                            lo = max(0, i_star - (xc - xth))
                            hi = i_star
                        else:
                            lo = i_star + xth
                            hi = min(n_shifts, i_star + (xc - xth) + 1)
                        st = xth
                    for i in range(lo, hi, st):
                        mean = 1 - (cum[i + d] - cum[i]) / d
                        if mean > transit_depth_min:
                            target_depth = mean * ov
                            k = 1 / (signal_depth / target_depth)
                            if prune and length >= PRUNE_MIN_LENGTH:
                                r2 = cum_r2[i + length] - cum_r2[i]
                                if ls_depth is not None:
                                    bound = r2
                                    if a2_win:
                                        bound = (
                                            r2
                                            * a2_wmax
                                            / (a2_unit * (cum_w[i + length] - cum_w[i]))
                                        )
                                elif uniform_weights:  # A2 known exactly
                                    bound = (
                                        2 * k * numpy.sqrt(a2_const * r2)
                                        - k * k * a2_const
                                    )
                                elif a2_win:  # approximate A2 (U2)
                                    # AR <= sqrt(A2_max R2)
                                    a2e = a2_unit * (cum_w[i + length] - cum_w[i])
                                    bound = (
                                        2 * k * numpy.sqrt(a2_wmax * r2) - k * k * a2e
                                    )
                                else:  # 0 <= A2 <= a2b; maximum of the concave bound
                                    a2b = amax2 * (cum_w[i + length] - cum_w[i])
                                    if a2b * k * k >= r2:
                                        bound = r2
                                    else:
                                        bound = (
                                            2 * k * numpy.sqrt(a2b * r2) - k * k * a2b
                                        )
                                if bound * (1 + 1e-9) <= best_gain or bound < threshold:
                                    continue
                            if (
                                screen is not None
                                and sn > 0
                                and prune
                                and ls_depth is None
                                and nb == 0
                                and length >= PRUNE_MIN_LENGTH
                                and k > 0
                            ):
                                cutoff = dur_gain if preserve_anchor else best_gain
                                if cutoff >= 0:
                                    proxy_ar = _pl_eval(sc, sp, dd_rw[i:]) + ss
                                    ar_upper = (
                                        proxy_ar + sr
                                        + numpy.sqrt(max(0.0, se * (r2 + srr)))
                                    )
                                    if uniform_weights:
                                        lower_a2 = a2_const
                                    elif a2_win:
                                        lower_a2 = a2_unit * (
                                            cum_w[i + length] - cum_w[i]
                                        )
                                    else:
                                        lower_a2 = (
                                            _pl_eval(sc2, sp2, dd_w[i:]) + ss2
                                            - se2 * (cum_w[i + length] - cum_w[i])
                                            - sr2
                                        )
                                    proxy_bound = 2 * k * ar_upper - k * k * lower_a2
                                    padding = 1e-9 * (abs(proxy_bound) + cutoff)
                                    if proxy_bound + padding <= cutoff:
                                        continue
                            if npl > 0:
                                ar = _pl_eval(pc, pp, dd_rw[i:]) + psum
                                if uniform_weights:
                                    a2 = a2_const
                                elif a2_win:
                                    a2 = a2_unit * (cum_w[i + length] - cum_w[i])
                                else:
                                    a2 = _pl_eval(pc2, pp2, dd_w[i:]) + psum2
                            elif bins is not None and nb > 0:
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
                                        prem,
                                        rw[i + rem0 : i + length],
                                        w[i + rem0 : i + length],
                                    )
                                    ar += ar2
                                    a2 = _dot1(a2bin, bw[q : q + nb]) + a22
                            elif uniform_weights:
                                ar = _dot1(prof, rw[i : i + length])
                                a2 = a2_const
                            else:
                                ar, a2 = _dot2(
                                    prof, rw[i : i + length], w[i : i + length]
                                )
                            if ls_depth is not None:
                                if ar <= 0 or a2 <= 0:
                                    continue
                                k = ar / a2
                                target_depth = k * signal_depth
                                gain = ar * ar / a2
                            else:
                                gain = 2 * k * ar - k * k * a2
                            if gain > dur_gain:
                                dur_gain = gain
                                dur_i = i
                            if gain > best_gain:
                                best_gain = gain
                                best_row = row
                                best_depth = 1 - target_depth
            if upass == 0 and use_scouts and dur_i >= 0:
                cc = dur_i + d // 2
                dup = False
                for q in range(n_cand):
                    if abs(cand_c[q] - cc) <= d // 4:
                        dup = True
                if not dup:
                    cand_c[n_cand] = cc
                    n_cand += 1

    return total - best_gain, best_row, best_depth


@numba.njit(cache=True)
def _coarse_stride(d, xth, margin, c):
    """Shift stride of the coarse T0 grid (idea L5): c * xth if
    int(c * margin * d) >= 2, else xth (unchanged)."""
    if c > 1 and int(c * margin * d) >= 2:
        return c * xth
    return xth


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


@numba.njit(cache=True)
def _segment_ok(a, b, c, tol):
    """Linear interpolant from sample b to sample c within tol of a[b..c]?"""
    s = (a[c] - a[b]) / (c - b)
    for j in range(b + 1, c):
        if abs(a[b] + s * (j - b) - a[j]) > tol:
            return False
    return True


@numba.njit(cache=True)
def _greedy_knots(a, tol):
    """Integer knots 0 = q_0 < ... < q_K = len(a) - 1 such that the linear
    interpolant through (q_k, a[q_k]) stays within tol of every sample.
    Each segment is grown by doubling, then bisection (O(L log L))."""
    n = len(a)
    knots = numpy.empty(n, dtype=numpy.int64)
    knots[0] = 0
    k = 1
    b = 0
    while b < n - 1:
        good = b + 1  # always valid
        step = 1
        bad = -1
        while True:
            c = b + 2 * step
            if c > n - 1:
                c = n - 1
            if c <= good:
                break
            if _segment_ok(a, b, c, tol):
                good = c
                if c == n - 1:
                    break
                step *= 2
            else:
                bad = c
                break
        if bad > 0:
            while bad - good > 1:
                c = (good + bad) // 2
                if _segment_ok(a, b, c, tol):
                    good = c
                else:
                    bad = c
        knots[k] = good
        k += 1
        b = good
    return knots[:k]


@numba.njit(cache=True)
def _pl_lsq(a, q):
    """Least-squares values v at the knots q of the piecewise-linear fit to a
    (hat-function basis, tridiagonal normal equations, Thomas algorithm).
    Returns (v, fitted samples)."""
    nq = len(q)
    diag = numpy.zeros(nq)
    off = numpy.zeros(nq)
    rhs = numpy.zeros(nq)
    for s in range(nq - 1):
        h = q[s + 1] - q[s]
        last = q[s + 1] if s == nq - 2 else q[s + 1] - 1
        for j in range(q[s], last + 1):
            t = (j - q[s]) / h
            h0 = 1.0 - t
            diag[s] += h0 * h0
            diag[s + 1] += t * t
            off[s] += h0 * t
            rhs[s] += h0 * a[j]
            rhs[s + 1] += t * a[j]
    # Thomas algorithm (symmetric tridiagonal, positive definite)
    cp = numpy.zeros(nq)
    dp = numpy.zeros(nq)
    cp[0] = off[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, nq):
        den = diag[i] - off[i - 1] * cp[i - 1]
        cp[i] = off[i] / den if i < nq - 1 else 0.0
        dp[i] = (rhs[i] - off[i - 1] * dp[i - 1]) / den
    v = numpy.zeros(nq)
    v[nq - 1] = dp[nq - 1]
    for i in range(nq - 2, -1, -1):
        v[i] = dp[i] - cp[i] * v[i + 1]
    p = numpy.empty(len(a))
    for s in range(nq - 1):
        h = q[s + 1] - q[s]
        for j in range(q[s], q[s + 1] + 1):
            t = (j - q[s]) / h
            p[j] = (1.0 - t) * v[s] + t * v[s + 1]
    return v, p


def pl_fit(a, eps):
    """Piecewise-linear approximation of the samples a (len >= 2), idea L3.

    Knots from _greedy_knots(a, eps * max|a|); then least-squares values at
    the knots (eps <= 0: every sample is a knot, exact). Returns
    (positions, coefficients, sum of the approximation, approximation): for
    any x, sum_j p_j x[i + j] = sum_t c_t D[i + pos_t] with D the double
    prefix sum D[k] = sum_{l<k} S[l], S[l] = sum_{q<l} x[q]. The
    coefficients are the second differences of the zero-padded p.
    """
    a = numpy.asarray(a, dtype=float)
    n = len(a)
    j = numpy.arange(n)
    if eps > 0:
        q = _greedy_knots(a, eps * numpy.max(numpy.abs(a)))
        v, p = _pl_lsq(a, q)
    else:
        q = j.copy()
        v = a.copy()
        p = a.copy()
    s = numpy.diff(v) / numpy.diff(q)  # slopes of the K segments
    pos = [0, 1]
    coef = [v[0], s[0] - v[0]]
    for k in range(1, len(q) - 1):
        pos.append(q[k] + 1)
        coef.append(s[k] - s[k - 1])
    pos += [n, n + 1]
    coef += [-v[-1] - s[-1], v[-1]]
    pos = numpy.array(pos, dtype=numpy.int64)
    coef = numpy.array(coef)
    keep = coef != 0
    return pos[keep], coef[keep], float(numpy.sum(p)), p


def pl_templates(templates, margin, min_length, max_stride, eps):
    """Piecewise-linear templates for the kernel (idea L3).

    Used for templates with length >= min_length, shift stride < max_stride
    and fewer terms than samples; else pl_n[u] = 0 (bins or exact dot
    product). Returns (pl_n, pl_off, pl_n2, pl_off2, pl_pos, pl_c, pl_sum,
    pl_sum2): terms for a (n, off, sum) and for a^2 (n2, off2, sum2) in the
    shared arrays pl_pos, pl_c.
    """
    widths, rows, offsets, lengths, profile, overshoot, sum_a2 = templates
    nu = len(widths)
    pl_n = numpy.zeros(nu, dtype=numpy.int64)
    pl_n2 = numpy.zeros(nu, dtype=numpy.int64)
    pl_off = numpy.zeros(nu, dtype=numpy.int64)
    pl_off2 = numpy.zeros(nu, dtype=numpy.int64)
    pl_sum = numpy.zeros(nu)
    pl_sum2 = numpy.zeros(nu)
    pos_all, c_all = [numpy.zeros(0, dtype=numpy.int64)], [numpy.zeros(0)]
    o = 0
    for u in range(nu):
        length = int(lengths[u])
        if length < max(2, min_length) or shift_stride(widths[u], margin) >= max_stride:
            continue
        a = profile[offsets[u] : offsets[u] + length]
        pos, c, sm, _ = pl_fit(a, eps)
        pos2, c2, sm2, _ = pl_fit(a * a, eps)
        if eps > 0 and len(c) >= length:  # no saving (eps <= 0: exact, for tests)
            continue
        pl_n[u], pl_off[u], pl_sum[u] = len(c), o, sm
        pl_n2[u], pl_off2[u], pl_sum2[u] = len(c2), o + len(c), sm2
        pos_all += [pos, pos2]
        c_all += [c, c2]
        o += len(c) + len(c2)
    return (
        pl_n,
        pl_off,
        pl_n2,
        pl_off2,
        numpy.concatenate(pos_all),
        numpy.concatenate(c_all),
        pl_sum,
        pl_sum2,
    )


def screen_round_budget(y, w, means, maxw):
    """Conservative allowances for both prefix levels and window differences.

    Global data mass protects small windows following large outliers, where
    a window-relative error allowance alone underestimates cancellation.
    The factor includes wrapping and error propagation through both scans.
    """
    n = len(y)
    maxw = int(maxw) + int(maxw) % 2
    m = n + maxw + 1
    eps = 16 * numpy.finfo(numpy.float64).eps
    r = 1 - y
    mass_rw = numpy.sum(numpy.abs(r * w)) + n * abs(means[0])
    mass_w = numpy.sum(w)
    return (
        eps * m * m * mass_rw,
        eps * m * m * (mass_w + n * abs(means[1])),
        eps * m * numpy.sum(r * r * w),
        eps * m * mass_w,
    )


# Proxy tolerance by template length (PERFORMANCE_LOG step 45): a full
# correlation costs ~length, a proxy ~20 terms regardless of length, so long
# templates profit from a tighter proxy that rejects more candidates.
SCREEN_PROXY_EPS = ((0, 0.1), (512, 0.01))
SCREEN_MIN_LENGTH = 64


def proxy_tolerance(rule, length):
    """Proxy PL tolerance for a template length: a float, or ascending
    (min_length, eps) pairs (the last pair with length >= min_length)."""
    if numpy.isscalar(rule):
        return float(rule)
    tol = rule[0][1]
    for lo, value in rule:
        if length >= lo:
            tol = value
    return float(tol)


def screen_templates(
    templates,
    pl,
    eps,
    weighted,
    proxy_eps=SCREEN_PROXY_EPS,
    min_length=SCREEN_MIN_LENGTH,
    round_budget=None,
):
    """Cheap PL proxies with certificates for the unchanged target statistic.

    |AR - AR_proxy| <= sqrt(w_max * ||target - proxy||² * R2).
    |A2 - A2_proxy| <= max|target_A2 - proxy2| * sum(w).
    For a PL target, fit its actual evaluated shape (and its separate A2 fit).
    Use measured residual norms, not the nominal knot-fitting tolerance.
    proxy_eps: a float, or a length rule (see proxy_tolerance).
    """
    widths, rows, offsets, lengths, profile, overshoot, sum_a2 = templates
    nu = len(widths)
    ns = numpy.zeros(nu, dtype=numpy.int64)
    off = ns.copy()
    ns2, off2 = ns.copy(), ns.copy()
    sums, sums2, err, err2 = [numpy.zeros(nu) for _ in range(4)]
    positions = [numpy.zeros(0, dtype=numpy.int64)]
    coefficients = [numpy.zeros(0)]
    offset = 0
    ar_round, a2_round = numpy.zeros(nu), numpy.zeros(nu)
    r2_round = 0.0 if round_budget is None else round_budget[2]
    for u in range(nu):
        if lengths[u] < min_length:
            continue
        a = profile[offsets[u] : offsets[u] + lengths[u]]
        target = pl_fit(a, eps)[3] if pl[0][u] > 0 else a
        tol = proxy_tolerance(proxy_eps, lengths[u])
        pos, c, sm, proxy = pl_fit(target, tol)
        if len(c) >= (pl[0][u] if pl[0][u] else lengths[u]):
            continue
        ns[u], off[u], sums[u] = len(c), offset, sm
        # Include a floating-point allowance even for zero-error proxy fits.
        err[u] = (
            numpy.linalg.norm(target - proxy) + 1e-8 * numpy.linalg.norm(target)
        ) ** 2
        positions.append(pos)
        coefficients.append(c)
        offset += len(c)
        if round_budget is not None:
            ar_round[u] = round_budget[0] * numpy.sum(numpy.abs(c))
        if weighted:
            target2 = pl_fit(a * a, eps)[3] if pl[0][u] > 0 else a * a
            pos, c, sm, proxy = pl_fit(target2, tol)
            ns2[u], off2[u], sums2[u] = len(c), offset, sm
            err2[u] = (
                numpy.max(numpy.abs(target2 - proxy))
                + 1e-8 * numpy.max(numpy.abs(target2))
            )
            positions.append(pos)
            coefficients.append(c)
            offset += len(c)
            if round_budget is not None:
                a2_round[u] = (
                    round_budget[1] * numpy.sum(numpy.abs(c))
                    + err2[u] * round_budget[3]
                )
    if offset == 0:
        return None
    return (
        ns, off, ns2, off2,
        numpy.concatenate(positions), numpy.concatenate(coefficients),
        sums, sums2, err, err2, ar_round, a2_round, r2_round,
    )


class FusedProblem:
    """Precomputed, numba-friendly view of a backends.SearchProblem."""

    def __init__(self, problem):
        self.t = numpy.ascontiguousarray(problem.t, dtype=float)
        self.y = numpy.ascontiguousarray(problem.y, dtype=float)
        dy = numpy.asarray(problem.dy, dtype=float)
        self.inv_dy2 = 1 / dy**2
        self.input_means = _input_means(self.y, self.inv_dy2)
        self.uniform_weights = bool(numpy.all(self.inv_dy2 == self.inv_dy2[0]))
        self.set_storage(numpy.float64)
        self.time_span = float(numpy.max(self.t) - numpy.min(self.t))
        self.index_dtype = sort_index_dtype(len(self.t))
        self.templates = flatten_templates(problem.lc_arr, problem.lc_cache_overview)
        self.problem = problem
        self.prune = True
        self.dtype = numpy.float64
        self.set_binning(0)
        self.set_pl(0)
        # walk fold: configured by set_walk (backends: with the searched
        # periods, before the search; else lazily by the first search)
        self._walk_ready = False
        self.walk = walk_disabled(self.index_dtype)
        self.walk_grid = None

    def set_walk(self, enabled=None, periods=None):
        """Three-gap walk fold (idea L12, exact) if the time stamps lie on a
        regular grid: grid detection (once) and the crossover period P* from
        a few probe periods within the range of `periods` (the searched
        periods; None: the default grid's range). Backends call this before
        the search (SearchBackend.plan); otherwise the first search() does.
        Environment TLS_WALK=0 disables it."""
        if enabled is None:
            enabled = os.environ.get("TLS_WALK", "1") != "0"
        self._walk_ready = True
        self.walk = walk_disabled(self.index_dtype)
        self.walk_grid = None
        if not enabled:
            return
        if not hasattr(self, "_grid"):
            self._grid = regular_grid(self.t, self.index_dtype)
        if self._grid is None:
            return
        span = None
        if periods is not None and len(periods) > 0:
            span = (numpy.min(periods), numpy.max(periods))
        p_star = regular_grid_crossover(
            self.t, self._grid, self.time_span, self.index_dtype, span
        )
        self.walk_grid = self._grid + (p_star,)
        if numpy.isfinite(p_star):
            self.walk = self.walk_grid

    def set_pl(
        self,
        min_length,
        max_stride=4,
        eps=1e-2,
        a2_approx=False,
        t0_coarsen=1,
        scout_every=0,
        ls_depth=False,
    ):
        """Piecewise-linear templates (idea L3, see pl_templates) for
        templates with length >= min_length and shift stride < max_stride;
        min_length <= 0: off. a2_approx: A2 from the window-mean weight
        (idea U2, approximate; for nearly uniform weights). t0_coarsen: coarse
        T0 grid with local refinement (idea L5, approximate; 1: off)."""
        self.pl_args = (min_length, max_stride, eps, a2_approx)
        if min_length <= 0:
            min_length = 1 << 62
        self.pl = pl_templates(
            self.templates,
            float(self.problem.T0_search_margin),
            min_length,
            max_stride,
            eps,
        ) + (
            bool(a2_approx),
            float(numpy.max(self.inv_dy2)),
            int(t0_coarsen),
            int(scout_every),
            True if ls_depth else None,  # None specializes the branch away
        )
        self.screen = None

    def set_screen(self, proxy_eps=SCREEN_PROXY_EPS, min_length=SCREEN_MIN_LENGTH):
        """Enable certified correlation screening without changing the target.

        The exact backends enable this after configuration. The already-cheap
        default PL correlations are faster without another screening stage.
        proxy_eps: float or length rule (see proxy_tolerance); templates
        shorter than min_length keep the direct correlation.
        """
        self.screen = None
        if self.dtype == numpy.float64 and self.prune and not self.pl[-1]:
            budget = screen_round_budget(
                self.y, self.inv_dy2, self.input_means, self.templates[0][-1]
            )
            self.screen = screen_templates(
                self.templates_kernel, self.pl, self.pl_args[2],
                not self.uniform_weights, proxy_eps, min_length, budget,
            )

    def set_storage(self, dtype):
        """Precision of the per-point inputs gathered every period: residuals
        r = 1 - y and weights. float64 (default, exact backends) or float32
        (approximate backends, PERFORMANCE_LOG step 50: |dr| <= 6e-8 |r|, SDE
        changes <= 5e-4, no recovery changes). Prefix sums stay float64."""
        self.storage = numpy.dtype(dtype).type
        self.r_in = (1.0 - self.y).astype(self.storage)
        self.w_in = self.inv_dy2.astype(self.storage)

    def set_precision(self, dtype):
        """float64 (default) or float32 for the dot products (approximate)."""
        self.dtype = numpy.dtype(dtype).type
        self.screen = None
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
        # None if no template is binned: the kernel's binned code is then
        # specialized away (shorter first-run compilation, step 52)
        self.binning = None
        if numpy.any(nbins > 0):
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
        # Use the kernel's profile precision, including experimental float32.
        if getattr(self, "_invariants_dtype", None) != self.dtype:
            kernel_profile = self.templates_kernel[4]
            max_a2 = numpy.array(
                [
                    numpy.max(kernel_profile[o : o + n] ** 2, initial=0.0)
                    for o, n in zip(offsets, lengths)
                ]
            )
            self.invariants = (*self.input_means, max_a2)
            self._invariants_dtype = self.dtype

    def search(self, period):
        if not self._walk_ready:
            self.set_walk()
        p = self.problem
        chi2, row, depth = search_period_fused(
            float(period),
            self.t,
            self.r_in,
            self.w_in,
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
            self.binning,
            *self.pl,
            *self.workspace(),
            self.invariants,
            self.screen,
            self.walk,
        )
        return period, chi2, row, depth

    def workspace(self):
        if getattr(self, "_ws", None) is None:
            maxw = int(self.templates[0][-1]) + 1
            nf, ni, nv = workspace_size(len(self.t), maxw)
            self._ws = (
                numpy.empty(nf),
                numpy.empty(ni, dtype=self.index_dtype),
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
    bins,
    pl_n,
    pl_off,
    pl_n2,
    pl_off2,
    pl_pos,
    pl_c,
    pl_sum,
    pl_sum2,
    pl_a2_approx,
    pl_w_max,
    t0_coarsen,
    scout_every,
    ls_depth,
    n_chunks,
    invariants,
    index_dtype,
    screen=None,
    walk=None,
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
    nf = 4 * n + 6 * m + 16
    ni = 2 * n + 2
    for c_idx in numba.prange(n_chunks):
        fbuf = numpy.empty(nf)
        ibuf = numpy.empty(ni, dtype=index_dtype)
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
                bins,
                pl_n,
                pl_off,
                pl_n2,
                pl_off2,
                pl_pos,
                pl_c,
                pl_sum,
                pl_sum2,
                pl_a2_approx,
                pl_w_max,
                t0_coarsen,
                scout_every,
                ls_depth,
                fbuf,
                ibuf,
                vbuf,
                invariants,
                screen,
                walk,
            )
            chi2[p] = c
            row[p] = r
            depth[p] = d
    return chi2, row, depth

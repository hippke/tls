"""Single and multiple transit events with the GP baseline marginalised.

For a trial transit the depth is linear, so with the GP integrated out

  delta chi2 = (s' C^-1 y)^2 / (s' C^-1 s),   C = K_gp + N_white

The Kalman filter applies L^-1 (C = L L'). For a single transit, L^-1 s is
zero before the transit; the filter runs over the transit and the rest of
the sums follows in closed form from one backward pass (tail_operators).
A periodic signal sums the single events along T0 + n P (multiple events);
cross terms between transits are neglected, which is exact when the
transits are further apart than the filter memory.
"""

import math

import numba
import numpy as np

from transitleastsquares import tls_constants
from transitleastsquares.transit import reference_transit

@numba.njit(cache=True)
def _template(tt, tc, D, table, texp, nsub):
    """Transit template (dip with depth 1 at the bottom, negative),
    averaged over the exposure time."""
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


@numba.njit(cache=True)
def transition_blocks(A):
    """Diagonal blocks of the per-point transitions A (n, d, d), from their
    non-zero pattern over all points: (starts, sizes); a model that is not
    block-diagonal is one block of size d. The models of transitleastsquares.gp
    have blocks of 1 (offset) and 2 (spline, SHO)."""
    n, d = A.shape[0], A.shape[1]
    Z = np.zeros((d, d), np.bool_)
    for i in range(n):
        for r in range(d):
            for c in range(d):
                if A[i, r, c] != 0.0:
                    Z[r, c] = True
    starts = np.empty(d, np.int64)
    sizes = np.empty(d, np.int64)
    nb = 0
    s = 0
    while s < d:
        e = s
        k = s
        while k <= e:  # grow the block while its rows/columns couple to later indices
            for c in range(d):
                if (Z[k, c] or Z[c, k]) and c > e:
                    e = c
            k += 1
        starts[nb] = s
        sizes[nb] = e - s + 1
        nb += 1
        s = e + 1
    return starts[:nb], sizes[:nb]


@numba.njit(cache=True, parallel=True)
def pack_blocks(A, starts, sizes):
    """The diagonal blocks of A per point: Ap[i, k] = A[i, block k]
    (zero-padded to the largest block)."""
    n, nb = A.shape[0], len(starts)
    B = 1
    for k in range(nb):
        B = max(B, sizes[k])
    Ap = np.zeros((n, nb, B, B))
    for i in numba.prange(n):
        for k in range(nb):
            s0 = starts[k]
            for r in range(sizes[k]):
                for c in range(sizes[k]):
                    Ap[i, k, r, c] = A[i, s0 + r, s0 + c]
    return Ap


@numba.njit(cache=True, inline="always")
def _predict_blocks(Ap, starts, sizes, i, x, xp):
    """xp = A_i x for block-diagonal A_i: the products with the zeros outside
    the blocks are skipped, the rest summed in the same order (0 + first
    term + second term, no fused multiply-add), so the result equals the
    dense product bit for bit. Blocks of 1 and 2 written out (no inner loop)."""
    for k in range(len(starts)):
        s0, sz = starts[k], sizes[k]
        if sz == 2:
            x0 = x[s0]
            x1 = x[s0 + 1]
            xp[s0] = (0.0 + Ap[i, k, 0, 0] * x0) + Ap[i, k, 0, 1] * x1
            xp[s0 + 1] = (0.0 + Ap[i, k, 1, 0] * x0) + Ap[i, k, 1, 1] * x1
        elif sz == 1:
            xp[s0] = 0.0 + Ap[i, k, 0, 0] * x[s0]
        else:
            for r in range(sz):
                acc = 0.0
                for c in range(sz):
                    acc += Ap[i, k, r, c] * x[s0 + c]
                xp[s0 + r] = acc


@numba.njit(cache=True, inline="always")
def _template_inv(tt, tc, invD, table, texp, nsub):
    """_template with the reciprocal duration (no division per sample)."""
    L = table.shape[0]
    acc = 0.0
    for k in range(nsub):
        off = (k + 0.5) / nsub - 0.5
        u = (tt + off * texp - tc) * invD
        if -0.5 < u < 0.5:
            x = (u + 0.5) * (L - 1)
            i = min(int(x), L - 2)
            f = x - i
            acc += table[i] * (1 - f) + table[i + 1] * f
    return -acc / nsub


@numba.njit(cache=True, nogil=True)
def _tail_step(A, K, S, H, ey, i, Wn, wn, Wm, wm, F, gv, tmp):
    """One backward step m = i - 1 of the tail recursion: from W_{m+1} = Wn,
    w_{m+1} = wn to Wm, wm; returns F_i (in F) for composing maps."""
    d = A.shape[1]
    s = 1.0 / math.sqrt(S[i])
    for k in range(d):
        hk = 0.0
        for l in range(d):
            hk += H[l] * A[i, l, k]
        gv[k] = -hk * s
    for r in range(d):  # F = A - K (H A)
        for c in range(d):
            F[r, c] = A[i, r, c] + K[i, r] * gv[c] / s
    for r in range(d):  # tmp = W_{m+1} F
        for c in range(d):
            acc = 0.0
            for l in range(d):
                acc += Wn[r, l] * F[l, c]
            tmp[r, c] = acc
    for r in range(d):  # W_m = g g' + F' tmp
        for c in range(d):
            acc = gv[r] * gv[c]
            for l in range(d):
                acc += F[l, r] * tmp[l, c]
            Wm[r, c] = acc
        acc = ey[i] * gv[r]
        for l in range(d):
            acc += F[l, r] * wn[l]
        wm[r] = acc


@numba.njit(cache=True, parallel=True)
def tail_operators(A, K, S, H, ey, nseg=1):
    """Closed form of the single-event sums after a template has ended.

    With zero template input, the normalised innovations are e_i = g_i' x_{i-1}
    and the state evolves as x_i = F_i x_{i-1}, with F_i = (I - K_i H) A_i and
    g_i = -(H A_i)' / sqrt(S_i). From the filtered state x_m at index m, the
    rest of the sums is a_tail = w_m' x_m and b_tail = x_m' W_m x_m, where
      W_m = g_{m+1} g_{m+1}' + F_{m+1}' W_{m+1} F_{m+1}
      w_m = ey_{m+1} g_{m+1} + F_{m+1}' w_{m+1}
    (one backward pass; W is data-independent).
    nseg > 1: the recursion is linear in (W, w), so each segment first
    composes its map W_lo = Phi' W_b Phi + C, w_lo = Phi' w_b + c from its
    boundary b (in parallel), the maps are joined backwards in order, and
    every segment recomputes its W, w from the known boundary (in parallel).
    Exact up to rounding."""
    n, d = A.shape[0], A.shape[1]
    W = np.zeros((n, d, d))
    w = np.zeros((n, d))
    edges = np.empty(nseg + 1, np.int64)  # segment k: m in [edges[k], edges[k + 1])
    for k in range(nseg + 1):
        edges[k] = (k * (n - 1)) // nseg  # the boundary of the last is n - 1: W, w = 0
    Phi = np.empty((nseg, d, d))
    C = np.zeros((nseg, d, d))
    cc = np.zeros((nseg, d))
    for k in numba.prange(1, nseg):  # maps; the first segment's start is not needed
        F = np.empty((d, d))
        gv = np.empty(d)
        tmp = np.empty((d, d))
        Wm = np.empty((d, d))
        wm = np.empty(d)
        Ph = np.eye(d)
        Pn = np.empty((d, d))
        for m in range(edges[k + 1] - 1, edges[k] - 1, -1):
            _tail_step(A, K, S, H, ey, m + 1, C[k], cc[k], Wm, wm, F, gv, tmp)
            C[k] = Wm
            cc[k] = wm
            for r in range(d):  # Phi = Phi F
                for c2 in range(d):
                    acc = 0.0
                    for l in range(d):
                        acc += Ph[r, l] * F[l, c2]
                    Pn[r, c2] = acc
            Ph[:] = Pn
        Phi[k] = Ph
    Wb = np.zeros((nseg, d, d))  # W, w at each segment's boundary edges[k + 1]
    wb = np.zeros((nseg, d))
    tmp = np.empty((d, d))
    for k in range(nseg - 2, -1, -1):  # W(edges[k+1]) = Phi' Wb Phi + C of segment k + 1
        q = k + 1
        for r in range(d):
            for c2 in range(d):
                acc = 0.0
                for l in range(d):
                    acc += Wb[q, r, l] * Phi[q, l, c2]
                tmp[r, c2] = acc
        for r in range(d):
            for c2 in range(d):
                acc = C[q, r, c2]
                for l in range(d):
                    acc += Phi[q, l, r] * tmp[l, c2]
                Wb[k, r, c2] = acc
            acc = cc[q, r]
            for l in range(d):
                acc += Phi[q, l, r] * wb[q, l]
            wb[k, r] = acc
    for k in numba.prange(nseg):  # W, w from the known boundaries
        F = np.empty((d, d))
        gv = np.empty(d)
        tmp2 = np.empty((d, d))
        Wn = Wb[k].copy()
        wn = wb[k].copy()
        for m in range(edges[k + 1] - 1, edges[k] - 1, -1):
            _tail_step(A, K, S, H, ey, m + 1, Wn, wn, W[m], w[m], F, gv, tmp2)
            Wn = W[m]
            wn = w[m]
    return W, w


@numba.njit(cache=True)
def _single(t, ey, Ap, starts, sizes, K, H, hidx, isS, W, w, tcm, D, table, texp, nsub, x, xp):
    """a = (L^-1 s)'(L^-1 y) and b = |L^-1 s|^2 for one transit at tcm with
    duration D: the filter runs over the template support only, the rest is
    the closed-form tail (tail_operators). x, xp: work arrays of length d."""
    n, d = len(t), len(x)  # Ap, starts, sizes: the diagonal blocks of the transitions (pack_blocks)
    t_start = tcm - 0.5 * D - 0.5 * texp
    t_end = tcm + 0.5 * D + 0.5 * texp
    invD = 1.0 / D
    i = np.searchsorted(t, t_start)
    x[:] = 0.0
    aa = 0.0
    bb = 0.0
    last = -1
    while i < n and t[i] < t_end:
        _predict_blocks(Ap, starts, sizes, i, x, xp)  # predict
        hx = 0.0
        for r in hidx:  # the zeros of H add nothing
            hx += H[r] * xp[r]
        innov = _template_inv(t[i], tcm, invD, table, texp, nsub) - hx
        e = innov * isS[i]
        aa += e * ey[i]
        bb += e * e
        for r in range(d):  # update
            x[r] = xp[r] + K[i, r] * innov
        last = i
        i += 1
    if last >= 0:
        for r in range(d):
            aa += w[last, r] * x[r]
            acc = 0.0
            for c in range(d):
                acc += W[last, r, c] * x[c]
            bb += x[r] * acc
    return aa, bb


@numba.njit(cache=True, parallel=True)
def single_events(t, ey, A, K, S, H, W, w, tc, durs, table, texp, nsub):
    """a[m, 0, j] and b[m, 1, j] for a single transit at tc[m] with duration
    durs[j], on the full epoch grid; layout (M, 2, J) (the uniform fold)."""
    M, J, d = len(tc), len(durs), A.shape[1]
    starts, sizes = transition_blocks(A)
    Ap = pack_blocks(A, starts, sizes)
    isS = 1.0 / np.sqrt(S)
    hidx = np.nonzero(H)[0]
    ab = np.zeros((M, 2, J))  # a and b of one k adjacent, each contiguous in j
    for mj in numba.prange(M * J):
        m, j = mj // J, mj % J
        x = np.zeros(d)
        xp = np.empty(d)
        aa, bb = _single(t, ey, Ap, starts, sizes, K, H, hidx, isS, W, w, tc[m], durs[j], table, texp, nsub, x, xp)
        ab[m, 0, j] = aa
        ab[m, 1, j] = bb
    return ab


def compact_offsets(M, mstep):
    """Start of each duration's block in the compact layout: duration j keeps
    the epochs m = kk * mstep[j], kk < (M - 1) // mstep[j] + 1."""
    sizes = (M - 1) // np.asarray(mstep, np.int64) + 1
    return np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)


SE_ROWS = 64  # rows per block of the compact single events
SE_TASKS = 256  # parallel tasks; each takes every SE_TASKS-th block


def single_events_compact(t, ey, A, K, S, H, W, w, tc, durs, table, texp, nsub, mstep, off):
    """Single events for the strided fold, compact (see
    _single_events_compact); computed in lanes of neighbouring events
    (_single_events_lanes, bit-identical)."""
    return _single_events_lanes(t, ey, np.zeros(0), A, K, S, H, W, w, tc, durs, table, texp, nsub, mstep, off,
                                False)


@numba.njit(cache=True, parallel=True)
def _single_events_compact(t, ey, A, K, S, H, W, w, tc, durs, table, texp, nsub, mstep, off):
    """Single events for the strided fold, compact: duration j only at the
    epochs tc[kk * mstep[j]], stored contiguously from row off[j]; a and b
    of one epoch side by side (row = (a, b)). Memory sum_j M / mstep[j]
    instead of J x M, and the fold reads consecutive rows.
    The cost of a row grows with its duration and the rows are sorted by
    duration, so the rows go in blocks of SE_ROWS round-robin to SE_TASKS
    tasks (balanced; every row is independent, so the result does not
    depend on the number of threads)."""
    J, d = len(durs), A.shape[1]
    starts, sizes = transition_blocks(A)
    Ap = pack_blocks(A, starts, sizes)
    isS = 1.0 / np.sqrt(S)
    hidx = np.nonzero(H)[0]
    total = off[J]
    ab = np.zeros((total, 2))
    nblocks = (total + SE_ROWS - 1) // SE_ROWS
    for task in numba.prange(SE_TASKS):
        x = np.zeros(d)
        xp = np.empty(d)
        for blk in range(task, nblocks, SE_TASKS):
            i0 = blk * SE_ROWS
            j = np.searchsorted(off, i0, side="right") - 1
            for i in range(i0, min(i0 + SE_ROWS, total)):
                while i >= off[j + 1]:
                    j += 1
                m = (i - off[j]) * mstep[j]
                aa, bb = _single(t, ey, Ap, starts, sizes, K, H, hidx, isS, W, w, tc[m], durs[j], table, texp, nsub, x, xp)
                ab[i, 0] = aa
                ab[i, 1] = bb
    return ab


@numba.njit(cache=True)
def _single_split(t, z, eyd, Ap, starts, sizes, K, H, hidx, isS, W, tcm, D, table, texp, nsub, x, xp):
    """_single through C^-1 y = N^-1 (y - f) (f: the GP posterior mean):
    a = s' C^-1 y = sum s z over the template support, z = (y - f) / sigma^2,
    without the filter; b = |L^-1 s|^2 as in _single (same operations, so
    bit-identical), which needs no data at all. The search zeroes the first
    n_diffuse normalised innovations (diffuse start); their terms
    (L^-1 s)_i (L^-1 y)_i, i < n_diffuse, are removed here with eyd (those
    innovations before zeroing), so a equals _single's up to rounding
    (spline GP: up to a few % for events that touch the first n_diffuse
    samples, where its three diffuse states are ill-conditioned)."""
    n, d = len(t), len(x)
    t_start = tcm - 0.5 * D - 0.5 * texp
    t_end = tcm + 0.5 * D + 0.5 * texp
    invD = 1.0 / D
    i = np.searchsorted(t, t_start)
    x[:] = 0.0
    aa = 0.0
    bb = 0.0
    last = -1
    while i < n and t[i] < t_end:
        _predict_blocks(Ap, starts, sizes, i, x, xp)  # predict
        hx = 0.0
        for r in hidx:  # the zeros of H add nothing
            hx += H[r] * xp[r]
        s = _template_inv(t[i], tcm, invD, table, texp, nsub)
        aa += s * z[i]
        innov = s - hx
        e = innov * isS[i]
        if i < len(eyd):
            aa -= e * eyd[i]
        bb += e * e
        for r in range(d):  # update
            x[r] = xp[r] + K[i, r] * innov
        last = i
        i += 1
    if last >= 0:
        for r in range(d):
            acc = 0.0
            for c in range(d):
                acc += W[last, r, c] * x[c]
            bb += x[r] * acc
    return aa, bb


def single_events_split(t, z, eyd, A, K, S, H, W, tc, durs, table, texp, nsub, mstep, off):
    """Split single events (see _single_events_split); computed in lanes of
    neighbouring events (_single_events_lanes, bit-identical)."""
    n, d = len(t), A.shape[1]
    return _single_events_lanes(t, z, np.ascontiguousarray(eyd, dtype=np.float64), A, K, S, H, W,
                                np.zeros((1, d)), tc, durs, table, texp, nsub, mstep, off, True)


@numba.njit(cache=True, parallel=True)
def _single_events_split(t, z, eyd, A, K, S, H, W, tc, durs, table, texp, nsub, mstep, off):
    """single_events_compact through the identity of _single_split: z =
    (y - f) / sigma^2 from the smoother (gp.statespace.smoothed_mean), eyd the
    first n_diffuse normalised innovations (before the search zeroes them), W
    from tail_operators (data-independent). b is bit-identical to
    single_events_compact, a equal up to rounding; b depends only on the
    times, errors and hyperparameters (reusable when only the flux changes),
    and the numerator is a white-noise correlation (the GPU path)."""
    J, d = len(durs), A.shape[1]
    starts, sizes = transition_blocks(A)
    Ap = pack_blocks(A, starts, sizes)
    isS = 1.0 / np.sqrt(S)
    hidx = np.nonzero(H)[0]
    total = off[J]
    ab = np.zeros((total, 2))
    nblocks = (total + SE_ROWS - 1) // SE_ROWS
    for task in numba.prange(SE_TASKS):
        x = np.zeros(d)
        xp = np.empty(d)
        for blk in range(task, nblocks, SE_TASKS):
            i0 = blk * SE_ROWS
            j = np.searchsorted(off, i0, side="right") - 1
            for i in range(i0, min(i0 + SE_ROWS, total)):
                while i >= off[j + 1]:
                    j += 1
                m = (i - off[j]) * mstep[j]
                aa, bb = _single_split(t, z, eyd, Ap, starts, sizes, K, H, hidx, isS, W, tc[m], durs[j], table, texp, nsub,
                                       x, xp)
                ab[i, 0] = aa
                ab[i, 1] = bb
    return ab


def numerator_weights(ey, S, K, w):
    """z = C^-1 y from the filter and the tail recursion, without a smoother:
    a single event's numerator a = sum_i s_i z_i, with z_i = da / ds_i =
    ey_i / sqrt(S_i) + K_i . w_i (s_i enters e_i directly and the filtered
    state through the gain K_i; w_i gives the rest of the sum after it).
    Uses the search's ey (first n_diffuse zeroed), so the split single events
    equal the Kalman ones up to rounding (single_events_split with an empty
    eyd)."""
    z = ey / np.sqrt(S)
    z += np.einsum("ir,ir->i", K, w)
    return z


LANES = 4  # neighbouring events filtered together (SIMD across events)


@numba.njit(cache=True, nogil=True)
def _single_lanes(t, data, eyd, Ap, starts, sizes, K, H, hidx, isS, W, w, tcs, nl, D, table, texp, nsub, split,
                  x, xp, aa, bb, lo, hi, last, sl, act):
    """_single (split=False: data = ey) or _single_split (split=True: data =
    z) for the nl <= LANES events at tcs[:nl] of one duration D, in lanes:
    one loop over the union of their windows, each lane masked to its own
    window (before it the state stays 0, after it frozen), so every lane
    equals its single-event result bit for bit (1 * v + 0 * x = v). The
    independent lanes fill the vector units that one latency-bound recursion
    leaves idle (1.7x for 4 lanes). Results in aa, bb."""
    n, d = len(t), K.shape[1]
    L = LANES  # compile-time constant: the lane loops unroll / vectorise
    invD = 1.0 / D
    half = 0.5 * D + 0.5 * texp
    for l in range(L):
        c = tcs[min(l, nl - 1)]
        lo[l] = c - half
        hi[l] = c + half if l < nl else c - half
        last[l] = -1
        aa[l] = 0.0
        bb[l] = 0.0
        for r in range(d):
            x[r, l] = 0.0
    i = np.searchsorted(t, lo[0])
    tend = hi[nl - 1]
    nd = len(eyd)
    while i < n and t[i] < tend:
        ti = t[i]
        for l in range(L):
            act[l] = 1.0 if (ti >= lo[l]) and (ti < hi[l]) else 0.0
            sl[l] = _template_inv(ti, tcs[min(l, nl - 1)], invD, table, texp, nsub)
        for b in range(len(starts)):  # predict, block-diagonal, all lanes
            s0 = starts[b]
            if sizes[b] == 2:
                a00 = Ap[i, b, 0, 0]
                a01 = Ap[i, b, 0, 1]
                a10 = Ap[i, b, 1, 0]
                a11 = Ap[i, b, 1, 1]
                for l in range(L):
                    x0 = x[s0, l]
                    x1 = x[s0 + 1, l]
                    xp[s0, l] = (0.0 + a00 * x0) + a01 * x1
                    xp[s0 + 1, l] = (0.0 + a10 * x0) + a11 * x1
            else:
                a00 = Ap[i, b, 0, 0]
                for l in range(L):
                    xp[s0, l] = 0.0 + a00 * x[s0, l]
        isi = isS[i]
        di = data[i]
        for l in range(L):
            hx = 0.0
            for q in hidx:
                hx += H[q] * xp[q, l]
            sv = sl[l]
            innov = (sv - hx) * act[l]
            e = innov * isi
            if split:
                aa[l] += (sv * act[l]) * di
                if i < nd:
                    aa[l] -= e * eyd[i]
            else:
                aa[l] += e * di
            bb[l] += e * e
            sl[l] = innov
        for r in range(d):  # update; inactive lanes keep their state
            kr = K[i, r]
            for l in range(L):
                x[r, l] = act[l] * (xp[r, l] + kr * sl[l]) + (1.0 - act[l]) * x[r, l]
        for l in range(L):
            if act[l] > 0.0:
                last[l] = i
        i += 1
    for l in range(nl):
        if last[l] >= 0:
            for r in range(d):
                if not split:
                    aa[l] += w[last[l], r] * x[r, l]
                acc = 0.0
                for c in range(d):
                    acc += W[last[l], r, c] * x[c, l]
                bb[l] += x[r, l] * acc


@numba.njit(cache=True, parallel=True)
def _single_events_lanes(t, data, eyd, A, K, S, H, W, w, tc, durs, table, texp, nsub, mstep, off, split):
    """single_events_compact (split=False) / single_events_split (split=True)
    with _single_lanes: rows of one duration in groups of LANES."""
    J, d = len(durs), A.shape[1]
    starts, sizes = transition_blocks(A)
    Ap = pack_blocks(A, starts, sizes)
    isS = 1.0 / np.sqrt(S)
    hidx = np.nonzero(H)[0]
    total = off[J]
    ab = np.zeros((total, 2))
    nblocks = (total + SE_ROWS - 1) // SE_ROWS
    for task in numba.prange(SE_TASKS):
        x = np.zeros((d, LANES))
        xp = np.zeros((d, LANES))
        aa = np.zeros(LANES)
        bb = np.zeros(LANES)
        lo = np.zeros(LANES)
        hi = np.zeros(LANES)
        last = np.zeros(LANES, np.int64)
        sl = np.zeros(LANES)
        act = np.zeros(LANES)
        tcs = np.zeros(LANES)
        for blk in range(task, nblocks, SE_TASKS):
            i0 = blk * SE_ROWS
            i1 = min(i0 + SE_ROWS, total)
            j = np.searchsorted(off, i0, side="right") - 1
            i = i0
            while i < i1:
                while i >= off[j + 1]:
                    j += 1
                nl = min(LANES, i1 - i, off[j + 1] - i)  # consecutive rows of duration j
                for l in range(nl):
                    tcs[l] = tc[(i + l - off[j]) * mstep[j]]
                _single_lanes(t, data, eyd, Ap, starts, sizes, K, H, hidx, isS, W, w, tcs, nl, durs[j], table,
                              texp, nsub, split, x, xp, aa, bb, lo, hi, last, sl, act)
                for l in range(nl):
                    ab[i + l, 0] = aa[l]
                    ab[i + l, 1] = bb[l]
                i += nl
    return ab


def single_event_sums(t, ey, A, K, S, H, tc, durs, table, texp, nsub=7):
    """Convenience wrapper: tail operators + single events."""
    W, w = tail_operators(A, K, S, H, ey)
    ab = single_events(t, ey, A, K, S, H, W, w, tc, durs, table, texp, nsub)
    return ab[:, 0, :], ab[:, 1, :]


@numba.njit(cache=True)
def _log_bf(A, B, s2):
    """log Bayes factor transit vs no transit for a half-normal depth prior
    with scale sqrt(s2)."""
    vp = 1.0 / (B + 1.0 / s2)
    mu = A * vp
    z = mu / math.sqrt(vp)
    phi = 0.5 * math.erfc(-z / math.sqrt(2.0))
    lphi = math.log(phi) if phi > 1e-300 else -0.5 * z * z - math.log(-z * math.sqrt(2 * math.pi))
    return math.log(2.0) - 0.5 * math.log1p(s2 * B) + 0.5 * A * A * vp + lphi


@numba.njit(cache=True, parallel=True)
def multiple_events(ab, periods, delta, jlo, jhi, ntr_min, s2, bayes):
    """Best periodic signal per period: delta chi2 = (sum a)^2 / sum b over
    the transits T0 + n P (positive depths, >= ntr_min transits with data).
    With bayes=True also log(sum over T0 and durations of the Bayes factor),
    which costs most of the time (several transcendentals per trial)."""
    M, J = ab.shape[0], ab.shape[2]
    NP = len(periods)
    dchi2 = np.zeros(NP)
    best_k0 = np.zeros(NP, np.int64)
    best_j = np.zeros(NP, np.int64)
    best_A = np.zeros(NP)
    best_B = np.zeros(NP)
    lse = np.full(NP, -np.inf)
    ntrials = np.zeros(NP)
    for p in numba.prange(NP):
        P = periods[p]
        j0, j1 = jlo[p], jhi[p] + 1
        nph = max(1, int(P / delta))
        ntr = int(M * delta / P) + 2
        off = np.empty(ntr, np.int64)  # transit offsets in grid steps
        for nn in range(ntr):
            off[nn] = int(round(nn * P / delta))
        sA = np.empty(J)
        sB = np.empty(J)
        sN = np.empty(J, np.int64)
        mx = -np.inf
        acc = 0.0
        for k0 in range(nph):
            sA[j0:j1] = 0.0
            sB[j0:j1] = 0.0
            sN[j0:j1] = 0
            for nn in range(ntr):
                k = k0 + off[nn]
                if k >= M:
                    break
                for j in range(j0, j1):  # branch-free: a = b = 0 without data
                    bb = ab[k, 1, j]
                    sA[j] += ab[k, 0, j]
                    sB[j] += bb
                    sN[j] += bb > 0
            for j in range(j0, j1):
                if sN[j] < ntr_min or sB[j] <= 0:
                    continue
                ntrials[p] += 1
                if bayes:
                    lb = _log_bf(sA[j], sB[j], s2)
                    if lb > mx:  # online logsumexp
                        acc = acc * math.exp(mx - lb) + 1.0 if mx > -np.inf else 1.0
                        mx = lb
                    else:
                        acc += math.exp(lb - mx)
                if sA[j] > 0:
                    d2 = sA[j] * sA[j] / sB[j]
                    if d2 > dchi2[p]:
                        dchi2[p] = d2
                        best_k0[p] = k0
                        best_j[p] = j
                        best_A[p] = sA[j]
                        best_B[p] = sB[j]
        if acc > 0:
            lse[p] = mx + math.log(acc)
    return dchi2, best_k0, best_j, best_A, best_B, lse, ntrials


@numba.njit(cache=True, parallel=True)
def multiple_events_strided(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, s2, bayes):
    """Like multiple_events, but duration j only uses every mstep[j]-th point
    of the grid: its transit times lie on a grid of step mstep[j] * delta
    (a coarser T0 grid for long durations). ab is the compact layout of
    single_events_compact (duration j in rows off[j]...). With all mstep = 1
    the result equals multiple_events."""
    NP = len(periods)
    dchi2 = np.zeros(NP)
    best_k0 = np.zeros(NP, np.int64)
    best_j = np.zeros(NP, np.int64)
    best_A = np.zeros(NP)
    best_B = np.zeros(NP)
    lse = np.full(NP, -np.inf)
    ntrials = np.zeros(NP)
    for p in numba.prange(NP):
        P = periods[p]
        mx = -np.inf
        acc = 0.0
        toff = np.empty(int(M * delta / P) + 2, np.int64)
        for j in range(jlo[p], jhi[p] + 1):
            m = mstep[j]
            dj = m * delta
            Mj = off[j + 1] - off[j]  # coarse grid: k = m * kk
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
                if sN < ntr_min or sB <= 0:
                    continue
                ntrials[p] += 1
                if bayes:
                    lb = _log_bf(sA, sB, s2)
                    if lb > mx:  # online logsumexp
                        acc = acc * math.exp(mx - lb) + 1.0 if mx > -np.inf else 1.0
                        mx = lb
                    else:
                        acc += math.exp(lb - mx)
                if sA > 0:
                    d2 = sA * sA / sB
                    if d2 > dchi2[p]:
                        dchi2[p] = d2
                        best_k0[p] = kk0 * m
                        best_j[p] = j
                        best_A[p] = sA
                        best_B[p] = sB
        if acc > 0:
            lse[p] = mx + math.log(acc)
    return dchi2, best_k0, best_j, best_A, best_B, lse, ntrials


FOLD_ACC_BYTES = 512 * 1024  # accumulator budget of a period group (with L2 room for the chunk)
FOLD_GROUP_MAX = 16  # periods per group
FOLD_CHUNK = 16384  # rows of single events per chunk
FOLD_TASKS = 256  # parallel tasks; each takes every FOLD_TASKS-th group


def fold_groups(periods, delta, mstep=None, jlo=None, jhi=None, budget=FOLD_ACC_BYTES, gmax=FOLD_GROUP_MAX):
    """Starts of groups of consecutive periods for multiple_events_blocked:
    the per-epoch sums of a group (16 B per epoch and period) fit the budget.
    With mstep, jlo, jhi (the fold's arguments) a period's sums are counted at
    the coarsest epoch grid it uses, P / (min mstep[jlo..jhi] delta), as the
    kernels allocate them; without, at the finest grid P / delta (groups 24-180x
    smaller than needed on a PLATO light curve: 1.4 periods on average, so
    a chunk of single events served about one period; 2026-11-12)."""
    used = mstep is not None
    if used:
        mstep = np.asarray(mstep, np.int64)
        jlo, jhi = np.asarray(jlo, np.int64), np.asarray(jhi, np.int64)
    starts = [0]
    acc = 0
    for p, P in enumerate(periods):
        if not used:
            need = 16 * (int(P / delta) + 1)
        elif jhi[p] >= jlo[p]:
            need = 16 * (int(P / (delta * int(mstep[jlo[p]:jhi[p] + 1].min()))) + 1)
        else:
            need = 16  # no durations
        if p > starts[-1] and (acc + need > budget or p - starts[-1] >= gmax):
            starts.append(p)
            acc = 0
        acc += need
    starts.append(len(periods))
    return np.array(starts, np.int64)


@numba.njit(cache=True)
def _min_step(mstep, j0, j1):
    """The smallest epoch step (in units of delta) of the durations j0..j1, 0 if there are none: a period's
    per-epoch sums never need more than P / (this step x delta) + 1 entries (nph of multiple_events_blocked)."""
    m = 0
    for j in range(j0, j1 + 1):
        if m == 0 or mstep[j] < m:
            m = mstep[j]
    return m


@numba.njit(cache=True, parallel=True)
def multiple_events_blocked(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, gstart, chunk, depth_min=0.0,
                            screen_hi=None):
    """multiple_events_strided without Bayes factors, identical results, ~5x
    faster on long light curves (a Kepler star: 1200 s -> ~250 s on one core).

    The fold is limited by memory traffic: every period reads all single
    events of its durations once (~100 MB for a Kepler star), at a stride of
    P / step rows. Here (1) the loops are swapped: for every transit the
    contiguous block of epochs is added to per-epoch sums, so memory is read
    sequentially; (2) a group of consecutive periods (gstart, fold_groups)
    walks through each duration's rows in chunks, so a chunk is loaded into
    the cache once per group, not once per period; (3) the number of
    transits with data (>= ntr_min) is counted only when an epoch would
    become the best of its period. Per epoch the transits are added in the
    same order as in multiple_events_strided, so the sums are bit-identical.
    Groups go round-robin to FOLD_TASKS tasks (balanced). The trial count
    (ntrials) is not computed (zeros); it is needed only for Bayes factors.
    depth_min: only depths (sum a / sum b) >= depth_min (the fold of the TLS
    backend "events"; 0 for the GP search).
    screen_hi: None, or the duration screen of fold_ffa (PERF_PLAN item 13):
    only even durations, jlo and screen_hi (the period's longest allowed
    duration) are folded."""
    NP = len(periods)
    NG = len(gstart) - 1
    dchi2 = np.zeros(NP)
    best_k0 = np.zeros(NP, np.int64)
    best_j = np.zeros(NP, np.int64)
    best_A = np.zeros(NP)
    best_B = np.zeros(NP)
    for task in numba.prange(FOLD_TASKS):
        for g in range(task, NG, FOLD_TASKS):
            p0, p1 = gstart[g], gstart[g + 1]
            G = p1 - p0
            nmax = 0
            jmin, jmax = jlo[p0], jhi[p0]
            for p in range(p0, p1):
                ms = _min_step(mstep, jlo[p], jhi[p])  # (0: no durations, e.g. all left to the FFA)
                nmax = max(nmax, int(periods[p] / (ms * delta)) + 1 if ms > 0 else 1)
                jmin = min(jmin, jlo[p])
                jmax = max(jmax, jhi[p])
            sA = np.empty((G, nmax))
            sB = np.empty((G, nmax))
            nph = np.empty(G, np.int64)
            ncur = np.empty(G, np.int64)
            for j in range(jmin, jmax + 1):
                m = mstep[j]
                dj = m * delta
                Mj = off[j + 1] - off[j]
                o = off[j]
                for q in range(G):
                    p = p0 + q
                    nph[q] = max(1, int(periods[p] / dj)) if jlo[p] <= j <= jhi[p] else 0
                    if screen_hi is not None:
                        if j % 2 != 0 and j != jlo[p] and j != screen_hi[p]:
                            nph[q] = 0
                    ncur[q] = 0
                    sA[q, : nph[q]] = 0.0
                    sB[q, : nph[q]] = 0.0
                for c0 in range(0, Mj, chunk):
                    c1 = min(c0 + chunk, Mj)
                    for q in range(G):
                        npq = nph[q]
                        if npq == 0:
                            continue
                        P = periods[p0 + q]
                        qa = sA[q]
                        qb = sB[q]
                        nn = ncur[q]
                        while True:  # transits whose epoch block overlaps the chunk
                            base = int(round(nn * P / dj))
                            if base >= c1:
                                break
                            end = min(base + npq, Mj)
                            s = o + base
                            for k in range(max(base, c0) - base, min(end, c1) - base):
                                qa[k] += ab[s + k, 0]
                                qb[k] += ab[s + k, 1]
                            if end > c1:
                                break
                            nn += 1
                        ncur[q] = nn
                for q in range(G):
                    p = p0 + q
                    P = periods[p]
                    for kk0 in range(nph[q]):
                        A = sA[q, kk0]
                        B = sB[q, kk0]
                        if B <= 0 or A <= 0 or A < depth_min * B:
                            continue
                        d2 = A * A / B
                        if d2 <= dchi2[p]:
                            continue
                        cnt = 0  # transits with data, only for a new best (rare)
                        nn = 0
                        while True:
                            kk = kk0 + int(round(nn * P / dj))
                            if kk >= Mj:
                                break
                            cnt += ab[o + kk, 1] > 0
                            nn += 1
                        if cnt < ntr_min:
                            continue
                        dchi2[p] = d2
                        best_k0[p] = kk0 * m
                        best_j[p] = j
                        best_A[p] = A
                        best_B[p] = B
    return dchi2, best_k0, best_j, best_A, best_B, np.zeros(NP)


@numba.njit(cache=True, nogil=True)
def exact_path(ab, o, Mj, P, dj, kk0, ntr_min, depth_min):
    """Brute-force path of epoch kk0 (duration rows ab[o:o + Mj], epoch step
    dj), as multiple_events_blocked (same rounding of n P / dj, same order of
    the sums): (dchi2 or 0, A, B)."""
    A = 0.0
    B = 0.0
    cnt = 0
    nn = 0
    while True:
        kk = kk0 + int(round(nn * P / dj))
        if kk >= Mj:
            break
        bb = ab[o + kk, 1]
        A += ab[o + kk, 0]
        B += bb
        cnt += bb > 0
        nn += 1
    if cnt < ntr_min or B <= 0 or A <= 0 or A < depth_min * B:
        return 0.0, A, B
    return A * A / B, A, B


@numba.njit(cache=True, parallel=True)
def multiple_events_blocked_coarse(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, gstart, chunk, depth_min,
                                   screen_hi, coarse):
    """multiple_events_blocked on every coarse-th epoch (PERF_PLAN item 12,
    approximate): per period and duration the best coarse epoch (with
    >= ntr_min transits), then its coarse - 1 neighbours on each side on the
    exact path. Every value is an exact brute-force path value; ~coarse x
    fewer cells. screen_hi: None or the duration screen (as
    multiple_events_blocked)."""
    NP = len(periods)
    NG = len(gstart) - 1
    dchi2 = np.zeros(NP)
    best_k0 = np.zeros(NP, np.int64)
    best_j = np.zeros(NP, np.int64)
    best_A = np.zeros(NP)
    best_B = np.zeros(NP)
    c = coarse
    for task in numba.prange(FOLD_TASKS):
        for g in range(task, NG, FOLD_TASKS):
            p0, p1 = gstart[g], gstart[g + 1]
            G = p1 - p0
            nmax = 0
            jmin, jmax = jlo[p0], jhi[p0]
            for p in range(p0, p1):
                ms = _min_step(mstep, jlo[p], jhi[p])
                nmax = max(nmax, int(periods[p] / (ms * delta)) // c + 2 if ms > 0 else 2)
                jmin = min(jmin, jlo[p])
                jmax = max(jmax, jhi[p])
            sA = np.empty((G, nmax))
            sB = np.empty((G, nmax))
            nph = np.empty(G, np.int64)
            nu = np.empty(G, np.int64)
            ncur = np.empty(G, np.int64)
            for j in range(jmin, jmax + 1):
                m = mstep[j]
                dj = m * delta
                Mj = off[j + 1] - off[j]
                o = off[j]
                for q in range(G):
                    p = p0 + q
                    nph[q] = max(1, int(periods[p] / dj)) if jlo[p] <= j <= jhi[p] else 0
                    if screen_hi is not None:
                        if j % 2 != 0 and j != jlo[p] and j != screen_hi[p]:
                            nph[q] = 0
                    nu[q] = (nph[q] + c - 1) // c
                    ncur[q] = 0
                    sA[q, : nu[q]] = 0.0
                    sB[q, : nu[q]] = 0.0
                for c0 in range(0, Mj, chunk):
                    c1 = min(c0 + chunk, Mj)
                    for q in range(G):
                        npq = nph[q]
                        if npq == 0:
                            continue
                        P = periods[p0 + q]
                        qa = sA[q]
                        qb = sB[q]
                        nn = ncur[q]
                        while True:  # transits whose epoch block overlaps the chunk
                            base = int(round(nn * P / dj))
                            if base >= c1:
                                break
                            end = min(base + npq, Mj)
                            s = o + base
                            lo = max(base, c0) - base
                            u = (lo + c - 1) // c  # epochs 0, c, 2c, ... of the period
                            for k in range(u * c, min(end, c1) - base, c):
                                qa[u] += ab[s + k, 0]
                                qb[u] += ab[s + k, 1]
                                u += 1
                            if end > c1:
                                break
                            nn += 1
                        ncur[q] = nn
                for q in range(G):
                    p = p0 + q
                    P = periods[p]
                    bu, bd2 = -1, 0.0  # best coarse epoch of this period and duration
                    for u in range(nu[q]):
                        A = sA[q, u]
                        B = sB[q, u]
                        if B <= 0 or A <= 0 or A < depth_min * B:
                            continue
                        d2 = A * A / B
                        if d2 <= bd2:
                            continue
                        cnt = 0
                        nn = 0
                        while True:
                            kk = u * c + int(round(nn * P / dj))
                            if kk >= Mj:
                                break
                            cnt += ab[o + kk, 1] > 0
                            nn += 1
                        if cnt < ntr_min:
                            continue
                        bu, bd2 = u, d2
                    if bu < 0:
                        continue
                    kb, gA, gB = bu * c, sA[q, bu], sB[q, bu]
                    for kk0 in range(max(0, bu * c - (c - 1)), min(nph[q], bu * c + c)):
                        if kk0 == bu * c:
                            continue
                        d2, A, B = exact_path(ab, o, Mj, P, dj, kk0, ntr_min, depth_min)
                        if d2 > bd2:
                            bd2, kb, gA, gB = d2, kk0, A, B
                    if bd2 > dchi2[p]:
                        dchi2[p] = bd2
                        best_k0[p] = kb * m
                        best_j[p] = j
                        best_A[p] = gA
                        best_B[p] = gB
    return dchi2, best_k0, best_j, best_A, best_B, np.zeros(NP)


def template_table(samples=1024):
    """TLS default transit shape: 0 at the edges, 1 at the bottom."""
    return 1.0 - reference_transit(
        samples=samples,
        per=tls_constants.DEFAULT_PERIOD,
        rp=tls_constants.DEFAULT_RP,
        a=tls_constants.DEFAULT_A,
        inc=tls_constants.DEFAULT_INC,
        ecc=tls_constants.DEFAULT_ECC,
        w=tls_constants.DEFAULT_W,
        u=tls_constants.DEFAULT_U,
        limb_dark=tls_constants.DEFAULT_LIMB_DARK,
    )

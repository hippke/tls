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
def tail_operators(A, K, S, H, ey):
    """Closed form of the single-event sums after a template has ended.

    With zero template input, the normalised innovations are e_i = g_i' x_{i-1}
    and the state evolves as x_i = F_i x_{i-1}, with F_i = (I - K_i H) A_i and
    g_i = -(H A_i)' / sqrt(S_i). From the filtered state x_m at index m, the
    rest of the sums is a_tail = w_m' x_m and b_tail = x_m' W_m x_m, where
      W_m = g_{m+1} g_{m+1}' + F_{m+1}' W_{m+1} F_{m+1}
      w_m = ey_{m+1} g_{m+1} + F_{m+1}' w_{m+1}
    (one backward pass; W is data-independent)."""
    n, d = A.shape[0], A.shape[1]
    W = np.zeros((n, d, d))
    w = np.zeros((n, d))
    F = np.empty((d, d))
    gv = np.empty(d)
    tmp = np.empty((d, d))
    for m in range(n - 2, -1, -1):
        i = m + 1
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
                    acc += W[i, r, l] * F[l, c]
                tmp[r, c] = acc
        for r in range(d):  # W_m = g g' + F' tmp
            for c in range(d):
                acc = gv[r] * gv[c]
                for l in range(d):
                    acc += F[l, r] * tmp[l, c]
                W[m, r, c] = acc
            acc = ey[i] * gv[r]
            for l in range(d):
                acc += F[l, r] * w[i, l]
            w[m, r] = acc
    return W, w


@numba.njit(cache=True)
def _single(t, ey, A, K, H, S, W, w, tcm, D, table, texp, nsub, x, xp):
    """a = (L^-1 s)'(L^-1 y) and b = |L^-1 s|^2 for one transit at tcm with
    duration D: the filter runs over the template support only, the rest is
    the closed-form tail (tail_operators). x, xp: work arrays of length d."""
    n, d = len(t), A.shape[1]
    t_start = tcm - 0.5 * D - 0.5 * texp
    t_end = tcm + 0.5 * D + 0.5 * texp
    i = np.searchsorted(t, t_start)
    x[:] = 0.0
    aa = 0.0
    bb = 0.0
    last = -1
    while i < n and t[i] < t_end:
        hx = 0.0
        for r in range(d):  # predict
            acc = 0.0
            for c in range(d):
                acc += A[i, r, c] * x[c]
            xp[r] = acc
            hx += H[r] * acc
        innov = _template(t[i], tcm, D, table, texp, nsub) - hx
        e = innov / math.sqrt(S[i])
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
    ab = np.zeros((M, 2, J))  # a and b of one k adjacent, each contiguous in j
    for mj in numba.prange(M * J):
        m, j = mj // J, mj % J
        x = np.zeros(d)
        xp = np.empty(d)
        aa, bb = _single(t, ey, A, K, H, S, W, w, tc[m], durs[j], table, texp, nsub, x, xp)
        ab[m, 0, j] = aa
        ab[m, 1, j] = bb
    return ab


def compact_offsets(M, mstep):
    """Start of each duration's block in the compact layout: duration j keeps
    the epochs m = kk * mstep[j], kk < (M - 1) // mstep[j] + 1."""
    sizes = (M - 1) // np.asarray(mstep, np.int64) + 1
    return np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)


@numba.njit(cache=True, parallel=True)
def single_events_compact(t, ey, A, K, S, H, W, w, tc, durs, table, texp, nsub, mstep, off):
    """Single events for the strided fold, compact: duration j only at the
    epochs tc[kk * mstep[j]], stored contiguously from row off[j]; a and b
    of one epoch side by side (row = (a, b)). Memory sum_j M / mstep[j]
    instead of J x M, and the fold reads consecutive rows."""
    J, d = len(durs), A.shape[1]
    ab = np.zeros((off[J], 2))
    for i in numba.prange(off[J]):
        j = np.searchsorted(off, i, side="right") - 1
        m = (i - off[j]) * mstep[j]
        x = np.zeros(d)
        xp = np.empty(d)
        aa, bb = _single(t, ey, A, K, H, S, W, w, tc[m], durs[j], table, texp, nsub, x, xp)
        ab[i, 0] = aa
        ab[i, 1] = bb
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

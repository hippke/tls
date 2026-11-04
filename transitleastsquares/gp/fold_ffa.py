"""Fast fold of single transit events: FFA + polish (PERF_PLAN item 14,
GPU_PLAN.md phase A), optionally with the duration screen (item 13).

multiple_events_blocked folds every period over all epochs of every allowed
duration (~Mj cells per period and duration). For durations with many
transits per period, a fast-folding transform (FFA, Staelin 1969) folds all
m = ceil(Mj / p) drifts of a base period p (epoch steps) at once, in
Mj ceil(log2 m) cells: the periods (p + s / (m - 1)) dj, s = 0 .. m - 1. Every
trial period takes the nearest drift.

The FFA adds transit n at an effective epoch shift that differs from
round(n P) by up to ~2-3 epoch steps (the shift plan below keeps this small).
So the FFA only *locates* the best epoch of each period and duration; that
epoch (+- POLISH_R) is then evaluated on the exact path round(n P), with
n_transits_min counted there. Every reported value is an exact value of the
brute-force fold, so dchi2 <= the brute-force dchi2 for every period, with
equality whenever the FFA finds the right epoch.

Per (duration, base period) block the FFA is used where it needs fewer cells
than the brute force (count of periods in the block > ceil(log2 m)); for
every period the durations up to the last brute-force one go through
multiple_events_blocked (exact), the longer ones through the FFA.

screen (item 13): only the even durations and each period's shortest and
longest are folded; the two neighbours of the best duration are then
evaluated locally (+- REFINE_R epochs around the best transit time).
"""

import math

import numba
import numpy as np

from transitleastsquares.gp.events import multiple_events_blocked

POLISH_R = 2  # exact epochs on each side of the FFA's best epoch
REFINE_R = 3  # screen: epochs on each side of the mapped best transit time
FFA_TASKS = 256  # parallel tasks (blocks go round-robin, costly ones first)
MIN_ROWS = 4  # the FFA needs at least this many rows (transits)

# ------------------------------------------------------------- shift plan
_PLAN = {}  # m -> (lo, hi, sh, st, k), arrays of length m (drifts s)


@numba.njit(cache=True)
def _plan_combine(m, loH, hiH, loT, hiT):
    """Choices (sh, st, k) per drift s of an m-row FFA from the residual
    ranges [lo, hi] of its head (m // 2 rows) and tail sub-transforms, and the
    resulting residual range of the m rows (a conservative bound of
    effective shift - n s / (m - 1))."""
    h = m // 2
    t = m - h
    lo = np.empty(m)
    hi = np.empty(m)
    sh_o = np.empty(m, np.int64)
    st_o = np.empty(m, np.int64)
    k_o = np.empty(m, np.int64)
    for s in range(m):
        slope = s / (m - 1)
        best = 1e300
        hl = hh = 0.0
        for d in range(-2, 3):
            sh = int(round(s * (h - 1) / (m - 1))) + d
            if sh < 0 or sh > h - 1:
                continue
            sl = sh / (h - 1) if h > 1 else 0.0
            lin = (h - 1) * (sl - slope)
            a = loH[sh] + min(0.0, lin)
            b = hiH[sh] + max(0.0, lin)
            e = max(abs(a), abs(b))
            if e < best:
                best, hl, hh, sh_o[s] = e, a, b, sh
        best = 1e300
        tl = th = 0.0
        for d in range(-2, 3):
            st = int(round(s * (t - 1) / (m - 1))) + d
            if st < 0 or st > t - 1:
                continue
            sl = st / (t - 1) if t > 1 else 0.0
            lin = (t - 1) * (sl - slope)
            a = loT[st] + min(0.0, lin) - h * slope
            b = hiT[st] + max(0.0, lin) - h * slope
            k = int(round(-(a + b) / 2))
            e = max(abs(a + k), abs(b + k))
            if e < best:
                best, tl, th, st_o[s], k_o[s] = e, a + k, b + k, st, k
        lo[s] = min(hl, tl)
        hi[s] = max(hh, th)
    return lo, hi, sh_o, st_o, k_o


def _plan(m):
    if m in _PLAN:
        return
    if m == 1:
        z = np.zeros(1)
        _PLAN[1] = (z, z, None, None, None)
        return
    h, t = m // 2, m - m // 2
    _plan(h)
    _plan(t)
    _PLAN[m] = _plan_combine(m, _PLAN[h][0], _PLAN[h][1], _PLAN[t][0], _PLAN[t][1])


def flat_plan(ms):
    """Plans of the given row counts (and their sub-transforms) as flat
    arrays: (offset by size, sh, st, k)."""
    for m in ms:
        _plan(int(m))
    sizes = sorted(k for k, v in _PLAN.items() if v[2] is not None)
    poff = np.zeros(max(sizes + [1]) + 1, np.int64)
    n = 0
    for k in sizes:
        poff[k] = n
        n += k
    psh, pst, pk = (np.zeros(max(n, 1), np.int64) for _ in range(3))
    for k in sizes:
        _, _, sh, st, kk = _PLAN[k]
        psh[poff[k]:poff[k] + k] = sh
        pst[poff[k]:poff[k] + k] = st
        pk[poff[k]:poff[k] + k] = kk
    return poff, psh, pst, pk


def effective_shifts(m):
    """eff[s, n]: the shift at which the FFA (with the plan) adds row n for
    drift s (dense, for tests and diagnostics)."""
    _plan(m)
    if m == 1:
        return np.zeros((1, 1))
    h = m // 2
    EH, ET = effective_shifts(h), effective_shifts(m - h)
    _, _, sh, st, k = _PLAN[m]
    return np.concatenate([EH[sh], ET[st] + k[:, None]], axis=1)


# ---------------------------------------------------------------- kernels
@numba.njit(cache=True, nogil=True)
def _ffa(x, out, tmp, poff, psh, pst, pk):
    """FFA of the rows x (m, p, 3) into out (m, p, 3): out[s] is the sum of the
    rows, row n shifted forward (circularly) by its effective shift for s."""
    m, p = x.shape[0], x.shape[1]
    if m == 1:
        out[0] = x[0]
        return
    h = m // 2
    _ffa(x[:h], tmp[:h], out[:h], poff, psh, pst, pk)  # out / tmp swap as scratch
    _ffa(x[h:], tmp[h:], out[h:], poff, psh, pst, pk)
    o0 = poff[m]
    for s in range(m):
        H = tmp[psh[o0 + s]]
        T = tmp[h + pst[o0 + s]]
        k = pk[o0 + s] % p
        o = out[s]
        for i in range(p - k):
            o[i, 0] = H[i, 0] + T[i + k, 0]
            o[i, 1] = H[i, 1] + T[i + k, 1]
            o[i, 2] = H[i, 2] + T[i + k, 2]
        for i in range(p - k, p):
            o[i, 0] = H[i, 0] + T[i + k - p, 0]
            o[i, 1] = H[i, 1] + T[i + k - p, 1]
            o[i, 2] = H[i, 2] + T[i + k - p, 2]


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


@numba.njit(cache=True, nogil=True)
def _local(ab, o, Mj, P, dj, center, r, ntr_min, depth_min):
    """Best exact path within +- r epochs of center (wrapped into the period)."""
    nph = max(1, int(P / dj))
    g, gA, gB, gk = 0.0, 0.0, 0.0, -1
    for d in range(-r, r + 1):
        kk0 = (center + d) % nph
        d2, A, B = exact_path(ab, o, Mj, P, dj, kk0, ntr_min, depth_min)
        if d2 > g:
            g, gA, gB, gk = d2, A, B, kk0
    return g, gA, gB, gk


@numba.njit(cache=True, parallel=True)
def _ffa_tasks(ab, off, mstep, delta, periods, t_j, t_p, t_start, t_count, targets, order,
               ntr_min, depth_min, polish_r, poff, psh, pst, pk, g, gA, gB, gk):
    """All FFA blocks (duration t_j, base period t_p, periods
    targets[t_start:t_start + t_count]), polished: writes the exact gain, sums
    and epoch (duration grid) per target."""
    NT = len(t_j)
    for task in numba.prange(FFA_TASKS):
        for u in range(task, NT, FFA_TASKS):
            b = order[u]
            j = t_j[b]
            p = t_p[b]
            o = off[j]
            Mj = off[j + 1] - o
            dj = mstep[j] * delta
            m = (Mj + p - 1) // p
            x = np.zeros((m, p, 3))
            for q in range(Mj):
                n = q // p
                i = q - n * p
                x[n, i, 0] = ab[o + q, 0]
                x[n, i, 1] = ab[o + q, 1]
                x[n, i, 2] = 1.0 if ab[o + q, 1] > 0 else 0.0
            out = np.empty((m, p, 3))
            tmp = np.empty((m, p, 3))
            _ffa(x, out, tmp, poff, psh, pst, pk)
            for c in range(t_count[b]):
                w = t_start[b] + c
                Pb = periods[targets[w]] / dj
                s = int(round((Pb - p) * (m - 1)))
                s = min(max(s, 0), m - 1)
                best, bi = 0.0, -1
                for i in range(p):
                    A = out[s, i, 0]
                    B = out[s, i, 1]
                    if B <= 0 or A <= 0 or A < depth_min * B or out[s, i, 2] < ntr_min - 0.5:
                        continue
                    if A * A / B > best:
                        best, bi = A * A / B, i
                if bi < 0:
                    continue
                g[w], gA[w], gB[w], gk[w] = _local(ab, o, Mj, periods[targets[w]], dj, bi, polish_r,
                                                   ntr_min, depth_min)


@numba.njit(cache=True, parallel=True)
def _refine_neighbours(ab, off, mstep, delta, periods, jlo, jhi, dchi2, best_k0, best_j, best_A, best_B,
                       ntr_min, depth_min, refine_r):
    """Duration screen: the neighbours j* +- 1 of the best duration (if they
    were not folded), locally around the best transit time."""
    for q in numba.prange(len(periods)):
        j0 = best_j[q]
        if dchi2[q] <= 0:
            continue
        t_best = best_k0[q]  # epoch on the full grid
        for jn in (j0 - 1, j0 + 1):
            if jn < jlo[q] or jn > jhi[q] or jn % 2 == 0 or jn == jlo[q] or jn == jhi[q]:
                continue  # outside, or folded already
            o = off[jn]
            Mj = off[jn + 1] - o
            center = int(round(t_best / mstep[jn]))
            d2, A, B, k = _local(ab, o, Mj, periods[q], mstep[jn] * delta, center, refine_r, ntr_min, depth_min)
            if d2 > dchi2[q]:
                dchi2[q], best_k0[q], best_j[q], best_A[q], best_B[q] = d2, k * mstep[jn], jn, A, B


@numba.njit(cache=True)
def _merge(targets, g, gA, gB, gk, t_jt, mstep, dchi2, best_k0, best_j, best_A, best_B):
    for w in range(len(targets)):
        q = targets[w]
        if g[w] > dchi2[q]:
            j = t_jt[w]
            dchi2[q], best_k0[q], best_j[q], best_A[q], best_B[q] = g[w], gk[w] * mstep[j], j, gA[w], gB[w]


def plan_blocks(Mj, dj, periods, jlo, jhi, screen):
    """Which (period, duration) pairs go through the FFA. Returns the last
    brute-force duration per period (jb_hi; < jlo: none) and the FFA blocks
    (lists of duration, base period, period indices)."""
    NP, J = len(periods), len(Mj)
    jj = np.arange(J)[None, :]
    allowed = (jj >= jlo[:, None]) & (jj <= jhi[:, None])
    if screen:
        allowed &= (jj % 2 == 0) | (jj == jlo[:, None]) | (jj == jhi[:, None])
    use = np.zeros((NP, J), bool)
    for j in range(J):
        q = np.nonzero(allowed[:, j])[0]
        if len(q) == 0:
            continue
        base = np.floor(periods[q] / dj[j]).astype(np.int64)
        ub, inv, cnt = np.unique(base, return_inverse=True, return_counts=True)
        m = -(-Mj[j] // ub)
        levels = np.ceil(np.log2(np.maximum(m, 2)))
        ok = (m >= MIN_ROWS) & (cnt > levels)
        use[q, j] = ok[inv]
    brute = allowed & ~use
    last = np.where(brute.any(axis=1), J - 1 - np.argmax(brute[:, ::-1], axis=1), -1)
    jb_hi = np.where(last >= 0, last, jlo - 1).astype(np.int64)
    ffa = allowed & (jj > jb_hi[:, None])
    blocks = []
    for j in range(J):
        q = np.nonzero(ffa[:, j])[0]
        if len(q) == 0:
            continue
        base = np.floor(periods[q] / dj[j]).astype(np.int64)
        for p in np.unique(base):
            blocks.append((j, int(p), q[base == p]))
    return jb_hi, blocks


def fold_ffa(ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, gstart, chunk, depth_min=0.0,
             polish_r=POLISH_R, screen=False, refine_r=REFINE_R, stats=None):
    """Drop-in for multiple_events_blocked (same arguments and outputs:
    dchi2, best epoch on the full grid, best duration, sums A and B, zeros);
    approximate, every value an exact brute-force fold value (module doc).
    stats: optional dict, filled with the cell counts (brute, FFA, polish)."""
    periods = np.ascontiguousarray(periods, float)
    jlo = np.asarray(jlo, np.int64)
    jhi = np.asarray(jhi, np.int64)
    Mj = np.diff(off)
    dj = mstep * delta
    jb_hi, blocks = plan_blocks(Mj, dj, periods, jlo, jhi, screen)
    dchi2, best_k0, best_j, best_A, best_B, ntrials = multiple_events_blocked(
        ab, off, M, periods, delta, mstep, jlo, jb_hi, ntr_min, gstart, chunk, depth_min,
        jhi if screen else None)
    if blocks:
        t_j = np.array([b[0] for b in blocks], np.int64)
        t_p = np.array([b[1] for b in blocks], np.int64)
        t_count = np.array([len(b[2]) for b in blocks], np.int64)
        t_start = np.concatenate([[0], np.cumsum(t_count)[:-1]]).astype(np.int64)
        targets = np.concatenate([b[2] for b in blocks]).astype(np.int64)
        t_jt = np.repeat(t_j, t_count)
        m = -(-Mj[t_j] // t_p)
        plan = flat_plan(np.unique(m))
        order = np.argsort(-(Mj[t_j] * np.ceil(np.log2(m))), kind="stable").astype(np.int64)
        W = len(targets)
        g, gA, gB = np.zeros(W), np.zeros(W), np.zeros(W)
        gk = np.zeros(W, np.int64)
        _ffa_tasks(ab, off, mstep, delta, periods, t_j, t_p, t_start, t_count, targets, order,
                   ntr_min, depth_min, polish_r, *plan, g, gA, gB, gk)
        _merge(targets, g, gA, gB, gk, t_jt, mstep, dchi2, best_k0, best_j, best_A, best_B)
    if screen:
        _refine_neighbours(ab, off, mstep, delta, periods, jlo, jhi, dchi2, best_k0, best_j, best_A, best_B,
                           ntr_min, depth_min, refine_r)
    if stats is not None:
        ntr = Mj[None, :] * dj[None, :] / periods[:, None]
        jj = np.arange(len(Mj))[None, :]
        brute_cells = float(np.sum(np.where((jj >= jlo[:, None]) & (jj <= jb_hi[:, None]), Mj[None, :], 0)))
        ffa_cells = float(sum(Mj[j] * math.ceil(math.log2(max(-(-Mj[j] // p), 2))) for j, p, _ in blocks))
        polish_cells = float(sum((2 * polish_r + 1) * ntr[q, j].sum() for j, _, q in blocks))
        full = float(np.sum(np.where((jj >= jlo[:, None]) & (jj <= jhi[:, None]), Mj[None, :], 0)))
        stats.update(cells_full_brute=full, cells_brute=brute_cells, cells_ffa=ffa_cells,
                     cells_polish=polish_cells, blocks=len(blocks))
    return dchi2, best_k0, best_j, best_A, best_B, ntrials

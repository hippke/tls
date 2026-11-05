"""CUDA kernels (numba-cuda). FP32 arithmetic; see the package docstring."""

import math

import numpy as np
from numba import cuda, float32, float64, int32

F0 = np.float32(0.0)
F1 = np.float32(1.0)
HALF = np.float32(0.5)
MAX_DIM = 8  # state dimension of the GP models (rotation: 7)
TOFF = 4096  # transit offsets kept in shared memory per fold block
FOLD_THREADS = 128


@cuda.jit(device=True, inline=True)
def _template(dt, invD, table, texp, nsub):
    """gp.events._template_inv in FP32 with dt = t - tc (FP32), invD = 1 / D."""
    L = table.shape[0]
    acc = F0
    for k in range(nsub):
        off = (float32(k) + HALF) / float32(nsub) - HALF
        u = (dt + off * texp) * invD
        if u > -HALF and u < HALF:
            x = (u + HALF) * float32(L - 1)
            i = min(int32(x), L - 2)
            fr = x - float32(i)
            acc += table[i] * (F1 - fr) + table[i + 1] * fr
    return -acc / float32(nsub)


@cuda.jit(device=True, inline=True)
def _first(t, x):
    """First index with t[i] >= x (numpy.searchsorted, side="left")."""
    lo = 0
    hi = t.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if t[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cuda.jit
def single_white(t, rw, w, t0, delta, durs, nsubs, table, texp, mstep, off, row_j, ab):
    """backends.events.single_events_white: a = sum s r w, b = sum s^2 w per
    row of the compact layout (one thread per row). Also the GP numerator
    with rw = z (then b is not used)."""
    i = cuda.grid(1)
    if i >= ab.shape[0]:
        return
    j = row_j[i]
    tcm = t0 + float64(i - off[j]) * float64(mstep[j]) * delta
    D = durs[j]
    invD = F1 / D
    half = 0.5 * float64(D) + 0.5 * float64(texp)
    n = t.shape[0]
    s = _first(t, tcm - half)
    aa = F0
    bb = F0
    while s < n and t[s] < tcm + half:
        v = _template(float32(t[s] - tcm), invD, table, texp, nsubs[j])
        aa += v * rw[s]
        bb += v * v * w[s]
        s += 1
    ab[i, 0] = aa
    ab[i, 1] = bb


@cuda.jit
def single_gp(t, z, eyd, At, idx, starts, sizes, hidx, K, isS, H, W, t0, delta, durs, nsubs, table, texp, mstep, off,
              row_j, ab):
    """gp.events.single_events_split in FP32 (one thread per row): a = sum s z
    minus the diffuse-start terms, b = |L^-1 s|^2 by the filter with the
    precomputed gains, the diagonal blocks of the transitions (At[idx[i]]:
    blocks of 1 or 2) and the tail x' W x."""
    i = cuda.grid(1)
    if i >= ab.shape[0]:
        return
    d = K.shape[1]
    nd = eyd.shape[0]
    nb = starts.shape[0]
    nh = hidx.shape[0]
    j = row_j[i]
    tcm = t0 + float64(i - off[j]) * float64(mstep[j]) * delta
    D = durs[j]
    invD = F1 / D
    half = 0.5 * float64(D) + 0.5 * float64(texp)
    n = t.shape[0]
    x = cuda.local.array(MAX_DIM, float32)
    xp = cuda.local.array(MAX_DIM, float32)
    for r in range(d):
        x[r] = F0
    s = _first(t, tcm - half)
    aa = F0
    bb = F0
    last = -1
    while s < n and t[s] < tcm + half:
        u = idx[s]
        for k in range(nb):  # predict, block-diagonal
            s0 = starts[k]
            if sizes[k] == 2:
                x0 = x[s0]
                x1 = x[s0 + 1]
                xp[s0] = At[u, k, 0, 0] * x0 + At[u, k, 0, 1] * x1
                xp[s0 + 1] = At[u, k, 1, 0] * x0 + At[u, k, 1, 1] * x1
            else:
                xp[s0] = At[u, k, 0, 0] * x[s0]
        hx = F0
        for q in range(nh):
            hx += H[hidx[q]] * xp[hidx[q]]
        v = _template(float32(t[s] - tcm), invD, table, texp, nsubs[j])
        aa += v * z[s]
        innov = v - hx
        e = innov * isS[s]
        if s < nd:
            aa -= e * eyd[s]
        bb += e * e
        for r in range(d):
            x[r] = xp[r] + K[s, r] * innov
        last = s
        s += 1
    if last >= 0:
        for r in range(d):
            acc = F0
            for c in range(d):
                acc += W[last, r, c] * x[c]
            bb += x[r] * acc
    ab[i, 0] = aa
    ab[i, 1] = bb


@cuda.jit(device=True, inline=True)
def _rint(v):
    """round(v) for an FP64 v >= 0, half to even (as round() on the CPU)."""
    f = math.floor(v)
    r = v - f
    if r > 0.5 or (r == 0.5 and int32(f) % 2 == 1):
        f += 1.0
    return int32(f)


@cuda.jit(device=True, inline=True)
def _offset(nn, P, dj):
    """round(nn P / dj) in FP64, as int(round(nn * P / dj)) on the CPU."""
    return _rint(float64(nn) * P / dj)


@cuda.jit(device=True, inline=True)
def _path(ab, o, Mj, P, dj, kk0, ntr_min, depth_min):
    """Exact path of epoch kk0 (as fold_ffa.exact_path): (gain or 0, A, B)."""
    A = F0
    B = F0
    cnt = 0
    nn = 0
    while True:
        kk = kk0 + _offset(nn, P, dj)
        if kk >= Mj:
            break
        b = ab[o + kk, 1]
        A += ab[o + kk, 0]
        B += b
        if b > F0:
            cnt += 1
        nn += 1
    if cnt < ntr_min or B <= F0 or A <= F0 or A < depth_min * B:
        return F0, A, B
    return A * A / B, A, B


@cuda.jit(device=True, inline=True)
def _better(g, k, bg, bk):
    """(g, k) beats (bg, bk): larger gain, ties to the smaller epoch (the CPU
    kernels keep the first maximum in ascending epoch order)."""
    return g > bg or (g == bg and g > F0 and k < bk)


@cuda.jit(device=True)
def _reduce_write(bg, bA, bB, bk, slot, out_g, out_A, out_B, out_k):
    """Block-wide best (gain, epoch) -> out[slot] (thread 0)."""
    sg = cuda.shared.array(FOLD_THREADS, float32)
    sA = cuda.shared.array(FOLD_THREADS, float32)
    sB = cuda.shared.array(FOLD_THREADS, float32)
    sk = cuda.shared.array(FOLD_THREADS, int32)
    tid = cuda.threadIdx.x
    sg[tid] = bg
    sA[tid] = bA
    sB[tid] = bB
    sk[tid] = bk
    cuda.syncthreads()
    step = cuda.blockDim.x // 2
    while step > 0:
        if tid < step:
            if _better(sg[tid + step], sk[tid + step], sg[tid], sk[tid]):
                sg[tid] = sg[tid + step]
                sA[tid] = sA[tid + step]
                sB[tid] = sB[tid + step]
                sk[tid] = sk[tid + step]
        cuda.syncthreads()
        step //= 2
    if tid == 0:
        out_g[slot] = sg[0]
        out_A[slot] = sA[0]
        out_B[slot] = sB[0]
        out_k[slot] = sk[0]


@cuda.jit
def ffa_load(ab, off, blk_j, blk_p, blk_base, blk_rows, buf0, buf1):
    """Rows of every FFA block (m x p x 3: a, b, has data) into both buffers;
    one CUDA block per FFA block."""
    bi = cuda.blockIdx.x
    j = blk_j[bi]
    p = blk_p[bi]
    o = off[j]
    Mj = off[j + 1] - o
    base = blk_base[bi]
    size = blk_rows[bi] * p
    for e in range(cuda.threadIdx.x, size, cuda.blockDim.x):
        a = F0
        b = F0
        if e < Mj:
            a = ab[o + e, 0]
            b = ab[o + e, 1]
        c = F1 if b > F0 else F0
        buf0[base + 3 * e] = a
        buf0[base + 3 * e + 1] = b
        buf0[base + 3 * e + 2] = c
        buf1[base + 3 * e] = a
        buf1[base + 3 * e + 1] = b
        buf1[base + 3 * e + 2] = c


@cuda.jit
def ffa_level(node_base, node_p, node_start, node_size, poff, psh, pst, pk, src, dst):
    """One depth of the FFA trees: every node merges its head (size // 2 rows)
    and tail sub-transforms from src into dst (as fold_ffa._ffa); one CUDA
    block per node."""
    ni = cuda.blockIdx.x
    base = node_base[ni]
    p = node_p[ni]
    r0 = node_start[ni]
    size = node_size[ni]
    h = size // 2
    po = poff[size]
    for e in range(cuda.threadIdx.x, size * p, cuda.blockDim.x):
        s = e // p
        i = e - s * p
        k = pk[po + s] % p
        ii = i + k
        if ii >= p:
            ii -= p
        hrow = r0 + psh[po + s]
        trow = r0 + h + pst[po + s]
        dd = base + 3 * ((r0 + s) * p + i)
        hh = base + 3 * (hrow * p + i)
        tt = base + 3 * (trow * p + ii)
        dst[dd] = src[hh] + src[tt]
        dst[dd + 1] = src[hh + 1] + src[tt + 1]
        dst[dd + 2] = src[hh + 2] + src[tt + 2]


@cuda.jit
def ffa_eval_polish(buf, ab, off, mstep, delta, periods, tgt_q, tgt_blk, blk_j, blk_p, blk_base, blk_rows,
                    ntr_min, depth_min, polish_r, out_g, out_A, out_B, out_k):
    """Per target period (one CUDA block): the best FFA phase of its drift,
    then the exact paths at that phase +- polish_r (as fold_ffa._ffa_tasks)."""
    w = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    q = tgt_q[w]
    bi = tgt_blk[w]
    j = blk_j[bi]
    p = blk_p[bi]
    m = blk_rows[bi]
    base = blk_base[bi]
    dj = float64(mstep[j]) * delta
    P = periods[q]
    s = _rint(max((P / dj - float64(p)) * float64(m - 1), 0.0))
    s = min(max(s, 0), m - 1)
    bg = F0
    bk = int32(-1)
    for i in range(tid, p, cuda.blockDim.x):
        e = base + 3 * (s * p + i)
        A = buf[e]
        B = buf[e + 1]
        if B <= F0 or A <= F0 or A < depth_min * B or buf[e + 2] < float32(ntr_min) - HALF:
            continue
        g = A * A / B
        if _better(g, i, bg, bk):
            bg, bk = g, i
    sg = cuda.shared.array(FOLD_THREADS, float32)
    sk = cuda.shared.array(FOLD_THREADS, int32)
    sg[tid] = bg
    sk[tid] = bk
    cuda.syncthreads()
    step = cuda.blockDim.x // 2
    while step > 0:
        if tid < step and _better(sg[tid + step], sk[tid + step], sg[tid], sk[tid]):
            sg[tid] = sg[tid + step]
            sk[tid] = sk[tid + step]
        cuda.syncthreads()
        step //= 2
    center = sk[0]
    cuda.syncthreads()
    # polish: 2 polish_r + 1 exact paths, one per thread (ties: the CPU scans d = -r .. r)
    o = off[j]
    Mj = off[j + 1] - o
    nph = max(1, int32(P / dj))
    g = F0
    A = F0
    B = F0
    k = int32(-1)
    dsel = int32(1 << 30)
    if center >= 0 and tid < 2 * polish_r + 1:
        kk0 = (center + tid - polish_r) % nph
        g, A, B = _path(ab, o, Mj, P, dj, kk0, ntr_min, depth_min)
        k = kk0
        dsel = tid
    _reduce_polish(g, A, B, k, dsel, w, out_g, out_A, out_B, out_k)


@cuda.jit(device=True)
def _reduce_polish(g, A, B, k, d, slot, out_g, out_A, out_B, out_k):
    """Best over the polish candidates; ties to the smaller offset d."""
    sg = cuda.shared.array(FOLD_THREADS, float32)
    sA = cuda.shared.array(FOLD_THREADS, float32)
    sB = cuda.shared.array(FOLD_THREADS, float32)
    sk = cuda.shared.array(FOLD_THREADS, int32)
    sd = cuda.shared.array(FOLD_THREADS, int32)
    tid = cuda.threadIdx.x
    sg[tid] = g
    sA[tid] = A
    sB[tid] = B
    sk[tid] = k
    sd[tid] = d
    cuda.syncthreads()
    step = cuda.blockDim.x // 2
    while step > 0:
        if tid < step:
            g2, d2 = sg[tid + step], sd[tid + step]
            if g2 > sg[tid] or (g2 == sg[tid] and g2 > F0 and d2 < sd[tid]):
                sg[tid] = g2
                sA[tid] = sA[tid + step]
                sB[tid] = sB[tid + step]
                sk[tid] = sk[tid + step]
                sd[tid] = d2
        cuda.syncthreads()
        step //= 2
    if tid == 0:
        out_g[slot] = sg[0]
        out_A[slot] = sA[0]
        out_B[slot] = sB[0]
        out_k[slot] = sk[0]


@cuda.jit(device=True)
def _block_best(g, k, sg, sk):
    """Block-wide best (gain, epoch; ties to the smaller epoch) of the
    threads' (g, k); every thread returns it."""
    tid = cuda.threadIdx.x
    sg[tid] = g
    sk[tid] = k
    cuda.syncthreads()
    step = cuda.blockDim.x // 2
    while step > 0:
        if tid < step and _better(sg[tid + step], sk[tid + step], sg[tid], sk[tid]):
            sg[tid] = sg[tid + step]
            sk[tid] = sk[tid + step]
        cuda.syncthreads()
        step //= 2
    bg = sg[0]
    bk = sk[0]
    cuda.syncthreads()
    return bg, bk


@cuda.jit
def fold_brute(abc, off, mstep, delta, periods, pair_q, pair_j, ntr_min, depth_min, coarse, out_g, out_A, out_B,
               out_k):
    """Brute-force fold of (period, duration) pairs, one CUDA block per pair:
    the best epoch over all epochs (as multiple_events_blocked). The single
    events as complex64 (a + i b: one 8-byte load per cell), two epochs per
    thread (two independent sums in flight). coarse > 1: only every
    coarse-th epoch, then its coarse - 1 neighbours on each side on the
    exact path (as multiple_events_blocked_coarse)."""
    pr = cuda.blockIdx.x
    if pr >= pair_q.shape[0]:
        return
    tid = cuda.threadIdx.x
    nt = cuda.blockDim.x
    q = pair_q[pr]
    j = pair_j[pr]
    P = periods[q]
    dj = float64(mstep[j]) * delta
    o = off[j]
    Mj = off[j + 1] - o
    nph = max(1, int32(P / dj))
    nu = (nph + coarse - 1) // coarse
    ntr = int32(float64(Mj) * dj / P) + 2
    toff = cuda.shared.array(TOFF, int32)
    nsh = min(ntr, TOFF)
    for nn in range(tid, nsh, nt):
        toff[nn] = _offset(nn, P, dj)
    cuda.syncthreads()
    bg = F0
    bA = F0
    bB = F0
    bk = int32(-1)
    for u1 in range(tid, nu, 2 * nt):
        k1 = u1 * coarse
        k2 = (u1 + nt) * coarse
        two = u1 + nt < nu
        A1 = F0
        B1 = F0
        c1 = 0
        A2 = F0
        B2 = F0
        c2 = 0
        nn = 0
        while True:
            base = toff[nn] if nn < nsh else _offset(nn, P, dj)
            kk = k1 + base
            if kk >= Mj:
                break
            v = abc[o + kk]
            A1 += v.real
            B1 += v.imag
            if v.imag > F0:
                c1 += 1
            kk = k2 + base
            if two and kk < Mj:
                v = abc[o + kk]
                A2 += v.real
                B2 += v.imag
                if v.imag > F0:
                    c2 += 1
            nn += 1
        if two:  # epoch k2 may have transits beyond the last of k1 (k2 > k1)
            while True:
                base = toff[nn] if nn < nsh else _offset(nn, P, dj)
                kk = k2 + base
                if kk >= Mj:
                    break
                v = abc[o + kk]
                A2 += v.real
                B2 += v.imag
                if v.imag > F0:
                    c2 += 1
                nn += 1
        if not (c1 < ntr_min or B1 <= F0 or A1 <= F0 or A1 < depth_min * B1):
            g = A1 * A1 / B1
            if _better(g, k1, bg, bk):
                bg, bA, bB, bk = g, A1, B1, k1
        if two and not (c2 < ntr_min or B2 <= F0 or A2 <= F0 or A2 < depth_min * B2):
            g = A2 * A2 / B2
            if _better(g, k2, bg, bk):
                bg, bA, bB, bk = g, A2, B2, k2
    if coarse > 1:  # refine around the best coarse epoch
        sg = cuda.shared.array(FOLD_THREADS, float32)
        sk = cuda.shared.array(FOLD_THREADS, int32)
        cg, ck = _block_best(bg, bk, sg, sk)
        if ck >= 0 and tid < 2 * (coarse - 1):
            d = tid - (coarse - 1)
            if d >= 0:
                d += 1  # offsets -(c-1) .. -1, 1 .. c-1
            kk0 = ck + d
            if 0 <= kk0 < nph:
                g, A, B = _path_c(abc, o, Mj, P, dj, kk0, toff, nsh, ntr_min, depth_min)
                if _better(g, kk0, bg, bk):
                    bg, bA, bB, bk = g, A, B, kk0
    _reduce_write(bg, bA, bB, bk, pr, out_g, out_A, out_B, out_k)


@cuda.jit(device=True, inline=True)
def _path_c(abc, o, Mj, P, dj, kk0, toff, nsh, ntr_min, depth_min):
    """Exact path of epoch kk0 on complex64 single events (offsets from the
    shared table where available)."""
    A = F0
    B = F0
    cnt = 0
    nn = 0
    while True:
        kk = kk0 + (toff[nn] if nn < nsh else _offset(nn, P, dj))
        if kk >= Mj:
            break
        v = abc[o + kk]
        A += v.real
        B += v.imag
        if v.imag > F0:
            cnt += 1
        nn += 1
    if cnt < ntr_min or B <= F0 or A <= F0 or A < depth_min * B:
        return F0, A, B
    return A * A / B, A, B


@cuda.jit
def refine_local(abc, off, mstep, delta, periods, task_q, task_j, center, radius, ntr_min, depth_min,
                 out_g, out_A, out_B, out_k):
    """Duration screen: exact paths of duration task_j around epoch center
    (+- radius, wrapped into the period), one CUDA block per task."""
    w = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    q = task_q[w]
    j = task_j[w]
    P = periods[q]
    dj = float64(mstep[j]) * delta
    o = off[j]
    Mj = off[j + 1] - o
    nph = max(1, int32(P / dj))
    toff = cuda.shared.array(TOFF, int32)
    ntr = int32(float64(Mj) * dj / P) + 2
    nsh = min(ntr, TOFF)
    for nn in range(tid, nsh, cuda.blockDim.x):
        toff[nn] = _offset(nn, P, dj)
    cuda.syncthreads()
    g = F0
    A = F0
    B = F0
    k = int32(-1)
    if tid < 2 * radius + 1:
        kk0 = (center[w] + tid - radius) % nph
        g, A, B = _path_c(abc, o, Mj, P, dj, kk0, toff, nsh, ntr_min, depth_min)
        k = kk0
    _reduce_write(g, A, B, k, w, out_g, out_A, out_B, out_k)


@cuda.jit(device=True, inline=True)
def _tail_step(At, idx, starts, sizes, K, isS, H, ey, i, d, Wn, wn, Wm, wm, F, gv, tmp):
    """gp.events._tail_step in FP32 with the transitions as blocks: from
    W_{m+1} = Wn, w_{m+1} = wn to Wm, wm at m = i - 1; F_i in F."""
    nb = starts.shape[0]
    s = isS[i]
    u = idx[i]
    for k in range(d):
        gv[k] = F0
    for b in range(nb):  # gv = -(H A_i) / sqrt(S_i)
        s0 = starts[b]
        for cc in range(sizes[b]):
            acc = F0
            for ll in range(sizes[b]):
                acc += H[s0 + ll] * At[u, b, ll, cc]
            gv[s0 + cc] = -acc * s
    for r in range(d):  # F = A - K (H A) = A + K gv / s
        for c in range(d):
            F[r, c] = K[i, r] * gv[c] / s
    for b in range(nb):
        s0 = starts[b]
        for rr in range(sizes[b]):
            for cc in range(sizes[b]):
                F[s0 + rr, s0 + cc] += At[u, b, rr, cc]
    for r in range(d):  # tmp = Wn F
        for c in range(d):
            acc = F0
            for l in range(d):
                acc += Wn[r, l] * F[l, c]
            tmp[r, c] = acc
    for r in range(d):  # Wm = g g' + F' tmp, wm = ey g + F' wn
        for c in range(d):
            acc = gv[r] * gv[c]
            for l in range(d):
                acc += F[l, r] * tmp[l, c]
            Wm[r, c] = acc
        acc = ey[i] * gv[r]
        for l in range(d):
            acc += F[l, r] * wn[l]
        wm[r] = acc


@cuda.jit
def tail_maps(At, idx, starts, sizes, K, isS, H, ey, edges, Phi, C, cc):
    """Phase 1 of gp.events.tail_operators (one thread per segment k >= 1):
    the segment's affine map W_lo = Phi' W_b Phi + C, w_lo = Phi' w_b + cc,
    composed backwards from its boundary edges[k + 1]."""
    k = cuda.grid(1) + 1
    if k >= Phi.shape[0]:
        return
    d = K.shape[1]
    Wn = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    Wm = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    F = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    tmp = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    Ph = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    Pn = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    wn = cuda.local.array(MAX_DIM, float32)
    wm = cuda.local.array(MAX_DIM, float32)
    gv = cuda.local.array(MAX_DIM, float32)
    for r in range(d):
        wn[r] = F0
        for c in range(d):
            Wn[r, c] = F0
            Ph[r, c] = F1 if r == c else F0
    for m in range(edges[k + 1] - 1, edges[k] - 1, -1):
        _tail_step(At, idx, starts, sizes, K, isS, H, ey, m + 1, d, Wn, wn, Wm, wm, F, gv, tmp)
        for r in range(d):
            wn[r] = wm[r]
            for c in range(d):
                Wn[r, c] = Wm[r, c]
        for r in range(d):  # Ph = Ph F
            for c in range(d):
                acc = F0
                for l in range(d):
                    acc += Ph[r, l] * F[l, c]
                Pn[r, c] = acc
        for r in range(d):
            for c in range(d):
                Ph[r, c] = Pn[r, c]
    for r in range(d):
        cc[k, r] = wn[r]
        for c in range(d):
            C[k, r, c] = Wn[r, c]
            Phi[k, r, c] = Ph[r, c]


@cuda.jit
def tail_fill(At, idx, starts, sizes, K, isS, H, ey, edges, Wb, wb, W, w):
    """Phase 3 of gp.events.tail_operators (one thread per segment): W, w of
    every point of the segment from the known boundary Wb[k], wb[k]."""
    k = cuda.grid(1)
    if k >= Wb.shape[0]:
        return
    d = K.shape[1]
    Wn = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    Wm = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    F = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    tmp = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    wn = cuda.local.array(MAX_DIM, float32)
    wm = cuda.local.array(MAX_DIM, float32)
    gv = cuda.local.array(MAX_DIM, float32)
    for r in range(d):
        wn[r] = wb[k, r]
        for c in range(d):
            Wn[r, c] = Wb[k, r, c]
    for m in range(edges[k + 1] - 1, edges[k] - 1, -1):
        _tail_step(At, idx, starts, sizes, K, isS, H, ey, m + 1, d, Wn, wn, Wm, wm, F, gv, tmp)
        for r in range(d):
            wn[r] = wm[r]
            w[m, r] = wm[r]
            for c in range(d):
                Wn[r, c] = Wm[r, c]
                W[m, r, c] = Wm[r, c]


@cuda.jit
def riccati_segments(Gt, Qt, idx, R, starts, sizes, Pk, PkH, SkH, keys, edges, K, isS, overlap):
    """The covariance recursion, one thread per segment: from Pk[q] (filtered at
    the point before keys[q], which is before the segment: the filter forgets
    the key's error) the gains and 1 / sqrt(S) of the points edges[q]..edges[q + 1]
    - 1. Only D = P - Pk is in FP32: the terms of the key that cancel (PkH = Pk H,
    SkH = H Pk H') come from FP64, and the prediction uses G = A - I, so that its
    rounding scales with |G| instead of |P|. overlap[q]: 1 / sqrt(S) at the last
    point before the segment, which the previous segment also computed."""
    q = cuda.grid(1)
    if q >= Pk.shape[0]:
        return
    d = K.shape[1]
    nb = starts.shape[0]
    P0 = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    D = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    M = cuda.local.array((MAX_DIM, MAX_DIM), float32)
    dH = cuda.local.array(MAX_DIM, float32)
    for r in range(d):
        for c in range(d):
            P0[r, c] = Pk[q, r, c]
            D[r, c] = F0
    lo = edges[q]
    for i in range(keys[q], edges[q + 1]):
        u = idx[i]
        for b in range(nb):  # M = G (Pk + D)
            s0 = starts[b]
            if sizes[b] == 2:
                g00, g01, g10, g11 = Gt[u, b, 0, 0], Gt[u, b, 0, 1], Gt[u, b, 1, 0], Gt[u, b, 1, 1]
                for c in range(d):
                    p0 = P0[s0, c] + D[s0, c]
                    p1 = P0[s0 + 1, c] + D[s0 + 1, c]
                    M[s0, c] = g00 * p0 + g01 * p1
                    M[s0 + 1, c] = g10 * p0 + g11 * p1
            else:
                g00 = Gt[u, b, 0, 0]
                for c in range(d):
                    M[s0, c] = g00 * (P0[s0, c] + D[s0, c])
        for r in range(d):  # D += M + M'
            for c in range(r, d):
                v = D[r, c] + M[r, c] + M[c, r]
                D[r, c] = v
                D[c, r] = v
        for b in range(nb):  # D += M G' + Q
            s0 = starts[b]
            if sizes[b] == 2:
                g00, g01, g10, g11 = Gt[u, b, 0, 0], Gt[u, b, 0, 1], Gt[u, b, 1, 0], Gt[u, b, 1, 1]
                for r in range(d):
                    m0 = M[r, s0]
                    m1 = M[r, s0 + 1]
                    D[r, s0] += m0 * g00 + m1 * g01
                    D[r, s0 + 1] += m0 * g10 + m1 * g11
                D[s0, s0] += Qt[u, b, 0, 0]
                D[s0, s0 + 1] += Qt[u, b, 0, 1]
                D[s0 + 1, s0] += Qt[u, b, 1, 0]
                D[s0 + 1, s0 + 1] += Qt[u, b, 1, 1]
            else:
                g00 = Gt[u, b, 0, 0]
                for r in range(d):
                    D[r, s0] += M[r, s0] * g00
                D[s0, s0] += Qt[u, b, 0, 0]
        for r in range(d):  # H = 1 on each block's first state
            acc = F0
            for b in range(nb):
                acc += D[r, starts[b]]
            dH[r] = acc
        s = R[i] + SkH[q]
        for b in range(nb):
            s += dH[starts[b]]
        inv = F1 / s
        for r in range(d):
            dH[r] += PkH[q, r]
        if i >= lo:
            isS[i] = float32(math.sqrt(inv))
            for r in range(d):
                K[i, r] = dH[r] * inv
        elif i == lo - 1:
            overlap[q] = float32(math.sqrt(inv))
        for r in range(d):
            for c in range(r, d):
                v = D[r, c] - dH[r] * dH[c] * inv
                D[r, c] = v
                D[c, r] = v


@cuda.jit
def numerator(ey, isS, K, w, z):
    """z_i = ey_i / sqrt(S_i) + K_i . w_i (gp.events.numerator_weights)."""
    i = cuda.grid(1)
    if i >= z.shape[0]:
        return
    acc = ey[i] * isS[i]
    for r in range(K.shape[1]):
        acc += K[i, r] * w[i, r]
    z[i] = acc

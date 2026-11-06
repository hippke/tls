"""Host side of the GPU kernels: single events (white and GP) and the period
fold (brute force, or FFA + polish with the same blocks, shift plans and tie
rules as gp.fold_ffa), returning the outputs of multiple_events_blocked."""

import functools
import math

import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver as _driver

from transitleastsquares.gp.fold_ffa import POLISH_R, REFINE_R, flat_plan, plan_blocks
from transitleastsquares.gpu import generated, kernels as K

THREADS = 128  # = kernels.FOLD_THREADS
ROW_THREADS = 32  # single events: threads per CUDA block
FFA_BUDGET = 1 << 27  # floats per FFA buffer and batch (512 MB)


def _row_j(off):
    return np.repeat(np.arange(len(off) - 1, dtype=np.int32), np.diff(off)).astype(np.int32)


def _f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def single_events_white(t, r, w, t0, delta, durs, nsubs, table, texp, mstep, off):
    """backends.events.single_events_white on the GPU: device array (rows, 2),
    FP32."""
    total = int(off[-1])
    d_ab = cuda.device_array((total, 2), np.float32)
    K.single_white[(total + ROW_THREADS - 1) // ROW_THREADS, ROW_THREADS](
        cuda.to_device(np.ascontiguousarray(t, float)), cuda.to_device(_f32(r * w)), cuda.to_device(_f32(w)),
        float(t0), float(delta), cuda.to_device(_f32(durs)), cuda.to_device(np.asarray(nsubs, np.int32)),
        cuda.to_device(_f32(table)), np.float32(texp), cuda.to_device(np.asarray(mstep, np.int64)),
        cuda.to_device(np.asarray(off, np.int64)), cuda.to_device(_row_j(off)), d_ab)
    return d_ab


def single_events_gp(t, z, eyd, A, Kg, S, H, W, t0, delta, durs, nsub, table, texp, mstep, off):
    """gp.events.single_events_split on the GPU (FP32; the gains, transitions
    and tail operator come from the FP64 CPU passes; the transitions as their
    diagonal blocks): device array (rows, 2)."""
    total = int(off[-1])
    d_ab = cuda.device_array((total, 2), np.float32)
    nsubs = np.full(len(durs), nsub, np.int32)
    dev = gains_on_device(A, Kg, S, H)
    generated.single_gp(tuple(int(x) for x in dev["sizes"].copy_to_host()))[(total + ROW_THREADS - 1) // ROW_THREADS, ROW_THREADS](
        cuda.to_device(np.ascontiguousarray(t, float)), cuda.to_device(_f32(z)), cuda.to_device(_f32(eyd)),
        dev["At"], dev["idx"], dev["K"], dev["isS"],
        cuda.to_device(_f32(W)), float(t0), float(delta), cuda.to_device(_f32(durs)), cuda.to_device(nsubs),
        cuda.to_device(_f32(table)), np.float32(texp), cuda.to_device(np.asarray(mstep, np.int64)),
        cuda.to_device(np.asarray(off, np.int64)), cuda.to_device(_row_j(off)), d_ab)
    return d_ab


def single_events_gp_dev(t, z, dev, W, t0, delta, durs, nsub, table, texp, mstep, off):
    """single_events_gp with the inputs already on the device (from
    tail_operators_gpu): z, W device arrays; dev the transitions and gains."""
    total = int(off[-1])
    d_ab = cuda.device_array((total, 2), np.float32)
    nsubs = np.full(len(durs), nsub, np.int32)
    generated.single_gp(tuple(int(x) for x in dev["sizes"].copy_to_host()))[(total + ROW_THREADS - 1) // ROW_THREADS, ROW_THREADS](
        cuda.to_device(np.ascontiguousarray(t, float)), z, cuda.to_device(np.zeros(0, np.float32)),
        dev["At"], dev["idx"], dev["K"], dev["isS"], W,
        float(t0), float(delta), cuda.to_device(_f32(durs)), cuda.to_device(nsubs), cuda.to_device(_f32(table)),
        np.float32(texp), cuda.to_device(np.asarray(mstep, np.int64)), cuda.to_device(np.asarray(off, np.int64)),
        cuda.to_device(_row_j(off)), d_ab)
    return d_ab


TAIL_SEGMENT = 256  # points per GPU thread in the tail recursion


def gains_on_device(A, Kg, S, H):
    """Gains and transitions from the CPU's passes on the device (FP32). A:
    the dense transitions; per point one table entry."""
    from transitleastsquares.gp.events import pack_blocks, transition_blocks

    starts, sizes = transition_blocks(A)
    if sizes.max() > 2:
        raise ValueError("GPU: transition blocks of more than 2 states")
    return dict(At=cuda.to_device(_f32(pack_blocks(A, starts, sizes))),
                idx=cuda.to_device(np.arange(len(A), dtype=np.int32)), starts=cuda.to_device(starts),
                sizes=cuda.to_device(sizes), K=cuda.to_device(_f32(Kg)), isS=cuda.to_device(_f32(1.0 / np.sqrt(S))),
                H=cuda.to_device(_f32(H)))


GPU_KALMAN_MIN = 8000  # points; below, the FP64 filter on the CPU is as fast
GPU_FIT_MIN = 2500  # points; below, the FP64 likelihood on the CPU is as fast
ELEMENT_BLOCK = 32  # threads per CUDA block: the blocks spread over the SMs (30), and more do not help
FIT_HEAD = 16  # the first points on the CPU, at least (diffuse start)


def _grid(nk):
    return (nk + ELEMENT_BLOCK - 1) // ELEMENT_BLOCK, ELEMENT_BLOCK


def _segment_length(n):
    """Points per segment that balance the kernel (a thread's 4.7 us per point) and the join on the CPU (2.4 us per segment)."""
    return max(64, int(math.sqrt(0.5 * n)))


class _Elements:
    """The elements of the segments of a fixed series on the device (FP64,
    generated.elements), for any model; what depends only on the data is set once."""

    def __init__(self, t, y, R):
        from transitleastsquares.gp.statespace import step_table

        self.t, self.y, self.R = t, y, R
        n = len(t)
        ns = max(1, n // _segment_length(n))
        edges = np.array([(k * n) // ns for k in range(ns + 1)], np.int64)
        q0 = min(int(np.searchsorted(edges, FIT_HEAD)), ns)
        self.n0, self.edges = int(edges[q0]), edges[q0:]
        self.nk = len(self.edges) - 1
        self.idx, self.steps = step_table(t)
        self.d = None

    def __call__(self, kinds, starts, pars, d, jit2=0.0):
        """The kernel for this model; returns the elements on the host, and At."""
        from transitleastsquares.gp.statespace import _block_sizes, step_transitions

        nk, nb = self.nk, len(kinds)
        At, Qt = step_transitions(self.steps, kinds, pars)
        if self.d is None:
            self.sizes = tuple(int(x) for x in _block_sizes(kinds))
            ne = len(self.steps)
            nd2, nd1 = nk * d * d, nk * d
            size = 3 * nd2 + 2 * nd1 + nk
            self.d = dict(idx=cuda.to_device(self.idx), y=cuda.to_device(self.y), R=cuda.to_device(self.R),
                          T=cuda.device_array((ne, 2, nb, 2, 2), np.float64), edges=cuda.to_device(self.edges),
                          E=cuda.device_array(size, np.float64))
            self.T = np.empty((ne, 2, nb, 2, 2))
            self.E = np.empty(size)
            E, o = self.E, 3 * nd2 + 2 * nd1
            self.h = (E[:nd2].reshape(nk, d, d), E[3 * nd2:3 * nd2 + nd1].reshape(nk, d), E[nd2:2 * nd2].reshape(nk, d, d),
                      E[2 * nd2:3 * nd2].reshape(nk, d, d), E[3 * nd2 + nd1:o].reshape(nk, d), E[o:])
            self.kernel = generated.elements(self.sizes)[_grid(nk)] if nk > 0 else None
        if nk > 0:
            dv = self.d
            self.T[:, 0], self.T[:, 1] = At, Qt
            _driver.host_to_device(dv["T"], self.T, self.T.nbytes)  # (copy_to_device's checks cost 0.3 ms, a fit makes thousands)
            self.kernel(dv["T"], dv["idx"], dv["y"], dv["R"], float(jit2), dv["edges"], nk, dv["E"])
            _driver.device_to_host(self.E, dv["E"], self.E.nbytes)
        return self.h, At


class KalmanLikelihood:
    """The log likelihood of fixed data under changing models (the fit), as
    gp.statespace.log_likelihood with the step table: the elements of the segments
    (FP64) on the device, joined in order on the host, so it is as exact as the
    sequential filter. Returns None if it is not finite: the caller then uses the
    FP64 filter."""

    def __init__(self, t, y, yerr):
        self.e = _Elements(t, y, yerr**2)
        self.t0, self.y0, self.e20 = t[:self.e.n0], y[:self.e.n0], yerr[:self.e.n0] ** 2

    def __call__(self, model):
        from transitleastsquares.gp.statespace import join_elements

        kinds, starts, pars, P0 = model.blocks()
        jit2 = model.jitter**2
        h, _ = self.e(kinds, starts, pars, P0.shape[0], jit2)
        ll = join_elements(self.t0, self.y0, self.e20 + jit2, kinds, starts, pars, P0, model.n_diffuse, self.e.n0, *h)
        return ll if np.isfinite(ll) else None


def _filtered(t, y, R, model, K_dtype):
    """The Kalman pass on the device (FP64 elements joined on the host, then the filter per
    segment): (elements' object, At, device K, 1 / sqrt(S), ey (FP64), and the head's K, S, ey)."""
    from transitleastsquares.gp.statespace import element_states

    kinds, starts, pars, P0 = model.blocks()
    n, d = len(t), P0.shape[0]
    e = _Elements(t, y, R)
    h, At = e(kinds, starts, pars, d)
    n0 = e.n0
    Kh, Sh, eyh, x0, P0s = element_states(t[:n0], y[:n0], R[:n0], kinds, starts, pars, P0, n0, *h)
    d_K, d_iS, d_ey = cuda.device_array((n, d), K_dtype), cuda.device_array(n, K_dtype), cuda.device_array(n, np.float64)
    dv = e.d
    if e.nk > 0:
        generated.filter_segments(e.sizes)[_grid(e.nk)](dv["T"], dv["idx"], dv["y"], dv["R"], dv["edges"],
                                                         cuda.to_device(x0), cuda.to_device(P0s), d_K, d_iS, d_ey)
    d_K[:n0].copy_to_device(Kh.astype(K_dtype))
    d_iS[:n0].copy_to_device((1.0 / np.sqrt(Sh)).astype(K_dtype))
    d_ey[:n0].copy_to_device(eyh)
    return e, At, d_K, d_iS, d_ey, Kh, Sh, eyh


def kalman_gpu(t, y, R, model, nseg=1):
    """The Kalman pass of a GPU search: (dev, ey) with the gains, 1 / sqrt(S)
    and transitions on the device (FP32, as gains_on_device) and ey (FP64) on the
    host. Long light curves: the pass runs on the device in FP64 (_filtered), with
    the transitions pooled within statespace.DT_TOL. Short ones: statespace.kalman_whiten on the host."""
    from transitleastsquares.gp.statespace import _block_sizes, kalman_whiten, transition_table

    kinds, starts, pars, P0 = model.blocks()
    if len(t) < GPU_KALMAN_MIN or len(t) < 2 * FIT_HEAD:
        idx, At, Qt = transition_table(t, kinds, pars)
        Kg, S, ey = kalman_whiten(t, y, R, kinds, starts, pars, P0, nseg)
        return dict(At=cuda.to_device(_f32(At)), idx=cuda.to_device(idx), starts=cuda.to_device(starts),
                    sizes=cuda.to_device(_block_sizes(kinds)), H=cuda.to_device(_f32(model.H)), K=cuda.to_device(_f32(Kg)),
                    isS=cuda.to_device(_f32(1.0 / np.sqrt(S))), kalman="short"), ey
    e, At, d_K, d_iS, d_ey, _, _, _ = _filtered(t, y, R, model, np.float32)
    return dict(At=cuda.to_device(_f32(At)), idx=e.d["idx"], starts=cuda.to_device(starts), sizes=cuda.to_device(_block_sizes(kinds)),
                H=cuda.to_device(_f32(model.H)), K=d_K, isS=d_iS, kalman="gpu"), d_ey.copy_to_host()


def tail_operators_gpu(dev, ey):
    """gp.events.tail_operators on the GPU in FP32 (segment-parallel: maps per
    segment on the device, joined backwards on the CPU in FP64, then W, w
    per segment on the device) and the numerator weights z. dev: gains and
    transitions (kalman_gpu or gains_on_device). Returns device arrays
    (W, w, z), W in its upper triangle; dev gets ey (single_events_gp_dev)."""
    from transitleastsquares.gp.statespace import tail_chain

    n, d = dev["K"].shape
    dev["ey"] = cuda.to_device(_f32(ey))
    nseg = max(1, (n - 1) // TAIL_SEGMENT)
    edges = np.array([(k * (n - 1)) // nseg for k in range(nseg + 1)], np.int64)
    d_edges = cuda.to_device(edges)
    Phi = cuda.device_array((nseg, d, d), np.float32)
    C = cuda.to_device(np.zeros((nseg, d, d), np.float32))
    cc = cuda.to_device(np.zeros((nseg, d), np.float32))
    args = (dev["At"], dev["idx"], dev["K"], dev["isS"], dev["ey"], d_edges)
    sizes = tuple(int(x) for x in dev["sizes"].copy_to_host())
    if nseg > 1:
        generated.tail_maps(sizes)[_grid(nseg - 1)](*args, Phi, C, cc)
    Wb, wb = tail_chain(*(x.copy_to_host().astype(float) for x in (Phi, C, cc)))
    W = cuda.device_array((n, d, d), np.float32)
    w = cuda.device_array((n, d), np.float32)
    W[n - 1:].copy_to_device(np.zeros((1, d, d), np.float32))
    w[n - 1:].copy_to_device(np.zeros((1, d), np.float32))
    generated.tail_fill(sizes)[_grid(nseg)](*args, cuda.to_device(_f32(Wb)), cuda.to_device(_f32(wb)), W, w)
    z = cuda.device_array(n, np.float32)
    K.numerator[(n + 255) // 256, 256](dev["ey"], dev["isS"], dev["K"], w, z)
    return W, w, z


@functools.lru_cache(maxsize=None)
def tree_nodes(m):
    """The merge nodes of an m-row FFA (as gp.fold_ffa._ffa): (start row,
    size, depth) of every node with >= 2 rows."""
    out = []

    def rec(start, size, depth):
        if size < 2:
            return
        out.append((start, size, depth))
        h = size // 2
        rec(start, h, depth + 1)
        rec(start + h, size - h, depth + 1)

    rec(0, m, 0)
    a = np.array(out, np.int64).reshape(-1, 3)
    return a[:, 0], a[:, 1], a[:, 2]


def _first_max(q, g, cols):
    """Per period q: the first candidate (input order) with the largest gain.
    Returns the selected rows of cols."""
    order = np.lexsort((-g, q))  # stable: ties keep the input order
    first = np.ones(len(order), bool)
    first[1:] = q[order][1:] != q[order][:-1]
    sel = order[first]
    return q[sel], [c[sel] for c in cols]


def fold(d_ab, off, M, periods, delta, mstep, jlo, jhi, ntr_min, depth_min=0.0, ffa=True, polish_r=POLISH_R,
         screen=False, coarse=1, refine_r=REFINE_R, stats=None):
    """multiple_events_blocked (ffa=False) or gp.fold_ffa.fold_ffa (ffa=True)
    on the GPU, with the same options (screen: duration screen; coarse:
    coarse epochs in the brute-force part). d_ab: device array of the single
    events (FP32). Returns dchi2, best epoch (full grid), best duration, sums
    A and B, zeros (host arrays)."""
    periods = np.ascontiguousarray(periods, float)
    jlo = np.asarray(jlo, np.int64)
    jhi = np.asarray(jhi, np.int64)
    NP = len(periods)
    Mj = np.diff(off)
    dj = mstep * delta
    if ffa:
        jb_hi, blocks = plan_blocks(Mj, dj, periods, jlo, jhi, screen)
    else:
        jb_hi, blocks = jhi, []
    d_off = cuda.to_device(np.asarray(off, np.int64))
    d_mstep = cuda.to_device(np.asarray(mstep, np.int64))
    d_periods = cuda.to_device(periods)
    cand = []  # (q, g, j, k, A, B) in the CPU's order: brute (j ascending), then FFA blocks

    # brute force: pairs (q, j), j = jlo .. jb_hi, duration-major: the CUDA
    # blocks of one duration run together, so its single events stay in L2
    n_pair = np.maximum(jb_hi - jlo + 1, 0)
    if n_pair.sum() > 0:
        jj = np.arange(len(Mj))[:, None]
        sel = (jj >= jlo[None, :]) & (jj <= jb_hi[None, :])
        if screen:  # even durations and each period's shortest and longest
            sel &= (jj % 2 == 0) | (jj == jlo[None, :]) | (jj == jhi[None, :])
        pj, pq = np.nonzero(sel)
        n = len(pq)
        out = [cuda.device_array(n, np.float32) for _ in range(3)] + [cuda.device_array(n, np.int32)]
        K.fold_brute[n, THREADS](d_ab.view(np.complex64).reshape(-1), d_off, d_mstep, float(delta), d_periods,
                                 cuda.to_device(pq.astype(np.int32)), cuda.to_device(pj.astype(np.int32)),
                                 int(ntr_min), np.float32(depth_min), int(coarse), *out)
        g, A, B, k = (x.copy_to_host() for x in out)
        cand.append((pq, g, pj, k, A, B))

    # FFA + polish, in batches of blocks that fit the buffer budget
    if blocks:
        bj = np.array([b[0] for b in blocks], np.int64)
        bp = np.array([b[1] for b in blocks], np.int64)
        bm = -(-Mj[bj] // bp)
        plan = [cuda.to_device(a) for a in flat_plan(np.unique(bm))]
        sizes = 3 * bm * bp
        b0 = 0
        while b0 < len(blocks):
            b1 = b0 + 1
            while b1 < len(blocks) and sizes[b0:b1 + 1].sum() <= FFA_BUDGET:
                b1 += 1
            cand.append(_ffa_batch(d_ab, d_off, d_mstep, d_periods, delta, blocks[b0:b1], bj[b0:b1], bp[b0:b1],
                                   bm[b0:b1], sizes[b0:b1], plan, ntr_min, depth_min, polish_r))
            b0 = b1
    if stats is not None:
        stats.update(pairs_brute=int(n_pair.sum()), blocks=len(blocks))

    # per period the first maximum over its candidates in the CPU's order
    # (durations ascending; brute-force before FFA): one dense table per
    # candidate source, argmax over durations (first maximum)
    J = len(Mj)
    dchi2 = np.zeros(NP)
    best_k0 = np.zeros(NP, np.int64)
    best_j = np.zeros(NP, np.int64)
    best_A = np.zeros(NP)
    best_B = np.zeros(NP)
    for q, g, j, k, A, B in cand:
        G = np.zeros((J, NP), np.float32)
        G[j, q] = g
        jb = np.argmax(G, axis=0)
        gb = G[jb, np.arange(NP)].astype(float)
        better = gb > dchi2  # strictly: earlier sources win ties
        if not better.any():
            continue
        idx = np.full((J, NP), -1, np.int64)
        idx[j, q] = np.arange(len(q))
        sel = idx[jb[better], np.nonzero(better)[0]]
        qs = q[sel]
        dchi2[qs] = g[sel]
        best_j[qs] = j[sel]
        best_k0[qs] = k[sel].astype(np.int64) * mstep[j[sel]]
        best_A[qs] = A[sel]
        best_B[qs] = B[sel]
    if screen:  # neighbours of the best duration, locally around its best epoch (as fold_ffa)
        for dn in (-1, 1):
            jn = best_j + dn
            ok = (dchi2 > 0) & (jn >= jlo) & (jn <= jhi) & ~((jn % 2 == 0) | (jn == jlo) | (jn == jhi))
            tq = np.nonzero(ok)[0]
            if len(tq) == 0:
                continue
            tj = jn[tq]
            center = np.round(best_k0[tq] / mstep[tj]).astype(np.int64)
            out = [cuda.device_array(len(tq), np.float32) for _ in range(3)] + [cuda.device_array(len(tq), np.int32)]
            K.refine_local[len(tq), THREADS](d_ab.view(np.complex64).reshape(-1), d_off, d_mstep, float(delta),
                                             d_periods, cuda.to_device(tq.astype(np.int32)),
                                             cuda.to_device(tj.astype(np.int32)), cuda.to_device(center),
                                             int(refine_r), int(ntr_min), np.float32(depth_min), *out)
            g, A, B, k = (x.copy_to_host() for x in out)
            better = g.astype(float) > dchi2[tq]
            q = tq[better]
            dchi2[q] = g[better]
            best_j[q] = tj[better]
            best_k0[q] = k[better].astype(np.int64) * mstep[tj[better]]
            best_A[q] = A[better]
            best_B[q] = B[better]
    return dchi2, best_k0, best_j, best_A, best_B, np.zeros(NP)


def _ffa_batch(d_ab, d_off, d_mstep, d_periods, delta, blocks, bj, bp, bm, sizes, plan, ntr_min, depth_min, polish_r):
    nb = len(blocks)
    base = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    total = int(sizes.sum())
    buf = [cuda.device_array(total, np.float32), cuda.device_array(total, np.float32)]
    d_bj, d_bp, d_base, d_bm = (cuda.to_device(a.astype(np.int64)) for a in (bj, bp, base, bm))
    K.ffa_load[nb, THREADS](d_ab, d_off, d_bj, d_bp, d_base, d_bm, buf[0], buf[1])
    # nodes of all trees, by depth (deepest first)
    nodes = []
    for b in range(nb):
        s, z, dp = tree_nodes(int(bm[b]))
        nodes.append((np.full(len(s), base[b]), np.full(len(s), bp[b]), s, z, dp))
    nb_, np_, ns_, nz_, nd_ = (np.concatenate(x) for x in zip(*nodes))
    for depth in range(int(nd_.max()), -1, -1):
        sel = nd_ == depth
        if not sel.any():
            continue
        src, dst = buf[(depth + 1) % 2], buf[depth % 2]
        args = [cuda.to_device(a[sel].astype(np.int64)) for a in (nb_, np_, ns_, nz_)]
        K.ffa_level[int(sel.sum()), THREADS](*args, *plan[:4], src, dst)
    # targets
    tq = np.concatenate([b[2] for b in blocks]).astype(np.int32)
    tb = np.repeat(np.arange(nb), [len(b[2]) for b in blocks]).astype(np.int32)
    n = len(tq)
    out = [cuda.device_array(n, np.float32) for _ in range(3)] + [cuda.device_array(n, np.int32)]
    K.ffa_eval_polish[n, THREADS](buf[0], d_ab, d_off, d_mstep, float(delta), d_periods, cuda.to_device(tq),
                                  cuda.to_device(tb), d_bj, d_bp, d_base, d_bm, int(ntr_min), np.float32(depth_min),
                                  int(polish_r), *out)
    g, A, B, k = (x.copy_to_host() for x in out)
    return tq.astype(np.int64), g, bj[tb], k, A, B


def trend_gpu(model, t, y, R, nseg=1):
    """Posterior mean of the baseline (ppt) without a smoother: f = y - R z with z = C^-1 y
    (statespace.mean_adjoint) on the device in FP64: the Kalman pass of _filtered, then the backward
    recursion per segment, as affine maps joined on the host, and z. Short series: mean_adjoint on the host."""
    from transitleastsquares.gp.statespace import adjoint_chain, adjoint_head, mean_adjoint, smoothed_mean

    n = len(t)
    kinds, starts, pars, P0 = model.blocks()
    if n < GPU_KALMAN_MIN or n < 2 * FIT_HEAD:
        return mean_adjoint(t, y, R, kinds, starts, pars, P0) if nseg == 1 else smoothed_mean(t, y, R, kinds, starts, pars, P0, nseg)
    d = P0.shape[0]
    e, _, d_K, d_iS, d_ey, Kh, Sh, eyh = _filtered(t, y, R, model, np.float64)
    dv, nk, n0 = e.d, e.nk, e.n0
    W, v = cuda.device_array((nk, d, d), np.float64), cuda.device_array((nk, d), np.float64)
    generated.adjoint_maps(e.sizes)[_grid(nk)](dv["T"], dv["idx"], d_K, d_iS, d_ey, dv["edges"], W, v)
    w, w0 = adjoint_chain(W.copy_to_host(), v.copy_to_host(), np.zeros(d))
    d_z = cuda.device_array(n, np.float64)
    generated.adjoint_fill(e.sizes)[_grid(nk)](dv["T"], dv["idx"], d_K, d_iS, d_ey, dv["edges"], cuda.to_device(w), d_z)
    z = d_z.copy_to_host()
    out = y - R * z
    adjoint_head(t[:n0], y[:n0], R[:n0], kinds, starts, pars, Kh, Sh, eyh, w0, out)
    return out

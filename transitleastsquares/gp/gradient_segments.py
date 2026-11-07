"""The gradient of the rotation model's log likelihood in segments, for the
GPU (gpu.gradient) and as its CPU reference (internal; data in ppt).

The gradient from the RTS smoother (Fisher's identity): with the smoothed
mean and covariance (ms, Ps) at point i, the predicted (xp, Pp) and
d = ms - xp, the adjoints of the predicted state are
dl/dxp = Pp^-1 d and dl/dPp = (Pp^-1 (Ps - Pp + d d') Pp^-1) / 2, and those
of the step's A and Q follow as in gradient.loglik_adjoint (equal to it to
~1e-9 on Talens light curves). Unlike the adjoint, this runs in parallel:
(1) the segment elements (F, c, Pe, J, eta), joined forwards, give the
filtered state before each segment (as gpu.driver.KalmanLikelihood), and
joined backwards the information of the data after it (J, eta of the
elements after it): together the smoothed state at each segment boundary;
(2) per segment, the filter again, storing the filtered states; (3) per
point, the predicted covariance, its Cholesky factor and the smoother gain;
(4) per segment, the smoother backwards from its last point, storing the
smoothed states; (5) per point, the adjoints of the step into it, added per
step-table entry. Steps (2) and (4) are sequential, about 900 multiply-adds
per point; (3) and (5) are parallel over points.

The steps (SRC) are compiled for the CPU here (build("cpu")) and for CUDA in
gpu.gradient (build("cuda")) from the same source; gpu.gradient replaces (2)
and (4) by kernels in scalars (gpu.generated). Rotation model only: 7 states
in the blocks offset (1), three SHO (2 each); H picks states 0, 1, 3, 5; the
transitions T (entries, 2, 4, 2, 2): A and Q per block, as
gpu.driver._Elements uploads them.
"""

import math

import numba
import numpy as np

from transitleastsquares.gp import gradient
from transitleastsquares.gp.statespace import (
    _combine,
    _combine_work,
    _filter_segment,
    _segment_element,
    rotation_model,
    step_table,
    step_transitions,
)

D = 7

SRC = '''
def a_mul(T, u, X, out):
    """out = A X (A block diagonal: entry u of T)."""
    a = T[u, 0, 0, 0, 0]
    for c in range(7):
        out[0, c] = a * X[0, c]
    for k in range(1, 4):
        s0 = 2 * k - 1
        a00 = T[u, 0, k, 0, 0]
        a01 = T[u, 0, k, 0, 1]
        a10 = T[u, 0, k, 1, 0]
        a11 = T[u, 0, k, 1, 1]
        for c in range(7):
            x0 = X[s0, c]
            x1 = X[s0 + 1, c]
            out[s0, c] = a00 * x0 + a01 * x1
            out[s0 + 1, c] = a10 * x0 + a11 * x1


def mul_at_q(T, u, X, out):
    """out = X A' + Q."""
    a = T[u, 0, 0, 0, 0]
    for r in range(7):
        out[r, 0] = X[r, 0] * a
    for k in range(1, 4):
        s0 = 2 * k - 1
        a00 = T[u, 0, k, 0, 0]
        a01 = T[u, 0, k, 0, 1]
        a10 = T[u, 0, k, 1, 0]
        a11 = T[u, 0, k, 1, 1]
        for r in range(7):
            x0 = X[r, s0]
            x1 = X[r, s0 + 1]
            out[r, s0] = x0 * a00 + x1 * a01
            out[r, s0 + 1] = x0 * a10 + x1 * a11
    out[0, 0] += T[u, 1, 0, 0, 0]
    for k in range(1, 4):
        s0 = 2 * k - 1
        for i in range(2):
            for j in range(2):
                out[s0 + i, s0 + j] += T[u, 1, k, i, j]


def a_vec(T, u, x, out):
    out[0] = T[u, 0, 0, 0, 0] * x[0]
    for k in range(1, 4):
        s0 = 2 * k - 1
        out[s0] = T[u, 0, k, 0, 0] * x[s0] + T[u, 0, k, 0, 1] * x[s0 + 1]
        out[s0 + 1] = T[u, 0, k, 1, 0] * x[s0] + T[u, 0, k, 1, 1] * x[s0 + 1]


def chol(Pm, L):
    """Cholesky factor (lower) of the symmetric Pm, with the reciprocals of its diagonal on the diagonal of L
    (FP64 division is slow on the GPU: solve multiplies); a ridge of 1e-12 max(diag) if Pm is not positive
    definite (as statespace._chol_solve). Returns False if that fails too."""
    ridge = F0
    for attempt in range(2):
        good = True
        for i in range(7):
            for j in range(i + 1):
                acc = Pm[i, j]
                if i == j:
                    acc += ridge
                for k in range(j):
                    acc -= L[i, k] * L[j, k]
                if i == j:
                    if not acc > F0:
                        good = False
                        break
                    L[i, j] = F1 / math.sqrt(acc)
                else:
                    L[i, j] = acc * L[j, j]
            if not good:
                break
        if good:
            return True
        dmax = F0
        for i in range(7):
            dmax = max(dmax, abs(Pm[i, i]))
        ridge = RIDGE * dmax + TINY
    return False


def solve(L, B, X, trans):
    """X = (L L')^-1 B (trans: B'); L from chol (reciprocal diagonal)."""
    for c in range(7):
        for i in range(7):
            acc = B[c, i] if trans else B[i, c]
            for k in range(i):
                acc -= L[i, k] * X[k, c]
            X[i, c] = acc * L[i, i]
        for i in range(6, -1, -1):
            acc = X[i, c]
            for k in range(i + 1, 7):
                acc -= L[k, i] * X[k, c]
            X[i, c] = acc * L[i, i]


def solve_vec(L, b, x):
    for i in range(7):
        acc = b[i]
        for k in range(i):
            acc -= L[i, k] * x[k]
        x[i] = acc * L[i, i]
    for i in range(6, -1, -1):
        acc = x[i]
        for k in range(i + 1, 7):
            acc -= L[k, i] * x[k]
        x[i] = acc * L[i, i]


def measurement(ms, Ps, yi, Ri):
    """dl/dR at a point from the smoothed state: -(1 / R - ((y - H ms)^2 + H Ps H') / R^2) / 2."""
    hm = ms[0] + ms[1] + ms[3] + ms[5]
    hph = F0
    for a in (0, 1, 3, 5):
        for b in (0, 1, 3, 5):
            hph += Ps[a, b]
    r = yi - hm
    return -HALF * (F1 / Ri - (r * r + hph) / (Ri * Ri))


def seg_forward(T, idx, y, R, jit2, a, b, x, P, XF, PF, M, v, PH):
    """The filter over the points a..b-1 from the filtered state (x, P) of point a-1 (a = 0: the prior,
    no prediction), storing the filtered states XF[i], PF[i]."""
    for i in range(a, b):
        if i > 0:
            u = idx[i]
            a_mul(T, u, P, M)
            mul_at_q(T, u, M, P)  # P = A P A' + Q (M = A P; P M' would be the same)
            a_vec(T, u, x, v)
            for r in range(7):
                x[r] = v[r]
        for r in range(7):
            PH[r] = P[r, 0] + P[r, 1] + P[r, 3] + P[r, 5]
        s = R[i] + jit2 + PH[0] + PH[1] + PH[3] + PH[5]
        innov = y[i] - (x[0] + x[1] + x[3] + x[5])
        for r in range(7):
            x[r] += PH[r] * innov / s
        for r in range(7):
            for c in range(r, 7):
                w = P[r, c] - PH[r] * PH[c] / s
                P[r, c] = w
                P[c, r] = w
        for r in range(7):
            XF[i, r] = x[r]
            for c in range(7):
                PF[i, r, c] = P[r, c]


def point_gains(T, idx, i, XF, PF, XP, PP, LL, GTS, Pf, B, Pp, L, GT, v):
    """The step into point i + 1: predicted mean and covariance XP, PP[i + 1], its Cholesky factor LL[i + 1]
    (lower triangle) and G' = Pp^-1 A Pf in GTS[i], in the work arrays and written once (global memory is
    slow on the GPU). Returns False if Pp is not positive definite."""
    u = idx[i + 1]
    for r in range(7):
        for c in range(7):
            Pf[r, c] = PF[i, r, c]
    a_mul(T, u, Pf, B)
    mul_at_q(T, u, B, Pp)
    if not chol(Pp, L):
        return False
    solve(L, B, GT, False)
    a_vec(T, u, XF[i], v)
    for r in range(7):
        XP[i + 1, r] = v[r]
        for c in range(7):
            PP[i + 1, r, c] = Pp[r, c]
            GTS[i, r, c] = GT[r, c]
            LL[i + 1, r, c] = L[r, c] if c <= r else F0
    return True


def seg_smooth(a, b, XF, PF, XP, PP, GTS, MS, PS, dP, W, dl):
    """The smoother from point b - 1 (MS, PS given there) to point a: ms = xf + G (ms' - xp'),
    Ps = Pf + G (Ps' - Pp') G'."""
    for i in range(b - 2, a - 1, -1):
        for r in range(7):
            dl[r] = MS[i + 1, r] - XP[i + 1, r]
            for c in range(7):
                dP[r, c] = PS[i + 1, r, c] - PP[i + 1, r, c]
        for r in range(7):
            acc = XF[i, r]
            for l in range(7):
                acc += GTS[i, l, r] * dl[l]
            MS[i, r] = acc
        for r in range(7):
            for c in range(7):
                acc = F0
                for l in range(7):
                    acc += dP[r, l] * GTS[i, l, c]
                W[r, c] = acc
        for r in range(7):
            for c in range(7):
                acc = PF[i, r, c]
                for l in range(7):
                    acc += GTS[i, l, r] * W[l, c]
                PS[i, r, c] = acc


def point_grad(y, R, jit2, i, MS, PS, XP, PP, LL, GTS, Y, Z, dP, w, dl, L, loc):
    """The adjoints of the step into point i (into loc, blocks 1-3: A, then Q) and dl/dR at point i."""
    for r in range(7):
        dl[r] = MS[i, r] - XP[i, r]
        for c in range(7):
            dP[r, c] = PS[i, r, c] - PP[i, r, c]
            L[r, c] = LL[i, r, c]
    solve(L, dP, Y, False)  # Pp^-1 dP
    solve_vec(L, dl, w)  # Pp^-1 d
    solve(L, Y, Z, True)  # Pp^-1 dP Pp^-1
    for k in range(3):  # dl/dA = Pp^-1 dP G' + w ms', dl/dQ = (Pp^-1 dP Pp^-1 + w w') / 2, block k + 1
        s0 = 2 * k + 1
        for p in range(2):
            for q in range(2):
                acc = w[s0 + p] * MS[i - 1, s0 + q]
                for l in range(7):
                    acc += Y[s0 + p, l] * GTS[i - 1, l, s0 + q]
                loc[k * 8 + p * 2 + q] += acc
                loc[k * 8 + 4 + p * 2 + q] += HALF * (Z[s0 + p, s0 + q] + w[s0 + p] * w[s0 + q])
    for r in range(7):  # (dP is free: the smoothed covariance for dl/dR)
        for c in range(7):
            dP[r, c] = PS[i, r, c]
    return measurement(MS[i], dP, y[i], R[i] + jit2)


def seg_grad(T, idx, y, R, jit2, a, bnd, last, XF, PF, ms, Ps, ACC, B, Pp, L, GT, Y, Z, v1, v2, w, loc):
    """The smoother backwards from the smoothed state (ms, Ps) at point bnd to point a (in place), adding
    the adjoints of the steps into point i in (a, bnd] to ACC[u, 0] (A) and ACC[u, 1] (Q) of their
    entries u (blocks 1-3; flush), and returning the sum of dl/dR over the points a..bnd-1 (and bnd if
    last). Returns NaN if a predicted covariance is not positive definite."""
    rs = F0
    if last:
        rs += measurement(ms, Ps, y[bnd], R[bnd] + jit2)
    ucur = -1
    for i in range(bnd, a, -1):
        u = idx[i]
        if u != ucur:
            if ucur >= 0:
                flush(ACC, ucur, loc)
            for j in range(24):
                loc[j] = F0
            ucur = u
        Pf = PF[i - 1]
        a_mul(T, u, Pf, B)  # B = A Pf
        mul_at_q(T, u, B, Pp)  # Pp
        if not chol(Pp, L):
            return math.nan
        solve(L, B, GT, False)  # GT = G'
        a_vec(T, u, XF[i - 1], v1)  # xp
        for r in range(7):
            v2[r] = ms[r] - v1[r]  # d = ms - xp
            for c in range(7):
                Z[r, c] = Ps[r, c] - Pp[r, c]  # dP
        solve(L, Z, Y, False)  # Y = Pp^-1 dP
        solve_vec(L, v2, w)  # w = Pp^-1 d
        solve(L, Y, Pp, True)  # Pp^-1 dP Pp^-1 (Pp is no longer needed)
        for r in range(7):  # the smoothed mean at i - 1: xf + G d
            acc = XF[i - 1, r]
            for l in range(7):
                acc += GT[l, r] * v2[l]
            v1[r] = acc
        for k in range(3):  # dl/dA = Y GT + w ms' and dl/dQ = (Pp^-1 dP Pp^-1 + w w') / 2, block k + 1
            s0 = 2 * k + 1
            for p in range(2):
                for q in range(2):
                    acc = w[s0 + p] * v1[s0 + q]
                    for l in range(7):
                        acc += Y[s0 + p, l] * GT[l, s0 + q]
                    loc[k * 8 + p * 2 + q] += acc
                    loc[k * 8 + 4 + p * 2 + q] += 0.5 * (Pp[s0 + p, s0 + q] + w[s0 + p] * w[s0 + q])
        for r in range(7):  # the smoothed covariance at i - 1: Pf + G dP G'
            for c in range(7):
                acc = F0
                for l in range(7):
                    acc += Z[r, l] * GT[l, c]
                B[r, c] = acc
        for r in range(7):
            ms[r] = v1[r]
            for c in range(7):
                acc = Pf[r, c]
                for l in range(7):
                    acc += GT[l, r] * B[l, c]
                Ps[r, c] = acc
        rs += measurement(ms, Ps, y[i - 1], R[i - 1] + jit2)
    if ucur >= 0:
        flush(ACC, ucur, loc)
    return rs
'''

FLUSH_CPU = '''
def flush(ACC, u, loc):
    for k in range(3):
        for p in range(2):
            for q in range(2):
                ACC[u, 0, k + 1, p, q] += loc[k * 8 + p * 2 + q]
                ACC[u, 1, k + 1, p, q] += loc[k * 8 + 4 + p * 2 + q]
'''

NAMES = ("a_mul", "mul_at_q", "a_vec", "chol", "solve", "solve_vec", "measurement", "flush", "seg_forward", "seg_grad",
         "point_gains", "seg_smooth", "point_grad")


CONSTANTS = {np.float64: dict(F0=0.0, F1=1.0, HALF=0.5, RIDGE=1e-12, TINY=1e-300),
             np.float32: dict(F0=np.float32(0.0), F1=np.float32(1.0), HALF=np.float32(0.5), RIDGE=np.float32(1e-6),
                              TINY=np.float32(1e-37))}


def build(target, flush_src=FLUSH_CPU, extra=None, dtype=np.float64):
    """The functions of SRC compiled for target ("cpu": numba.njit; "cuda": device functions), as a dict;
    dtype: the precision of SRC's constants (with arrays of that type the arithmetic stays in it)."""
    ns = dict(math=math, **CONSTANTS[dtype])
    if extra:
        ns.update(extra)
    exec(compile(SRC + flush_src, __file__, "exec"), ns)  # noqa: S102 (fixed source above)
    if target == "cpu":
        jit = numba.njit(nogil=True)
    else:
        from numba import cuda

        jit = cuda.jit(device=True)
    for name in NAMES:
        ns[name] = jit(ns[name])
    return ns


_CPU = None


def _cpu():
    global _CPU
    if _CPU is None:
        _CPU = build("cpu")
    return _CPU


@numba.njit(cache=True, nogil=True)
def join_forward(t, y, R, kinds, starts, pars, P0, n_diffuse, n0, F, c, Pe, J, eta, const):
    """join_elements and element_states in one pass: the log likelihood, the filtered state before each
    segment (x0, P0s) and at the last point."""
    nk, d = len(const), P0.shape[0]
    x, P = np.zeros(d), P0.copy()
    ll = _filter_segment(t, y, R, kinds, starts, pars, x, P, 0, n0, n_diffuse)
    x0, P0s = np.empty((nk, d)), np.empty((nk, d, d))
    work = _combine_work(d)
    for k in range(nk):
        x0[k] = x
        P0s[k] = P
        ll += _combine(x, P, F[k], c[k], Pe[k], J[k], eta[k], const[k], work)
    return ll, x0, P0s, x, P


@numba.njit(cache=True, nogil=True, fastmath={"contract"})
def _lu_solve_cols(Am, B, X):
    """Solve Am X = B (Am d x d, B d x m) by Gaussian elimination with partial pivoting, as
    statespace._lu_solve; Am and B are overwritten."""
    d, m = Am.shape[0], B.shape[1]
    for k in range(d):
        p = k
        for i in range(k + 1, d):
            if abs(Am[i, k]) > abs(Am[p, k]):
                p = i
        if p != k:
            for j in range(d):
                Am[k, j], Am[p, j] = Am[p, j], Am[k, j]
            for j in range(m):
                B[k, j], B[p, j] = B[p, j], B[k, j]
        for i in range(k + 1, d):
            f = Am[i, k] / Am[k, k]
            for j in range(k, d):
                Am[i, j] -= f * Am[k, j]
            for j in range(m):
                B[i, j] -= f * B[k, j]
    for j in range(m):
        for i in range(d - 1, -1, -1):
            acc = B[i, j]
            for k in range(i + 1, d):
                acc -= Am[i, k] * X[k, j]
            X[i, j] = acc / Am[i, i]


@numba.njit(cache=True, nogil=True)
def info_backward(F, c, Pe, J, eta):
    """The information of the data from each segment on about the state s before it,
    exp(-s'Jb s / 2 + etab's): the element's own (J, eta) and that of the elements after it, through the
    segment, J' = F' (I + Jb Pe)^-1 Jb F, eta' = F' (I + Jb Pe)^-1 (etab - Jb c). Explicit loops in
    preallocated arrays (numba takes BLAS for @ from SciPy, which is not a dependency)."""
    nk, d = len(eta), eta.shape[1]
    Jbs, ebs = np.empty((nk, d, d)), np.empty((nk, d))
    Am, Bm, X, M = np.empty((d, d)), np.empty((d, d + 1)), np.empty((d, d + 1)), np.empty((d, d))
    v = np.empty(d)
    for k in range(nk - 1, -1, -1):
        Jt, et = Jbs[k], ebs[k]
        for r in range(d):
            et[r] = eta[k, r]
            for q in range(d):
                Jt[r, q] = J[k, r, q]
        if k < nk - 1:
            Jb, eb = Jbs[k + 1], ebs[k + 1]
            for r in range(d):  # Am = I + Jb Pe; X = Am^-1 [Jb, eb]
                for q in range(d):
                    acc = 1.0 if r == q else 0.0
                    for l in range(d):
                        acc += Jb[r, l] * Pe[k, l, q]
                    Am[r, q] = acc
                    Bm[r, q] = Jb[r, q]
                Bm[r, d] = eb[r]
            _lu_solve_cols(Am, Bm, X)
            for r in range(d):  # M = Jc F, v = ec - Jc c
                acc = X[r, d]
                for l in range(d):
                    acc -= X[r, l] * c[k, l]
                v[r] = acc
                for q in range(d):
                    acc = 0.0
                    for l in range(d):
                        acc += X[r, l] * F[k, l, q]
                    M[r, q] = acc
            for r in range(d):  # Jt += F' M, et += F' v
                acc = 0.0
                for l in range(d):
                    acc += F[k, l, r] * v[l]
                et[r] += acc
                for q in range(d):
                    acc = 0.0
                    for l in range(d):
                        acc += F[k, l, r] * M[l, q]
                    Jt[r, q] += acc
        for r in range(d):
            for q in range(r + 1, d):
                w = 0.5 * (Jt[r, q] + Jt[q, r])
                Jt[r, q] = w
                Jt[q, r] = w
    return Jbs, ebs


@numba.njit(cache=True, nogil=True)
def boundary_smoothed(Jbs, ebs, x0, P0s, x_last, P_last):
    """The smoothed state at the point before each segment (ms_b[k], Ps_b[k]; k = nk: the last point, the
    filtered state there): the filtered state x0[k], P0s[k] conditioned on the information Jbs[k], ebs[k]."""
    nk, d = len(ebs), len(x_last)
    ms_b, Ps_b = np.empty((nk + 1, d)), np.empty((nk + 1, d, d))
    ms_b[nk], Ps_b[nk] = x_last, P_last
    Am, Bm, X = np.empty((d, d)), np.empty((d, d + 1)), np.empty((d, d + 1))
    v = np.empty(d)
    for k in range(nk):
        Jt, et = Jbs[k], ebs[k]
        for r in range(d):  # v = et - Jt x0
            acc = et[r]
            for l in range(d):
                acc -= Jt[r, l] * x0[k, l]
            v[r] = acc
        for r in range(d):  # Am = I + P Jt; X = Am^-1 [P, P v]
            acc = 0.0
            for l in range(d):
                acc += P0s[k, r, l] * v[l]
            Bm[r, d] = acc
            for q in range(d):
                acc = 1.0 if r == q else 0.0
                for l in range(d):
                    acc += P0s[k, r, l] * Jt[l, q]
                Am[r, q] = acc
                Bm[r, q] = P0s[k, r, q]
        _lu_solve_cols(Am, Bm, X)
        for r in range(d):  # Pc = X[:, :d] (symmetrised), ms = x0 + Pc v = x0 + X[:, d]
            ms_b[k, r] = x0[k, r] + X[r, d]
            for q in range(d):
                Ps_b[k, r, q] = 0.5 * (X[r, q] + X[q, r])
    return ms_b, Ps_b


def join_backward(F, c, Pe, J, eta, x0, P0s, x_last, P_last):
    """The smoothed state at the point before each segment and at the last point (info_backward and
    boundary_smoothed)."""
    Jbs, ebs = info_backward(F, c, Pe, J, eta)
    return boundary_smoothed(Jbs, ebs, x0, P0s, x_last, P_last)


def head_and_prior(fns, T, idx, y, R, jit2, P0, n0, ms, Ps, ACC):
    """The first n0 points on the CPU (from the prior): the filter, the smoother backwards from the
    smoothed state at point n0 - 1 (ms, Ps; in place; dl/dR there included), and the adjoints of P0. The likelihood leaves out the
    first point's innovation (n_diffuse = 1), so its derivative is taken off. Returns (P0b, sum of dl/dR)."""
    XF, PF = np.empty((n0, D)), np.empty((n0, D, D))
    x, P = np.zeros(D), P0.copy()
    w = [np.empty((D, D)) for _ in range(6)]
    v = [np.empty(D) for _ in range(3)]
    fns["seg_forward"](T, idx, y, R, jit2, 0, n0, x, P, XF, PF, w[0], v[0], v[1])
    rs = fns["seg_grad"](T, idx, y, R, jit2, 0, n0 - 1, True, XF, PF, ms, Ps, ACC, *w, *v, np.empty(24))
    if not np.isfinite(rs):
        return None, rs
    p = np.diag(P0)
    P0b = 0.5 * ((Ps + np.outer(ms, ms)) / np.outer(p, p) - np.diag(1.0 / p))
    h = np.zeros(D)
    h[[0, 1, 3, 5]] = 1.0
    S0 = h @ P0 @ h + R[0] + jit2
    wt = -0.5 * (1.0 / S0 - y[0] ** 2 / S0**2)
    return P0b - wt * np.outer(h, h), rs - wt


def transitions(lx, steps):
    """The rotation model of lx, its T (entries, 2, 4, 2, 2), P0 and jitter^2."""
    model = rotation_model(dict(zip(gradient.ROT_KEYS, np.exp(lx))))
    kinds, starts, pars, P0 = model.blocks()
    At, Qt = step_transitions(steps, kinds, pars)
    return model, (kinds, starts, pars, P0), np.ascontiguousarray(np.stack([At, Qt], axis=1)), model.jitter**2


class SegmentGradient:
    """The log likelihood and its gradient in nseg segments on the CPU, as gpu.gradient computes them
    (the reference for its tests; slower than gradient.RotationGradient). Data in ppt."""

    def __init__(self, t, y, yerr, nseg=8, n0=16):
        self.t, self.y, self.e2 = (np.asarray(a, float) for a in (t, y, np.asarray(yerr, float) ** 2))
        self.idx, self.steps = step_table(self.t)
        n = len(self.t)
        self.edges = np.array([n0 + ((n - n0) * k) // nseg for k in range(nseg + 1)], np.int64)

    def __call__(self, lx):
        fns = _cpu()
        lx = np.asarray(lx, float)
        model, (kinds, starts, pars, P0), T, jit2 = transitions(lx, self.steps)
        t, y, R, idx, edges = self.t, self.y, self.e2, self.idx, self.edges
        n, nk, d, n0 = len(y), len(edges) - 1, D, int(edges[0])
        Rj = R + jit2
        F, c, Pe, J, eta, const = (np.empty((nk, d, d)), np.empty((nk, d)), np.empty((nk, d, d)), np.empty((nk, d, d)),
                                   np.empty((nk, d)), np.empty(nk))
        for k in range(nk):
            const[k] = _segment_element(t, y, Rj, kinds, starts, pars, edges[k], edges[k + 1], F[k], c[k], Pe[k], J[k],
                                        eta[k])
        ll, x0, P0s, xl, Pl = join_forward(t, y, Rj, kinds, starts, pars, P0, model.n_diffuse, n0, F, c, Pe, J, eta,
                                           const)
        ms_b, Ps_b = join_backward(F, c, Pe, J, eta, x0, P0s, xl, Pl)
        XF, PF, XP, PP, LL, GTS, MS, PS = (np.zeros((n, d)), np.zeros((n, d, d)), np.zeros((n, d)), np.zeros((n, d, d)),
                                           np.zeros((n, d, d)), np.zeros((n, d, d)), np.zeros((n, d)), np.zeros((n, d, d)))
        M, B, W, Y, Z = (np.empty((d, d)) for _ in range(5))
        v1, v2 = np.empty(d), np.empty(d)
        for k in range(nk):
            x, P = x0[k].copy(), P0s[k].copy()
            fns["seg_forward"](T, idx, y, R, jit2, edges[k], edges[k + 1], x, P, XF, PF, M, v1, v2)
        XF[n0 - 1], PF[n0 - 1] = x0[0], P0s[0]
        for i in range(n0 - 1, n - 1):
            if not fns["point_gains"](T, idx, i, XF, PF, XP, PP, LL, GTS, M, B, W, Y, Z, v1):
                raise np.linalg.LinAlgError("predicted covariance not positive definite")
        for k in range(nk):
            MS[edges[k + 1] - 1], PS[edges[k + 1] - 1] = ms_b[k + 1], Ps_b[k + 1]
            fns["seg_smooth"](edges[k], edges[k + 1], XF, PF, XP, PP, GTS, MS, PS, M, W, v1)
        MS[n0 - 1], PS[n0 - 1] = ms_b[0], Ps_b[0]
        ACC = np.zeros((len(self.steps), 2, 4, 2, 2))
        loc = np.zeros(24)
        rs, ucur = 0.0, -1
        Y, Z, Lw = np.empty((d, d)), np.empty((d, d)), np.empty((d, d))
        for i in range(n0, n):
            if idx[i] != ucur:
                if ucur >= 0:
                    fns["flush"](ACC, ucur, loc)
                loc[:] = 0.0
                ucur = idx[i]
            rs += fns["point_grad"](y, R, jit2, i, MS, PS, XP, PP, LL, GTS, Y, Z, M, v1, v2, Lw, loc)
        fns["flush"](ACC, ucur, loc)
        P0b, rh = head_and_prior(fns, T, idx, y, R, jit2, P0, n0, ms_b[0].copy(), Ps_b[0].copy(), ACC)
        if P0b is None or not np.isfinite(rs):
            raise np.linalg.LinAlgError("predicted covariance not positive definite")
        g = gradient.contract(lx, self.steps, starts, ACC[:, 0], ACC[:, 1], P0b, rs + rh, model.jitter)
        return ll, g

"""The log likelihood of the rotation model and its exact gradient in the
log hyperparameters (internal; data in ppt), for gradient fits
(fitting.fit_rotation with method="gradient").

Reverse mode (adjoint): one forward pass of the Kalman filter stores the
predicted states, one backward pass accumulates the adjoints of the
transitions A and Q per entry of the step table (block-diagonal parts
only), of the prior P0 and of the white-noise variances R. The derivatives
of the entries in the hyperparameters come from the closed form of the SHO
transitions (statespace._block_transition) by complex step, exact to
rounding. Together this costs about 2.3 evaluations of the likelihood.
Long series keep checkpoints instead of all states: the forward pass runs a
second time per segment (about 3.3 evaluations), in bounded memory.

The log likelihood is bit-identical to that of the sequential filter with
the step table (statespace.kalman_loglik_table).
"""

import math

import numba
import numpy as np

from transitleastsquares.gp.statespace import _block_sizes, _predict, rotation_model, step_table, step_transitions

ROT_KEYS = ("sigma", "period", "Q0", "dQ", "f", "sigma_slow", "rho_slow", "jitter")
STORE_MAX = 1 << 25  # doubles of stored states (256 MB); longer series use checkpoints
SEGMENT = 1 << 16  # points per checkpointed segment
_H = 1e-30  # complex step


@numba.njit(cache=True, nogil=True)
def _forward(idx, At, Qt, y, R, kinds, starts, size, x, P, i0, i1, n_diffuse, store, xp, Pp, inn, S, ll):
    """The filter over the points i0..i1-1 from the filtered state (x, P) of
    point i0-1 (i0 = 0: the prior), as kalman_loglik_table (in place). store:
    the predicted state, innovation and its variance at buffer index i - i0.
    Returns ll plus the log likelihood of the points (one running sum, as the
    sequential filter, also over segments)."""
    nb, d = len(kinds), P.shape[0]
    M = np.empty((d, d))
    PH = np.empty(d)
    for i in range(i0, i1):
        if i > 0:
            u = idx[i]
            _predict(kinds, starts, size, At[u], Qt[u], P, M, x)
        s = R[i]
        hx = 0.0
        for r in range(d):
            acc = 0.0
            for k in range(nb):
                acc += P[r, starts[k]]
            PH[r] = acc
        for k in range(nb):
            s += PH[starts[k]]
            hx += x[starts[k]]
        innov = y[i] - hx
        if store:
            j = i - i0
            for r in range(d):
                xp[j, r] = x[r]
                for c in range(d):
                    Pp[j, r, c] = P[r, c]
            inn[j] = innov
            S[j] = s
        if i >= n_diffuse:
            ll -= 0.5 * (math.log(2 * math.pi * s) + innov * innov / s)
        for r in range(d):
            x[r] += PH[r] * innov / s
        for r in range(d):
            for c in range(r, d):
                v = P[r, c] - PH[r] * PH[c] / s
                P[r, c] = v
                P[c, r] = v
    return ll


@numba.njit(cache=True, nogil=True)
def _backward(idx, At, starts, size, i0, i1, n_diffuse, xp, Pp, inn, S, xf0, Pf0, xb, Pb, Ab, Qb, P0b):
    """The adjoints over the points i1-1 ... i0 (buffers from _forward with
    store, index i - i0). xb, Pb: adjoints of the filtered state of point
    i1-1, on return those of point i0-1; (xf0, Pf0): the filtered state of
    point i0-1 (unused for i0 = 0). Adds to Ab, Qb (per entry and block) and
    sets P0b at point 0. Returns the sum of the adjoints of R."""
    nb, d = len(starts), Pb.shape[0]
    PH = np.empty(d)
    PHb = np.empty(d)
    xf = np.empty(d)
    Pf = np.empty((d, d))
    T = np.empty((d, d))
    tmp = np.empty(d)
    rsum = 0.0
    for i in range(i1 - 1, i0 - 1, -1):
        j = i - i0
        P = Pp[j]
        s, innov = S[j], inn[j]
        for r in range(d):
            acc = 0.0
            for k in range(nb):
                acc += P[r, starts[k]]
            PH[r] = acc
        # update: x_f = x_p + PH innov / s, P_f = P_p - PH PH' / s, ll -= (log s + innov^2 / s) / 2
        sb, ib = 0.0, 0.0
        if i >= n_diffuse:
            sb -= 0.5 * (1.0 / s - innov * innov / (s * s))
            ib -= innov / s
        xbPH = 0.0
        for r in range(d):
            xbPH += xb[r] * PH[r]
        ib += xbPH / s
        sb -= xbPH * innov / (s * s)
        quad = 0.0
        for r in range(d):  # (Pb is symmetric: only its symmetric part acts on the symmetric P)
            acc = 0.0
            for c in range(d):
                acc += Pb[r, c] * PH[c]
            quad += PH[r] * acc
            PHb[r] = xb[r] * innov / s - 2 * acc / s
        sb += quad / (s * s)
        rsum += sb
        # s = R + sum_k PH[starts[k]], innov = y - sum_k x_p[starts[k]], PH = P_p H'
        for k in range(nb):
            PHb[starts[k]] += sb
            xb[starts[k]] -= ib
        for r in range(d):
            for k in range(nb):
                h = 0.5 * PHb[r]
                Pb[r, starts[k]] += h
                Pb[starts[k], r] += h
        if i == 0:
            for r in range(d):
                for c in range(d):
                    P0b[r, c] = Pb[r, c]
            break
        # predict: x_p = A x_f, P_p = A P_f A' + Q, with the filtered state of point i-1
        if i == i0:
            for r in range(d):
                xf[r] = xf0[r]
                for c in range(d):
                    Pf[r, c] = Pf0[r, c]
        else:
            Pq, sq, iq = Pp[j - 1], S[j - 1], inn[j - 1]
            for r in range(d):
                acc = 0.0
                for k in range(nb):
                    acc += Pq[r, starts[k]]
                tmp[r] = acc
            for r in range(d):
                xf[r] = xp[j - 1, r] + tmp[r] * iq / sq
                for c in range(d):
                    Pf[r, c] = Pq[r, c] - tmp[r] * tmp[c] / sq
        u = idx[i]
        A = At[u]
        for k in range(nb):  # T = Pb A
            s0 = starts[k]
            if size[k] == 2:
                for r in range(d):
                    p0, p1 = Pb[r, s0], Pb[r, s0 + 1]
                    T[r, s0] = p0 * A[k, 0, 0] + p1 * A[k, 1, 0]
                    T[r, s0 + 1] = p0 * A[k, 0, 1] + p1 * A[k, 1, 1]
            else:
                for r in range(d):
                    T[r, s0] = Pb[r, s0] * A[k, 0, 0]
        for k in range(nb):  # Ab_k += [2 Pb A P_f + xb x_f']_kk, Qb_k += Pb_kk
            s0 = starts[k]
            for a in range(size[k]):
                for b in range(size[k]):
                    acc = 0.0
                    for c in range(d):
                        acc += T[s0 + a, c] * Pf[s0 + b, c]
                    Ab[u, k, a, b] += 2 * acc + xb[s0 + a] * xf[s0 + b]
                    Qb[u, k, a, b] += Pb[s0 + a, s0 + b]
        for k in range(nb):  # xb = A' xb, Pb = A' T
            s0 = starts[k]
            if size[k] == 2:
                x0, x1 = xb[s0], xb[s0 + 1]
                xb[s0] = A[k, 0, 0] * x0 + A[k, 1, 0] * x1
                xb[s0 + 1] = A[k, 0, 1] * x0 + A[k, 1, 1] * x1
                for c in range(d):
                    t0, t1 = T[s0, c], T[s0 + 1, c]
                    Pb[s0, c] = A[k, 0, 0] * t0 + A[k, 1, 0] * t1
                    Pb[s0 + 1, c] = A[k, 0, 1] * t0 + A[k, 1, 1] * t1
            else:
                xb[s0] = A[k, 0, 0] * xb[s0]
                for c in range(d):
                    Pb[s0, c] = A[k, 0, 0] * T[s0, c]
    return rsum


@numba.njit(cache=True, nogil=True)
def loglik_adjoint(idx, At, Qt, y, R, kinds, starts, P0, n_diffuse, seg, xp, Pp, inn, S, xc, Pc, Ab, Qb, P0b):
    """Log likelihood, the adjoints of the entries' A and Q (Ab, Qb, filled),
    of P0 (P0b) and the sum of the adjoints of R. seg: points per segment
    (the buffers xp, Pp, inn, S hold one segment); xc, Pc: checkpoints
    (filtered state before each segment)."""
    n, d = len(y), P0.shape[0]
    size = _block_sizes(kinds)
    nseg = (n + seg - 1) // seg
    x = np.zeros(d)
    P = P0.copy()
    ll = 0.0
    for k in range(nseg):
        a, b = k * seg, min(n, (k + 1) * seg)
        xc[k] = x
        Pc[k] = P
        ll = _forward(idx, At, Qt, y, R, kinds, starts, size, x, P, a, b, n_diffuse, nseg == 1, xp, Pp, inn, S, ll)
    Ab[:] = 0.0
    Qb[:] = 0.0
    xb = np.zeros(d)
    Pb = np.zeros((d, d))
    rsum = 0.0
    for k in range(nseg - 1, -1, -1):
        a, b = k * seg, min(n, (k + 1) * seg)
        if nseg > 1:  # the segment's states again, from its checkpoint
            x[:] = xc[k]
            P[:] = Pc[k]
            _forward(idx, At, Qt, y, R, kinds, starts, size, x, P, a, b, n_diffuse, True, xp, Pp, inn, S, 0.0)
        rsum += _backward(idx, At, starts, size, a, b, n_diffuse, xp, Pp, inn, S, xc[k], Pc[k], xb, Pb, Ab, Qb, P0b)
    return ll, rsum


def _rotation_blocks(lx):
    """(sigma, w0, Q) of the three SHO blocks of rotation_model, from the log
    hyperparameters lx (real or complex)."""
    s, period, Q0, dQ, f, ss, rho, _ = np.exp(lx)
    return [(np.sqrt(s * s / (1 + f)), 2 * np.pi / period, 0.5 + Q0 + dQ),
            (np.sqrt(s * s * f / (1 + f)), 4 * np.pi / period, 0.5 + Q0),
            (ss, 2 * np.pi / rho, 1 / np.sqrt(2) + 0 * ss)]


def _sho_complex(sig, w0, Qf, dt):
    """A, Q (..., steps, 2, 2) and the prior diagonal of an SHO, as _block_transition (the same branches,
    chosen on the real parts), in complex arithmetic; the parameters broadcast against dt."""
    a = w0 / (2 * Qf)
    disc = w0 * w0 - a * a + 0 * dt
    lim = 1e-14 * (w0 * w0 + 0 * dt).real
    under, over = disc.real > lim, disc.real < -lim
    bb_u = np.sqrt(np.where(under, disc, 1.0))
    bb_o = np.sqrt(np.where(over, -disc, 1.0))
    c = np.where(under, np.cos(bb_u * dt), np.where(over, np.cosh(bb_o * dt), 1.0))
    sb = np.where(under, np.sin(bb_u * dt) / bb_u, np.where(over, np.sinh(bb_o * dt) / bb_o, dt))
    ex = np.exp(-a * dt)
    A = np.empty(disc.shape + (2, 2), complex)
    A[..., 0, 0] = ex * (c + a * sb)
    A[..., 0, 1] = ex * sb
    A[..., 1, 0] = -ex * w0 * w0 * sb
    A[..., 1, 1] = ex * (c - a * sb)
    p00, p11 = sig * sig, (w0 * sig) ** 2
    Q = np.empty_like(A)
    Q[..., 0, 0] = p00 - (A[..., 0, 0] ** 2 * p00 + A[..., 0, 1] ** 2 * p11)
    Q[..., 0, 1] = -(A[..., 0, 0] * A[..., 1, 0] * p00 + A[..., 0, 1] * A[..., 1, 1] * p11)
    Q[..., 1, 0] = Q[..., 0, 1]
    Q[..., 1, 1] = p11 - (A[..., 1, 0] ** 2 * p00 + A[..., 1, 1] ** 2 * p11)
    return A, Q, (p00, p11)


def derivative_tables(lx, steps, starts):
    """The derivatives of the entries' A and Q and of the prior diagonal of each SHO block in the log
    hyperparameters (all but the jitter), by complex step: [(rows j, block b, s0, dA, dQ, dp00, dp11)]."""
    n = len(lx)
    lz = np.tile(np.asarray(lx, float).astype(complex), (n - 1, 1))
    lz[np.arange(n - 1), np.arange(n - 1)] += 1j * _H  # row j: the step in lx[j]; the jitter only enters R
    dt = np.asarray(steps, float).astype(complex)
    out = []
    for k, (sig, w0, Qf) in enumerate(_rotation_blocks(lz.T)):
        on = (sig.imag != 0) | (w0.imag != 0) | (np.imag(Qf) != 0)
        if not on.any():
            continue
        A, Q, (p00, p11) = _sho_complex(sig[on, None], w0[on, None], Qf[on, None] if np.ndim(Qf) else Qf, dt[None, :])
        out.append((np.flatnonzero(on), k + 1, starts[k + 1], A.imag / _H, Q.imag / _H, p00.imag[:, 0] / _H, p11.imag[:, 0] / _H))
    return out


def contract(lx, steps, starts, Ab, Qb, P0b, rsum, jitter, tables=None):
    """The gradient in lx from the adjoints of the entries' A and Q (Ab, Qb: entries, blocks, 2, 2), of P0
    (P0b) and the sum of those of R (rsum); tables: derivative_tables (computed if not given)."""
    g = np.zeros(len(lx))
    for j, b, s0, dA, dQ, d00, d11 in (derivative_tables(lx, steps, starts) if tables is None else tables):
        g[j] += (np.einsum("uab,juab->j", Ab[:, b], dA) + np.einsum("uab,juab->j", Qb[:, b], dQ) + P0b[s0, s0] * d00
                 + P0b[s0 + 1, s0 + 1] * d11)
    g[-1] = rsum * 2 * jitter**2
    return g


class RotationGradient:
    """f(lx) -> (log likelihood, gradient) of the rotation model for fixed data
    t, y, yerr (ppt); lx: the log of the hyperparameters in the order of
    ROT_KEYS. The transitions are pooled within statespace.DT_TOL (step_table)."""

    def __init__(self, t, y, yerr):
        self.idx, self.steps = step_table(np.asarray(t, float))
        self.y, self.e2 = np.asarray(y, float), np.asarray(yerr, float) ** 2
        n, d = len(self.y), 7
        self.seg = n if n * (d * d + d + 2) <= STORE_MAX else SEGMENT
        m, nseg = min(n, self.seg), (n + self.seg - 1) // self.seg
        self.buf = (np.empty((m, d)), np.empty((m, d, d)), np.empty(m), np.empty(m), np.empty((nseg, d)),
                    np.empty((nseg, d, d)))

    def __call__(self, lx):
        lx = np.asarray(lx, float)
        model = rotation_model(dict(zip(ROT_KEYS, np.exp(lx))))
        kinds, starts, pars, P0 = model.blocks()
        At, Qt = step_transitions(self.steps, kinds, pars)
        ne, nb = len(self.steps), len(kinds)
        Ab, Qb, P0b = np.empty((ne, nb, 2, 2)), np.empty((ne, nb, 2, 2)), np.zeros(P0.shape)
        ll, rsum = loglik_adjoint(self.idx, At, Qt, self.y, self.e2 + model.jitter**2, kinds, starts, P0,
                                  model.n_diffuse, self.seg, *self.buf, Ab, Qb, P0b)
        return ll, contract(lx, self.steps, starts, Ab, Qb, P0b, rsum, model.jitter)

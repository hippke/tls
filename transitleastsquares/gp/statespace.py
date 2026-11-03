"""State-space Gaussian processes (Kalman filter, O(N)) for the stellar baseline.

Components (all linear state-space models; data in ppt internally):
  Spline    cubic smoothing spline = integrated Wiener process (Wahba 1978),
            parametrised by its smoothing length h (days)
  SHO       stochastically driven damped oscillator (celerite's SHOTerm)
  Offset    constant mean with a diffuse prior
The compiled kernels use explicit loops: numba would take BLAS/LAPACK for
@ and np.linalg from SciPy, which is not a dependency of TLS.
"""

import math

import numba
import numpy as np

DIFFUSE = 1e6  # prior variance of diffuse states (data in ppt)


# --------------------------------------------------------------------- kernels
class Offset:
    """Constant mean, diffuse prior."""

    dim = 1
    n_diffuse = 1

    def p0(self):
        return np.array([[DIFFUSE]])

    def discretize(self, dt):
        n = len(dt)
        return np.ones((n, 1, 1)), np.zeros((n, 1, 1)), self.p0()


class Spline:
    """Cubic smoothing spline as a GP: integrated Wiener process with a
    diffuse prior on level and slope. The equivalent-kernel width is about
    h (Silverman 1984): q = sigma^2 * cadence / h^4."""

    dim = 2
    n_diffuse = 2

    def __init__(self, h, sigma, cadence):
        self.h, self.q = h, sigma**2 * cadence / h**4

    def discretize(self, dt):
        n = len(dt)
        A = np.zeros((n, 2, 2))
        A[:, 0, 0] = A[:, 1, 1] = 1
        A[:, 0, 1] = dt
        Q = np.empty((n, 2, 2))
        Q[:, 0, 0] = self.q * dt**3 / 3
        Q[:, 0, 1] = Q[:, 1, 0] = self.q * dt**2 / 2
        Q[:, 1, 1] = self.q * dt
        return A, Q, self.p0()

    def p0(self):
        return np.diag([DIFFUSE, DIFFUSE])


class SHO:
    """celerite SHOTerm with standard deviation sigma, undamped period and Q.
    Stationary: Q_d = P_inf - A P_inf A'. A = exp(F dt) in closed form (as in
    _block_transition), valid also at and near critical damping (Q = 1/2)."""

    dim = 2
    n_diffuse = 0

    def __init__(self, sigma, period, Q):
        self.sigma, self.w0, self.Q = sigma, 2 * np.pi / period, Q

    def p0(self):
        return np.diag([self.sigma**2, (self.w0 * self.sigma) ** 2])

    def discretize(self, dt):
        w0 = self.w0
        a = w0 / (2 * self.Q)
        disc = w0 * w0 - a * a
        if disc > 1e-14 * w0 * w0:
            bb = np.sqrt(disc)
            c, sb = np.cos(bb * dt), np.sin(bb * dt) / bb
        elif disc < -1e-14 * w0 * w0:
            bb = np.sqrt(-disc)
            c, sb = np.cosh(bb * dt), np.sinh(bb * dt) / bb
        else:  # critically damped limit
            c, sb = np.ones_like(dt), np.asarray(dt, float).copy()
        ex = np.exp(-a * dt)
        A = np.empty((len(dt), 2, 2))
        A[:, 0, 0] = ex * (c + a * sb)
        A[:, 0, 1] = ex * sb
        A[:, 1, 0] = -ex * w0 * w0 * sb
        A[:, 1, 1] = ex * (c - a * sb)
        Pinf = self.p0()
        Qd = Pinf[None] - A @ Pinf @ A.transpose(0, 2, 1)
        return A, Qd, Pinf


class Model:
    """Sum of state-space components plus white noise (errors and jitter)."""

    def __init__(self, components, jitter=0.0):
        self.components, self.jitter = components, jitter
        self.dim = sum(c.dim for c in components)
        self.n_diffuse = sum(c.n_diffuse for c in components)
        self.H = np.concatenate(
            [np.eye(c.dim)[0] if c.dim > 1 else np.ones(1) for c in components]
        )

    def matrices(self, t, yerr):
        dt = np.diff(t, prepend=t[0])
        n, d = len(t), self.dim
        A, Q, P0 = np.zeros((n, d, d)), np.zeros((n, d, d)), np.zeros((d, d))
        i = 0
        for c in self.components:
            a, q, p0 = c.discretize(dt)
            A[:, i : i + c.dim, i : i + c.dim] = a
            Q[:, i : i + c.dim, i : i + c.dim] = q
            P0[i : i + c.dim, i : i + c.dim] = p0
            i += c.dim
        A[0] = np.eye(d)
        Q[0] = 0
        return A, Q, P0, yerr**2 + self.jitter**2

    def blocks(self):
        """Block description for the fused likelihood kernel."""
        nb = len(self.components)
        kinds = np.empty(nb, np.int64)
        starts = np.empty(nb, np.int64)
        pars = np.zeros((nb, 3))
        P0 = np.zeros((self.dim, self.dim))
        i = 0
        for k, c in enumerate(self.components):
            starts[k] = i
            if isinstance(c, Offset):
                kinds[k] = 0
            elif isinstance(c, Spline):
                kinds[k], pars[k, 0] = 1, c.q
            elif isinstance(c, SHO):
                kinds[k] = 2
                pars[k] = c.sigma, c.w0, c.Q
            else:
                raise TypeError(f"no block kernel for {type(c).__name__}")
            P0[i : i + c.dim, i : i + c.dim] = c.p0()
            i += c.dim
        return kinds, starts, pars, P0


# --------------------------------------------------------------- Kalman core
@numba.njit(cache=True)
def kalman_gains(A, Q, H, R, P0, store_pred=True, store_filt=True):
    """Data-independent pass: gains K, innovation variances S, and the
    predicted / filtered covariances (for the smoother). store_pred,
    store_filt False: placeholders of length 1 instead (each saves an
    N x d x d array; the search needs neither, the smoother only Pfilt).
    Explicit loops: no BLAS calls, which numba would take from SciPy."""
    n, d = A.shape[0], A.shape[1]
    K = np.empty((n, d))
    S = np.empty(n)
    Ppred = np.empty((n if store_pred else 1, d, d))
    Pfilt = np.empty((n if store_filt else 1, d, d))
    P = P0.copy()
    M = np.empty((d, d))
    PH = np.empty(d)
    for t in range(n):
        if t > 0:
            for r in range(d):  # M = A P
                for c in range(d):
                    acc = 0.0
                    for l in range(d):
                        acc += A[t, r, l] * P[l, c]
                    M[r, c] = acc
            for r in range(d):  # P = M A' + Q
                for c in range(d):
                    acc = Q[t, r, c]
                    for l in range(d):
                        acc += M[r, l] * A[t, c, l]
                    P[r, c] = acc
        if store_pred:
            Ppred[t] = P
        s = R[t]
        for r in range(d):
            acc = 0.0
            for c in range(d):
                acc += P[r, c] * H[c]
            PH[r] = acc
            s += H[r] * acc
        S[t] = s
        for r in range(d):
            K[t, r] = PH[r] / s
        for r in range(d):  # P = P - K PH', symmetrised
            for c in range(r, d):
                v = 0.5 * ((P[r, c] - K[t, r] * PH[c]) + (P[c, r] - K[t, c] * PH[r]))
                P[r, c] = v
                P[c, r] = v
        if store_filt:
            Pfilt[t] = P
    return K, S, Ppred, Pfilt


@numba.njit(cache=True)
def whiten(y, A, K, S, H):
    """Normalised innovations e = L^-1 y, plus predicted and filtered means."""
    n, d = A.shape[0], A.shape[1]
    e = np.empty(n)
    xpred = np.empty((n, d))
    xfilt = np.empty((n, d))
    x = np.zeros(d)
    xp = np.zeros(d)
    for t in range(n):
        if t > 0:
            for r in range(d):
                acc = 0.0
                for c in range(d):
                    acc += A[t, r, c] * x[c]
                xp[r] = acc
        else:
            xp[:] = x
        hx = 0.0
        for r in range(d):
            hx += H[r] * xp[r]
        xpred[t] = xp
        innov = y[t] - hx
        e[t] = innov / math.sqrt(S[t])
        for r in range(d):
            x[r] = xp[r] + K[t, r] * innov
        xfilt[t] = x
    return e, xpred, xfilt


@numba.njit(cache=True)
def _chol_solve(Pm, B, L, X):
    """Solve Pm X = B for symmetric positive definite Pm (d x d, small) by
    Cholesky; L is workspace. No LAPACK (which numba would take from SciPy).
    If Pm is singular (e.g. a component with zero variance), a ridge of
    1e-12 * max(diag) is added: the solution then sets the undetermined
    components to ~0, which is irrelevant for the smoother (their covariance
    is zero). Positive definite Pm are solved unchanged."""
    d = Pm.shape[0]
    ridge = 0.0
    for attempt in range(2):
        good = True
        for i in range(d):
            for j in range(i + 1):
                acc = Pm[i, j] + (ridge if i == j else 0.0)
                for k in range(j):
                    acc -= L[i, k] * L[j, k]
                if i == j:
                    if not acc > 0.0:
                        good = False
                        break
                    L[i, j] = math.sqrt(acc)
                else:
                    L[i, j] = acc / L[j, j]
            if not good:
                break
        if good:
            break
        dmax = 0.0
        for i in range(d):
            dmax = max(dmax, abs(Pm[i, i]))
        ridge = 1e-12 * dmax + 1e-300
    for c in range(B.shape[1]):
        for i in range(d):  # forward: L z = b
            acc = B[i, c]
            for k in range(i):
                acc -= L[i, k] * X[k, c]
            X[i, c] = acc / L[i, i]
        for i in range(d - 1, -1, -1):  # backward: L' x = z
            acc = X[i, c]
            for k in range(i + 1, d):
                acc -= L[k, i] * X[k, c]
            X[i, c] = acc / L[i, i]


@numba.njit(cache=True)
def rts_smooth(A, Q, xpred, xfilt, Pfilt):
    """Rauch-Tung-Striebel smoother: posterior mean of the state.
    G = Pf A' Pp^-1, so G' solves Pp G' = A Pf (Pp, Pf symmetric). The
    predicted covariance Pp_{t+1} = A Pf_t A' + Q is recomputed here in the
    order of kalman_gains (bit-identical) instead of being stored."""
    n, d = A.shape[0], A.shape[1]
    xs = xfilt.copy()
    B = np.empty((d, d))
    Pp = np.empty((d, d))
    GT = np.empty((d, d))
    L = np.zeros((d, d))
    for t in range(n - 2, -1, -1):
        for r in range(d):  # B = A_{t+1} Pf_t
            for c in range(d):
                acc = 0.0
                for l in range(d):
                    acc += A[t + 1, r, l] * Pfilt[t, l, c]
                B[r, c] = acc
        for r in range(d):  # Pp = B A' + Q
            for c in range(d):
                acc = Q[t + 1, r, c]
                for l in range(d):
                    acc += B[r, l] * A[t + 1, c, l]
                Pp[r, c] = acc
        _chol_solve(Pp, B, L, GT)
        for r in range(d):  # xs = xf + G (xs_{t+1} - xp_{t+1})
            acc = 0.0
            for l in range(d):
                acc += GT[l, r] * (xs[t + 1, l] - xpred[t + 1, l])
            xs[t, r] = xfilt[t, r] + acc
    return xs


# ------------------------------------------------- fused log likelihood
@numba.njit(cache=True, nogil=True)
def _block_transition(kind, par, dt, A2, Q2):
    """Discrete transition A and process noise Q of one component (1x1 or
    2x2, in the top-left of A2/Q2). kind 0 offset, 1 spline (q), 2 SHO
    (sigma, w0, Q) with the closed-form exp(F dt) instead of eig."""
    if kind == 0:
        A2[0, 0] = 1.0
        Q2[0, 0] = 0.0
    elif kind == 1:
        q = par[0]
        A2[0, 0] = 1.0
        A2[0, 1] = dt
        A2[1, 0] = 0.0
        A2[1, 1] = 1.0
        Q2[0, 0] = q * dt**3 / 3
        Q2[0, 1] = Q2[1, 0] = q * dt**2 / 2
        Q2[1, 1] = q * dt
    else:
        sig, w0, Qf = par[0], par[1], par[2]
        a = w0 / (2 * Qf)
        disc = w0 * w0 - a * a
        if disc > 1e-14 * w0 * w0:
            bb = math.sqrt(disc)
            c = math.cos(bb * dt)
            sb = math.sin(bb * dt) / bb
        elif disc < -1e-14 * w0 * w0:
            bb = math.sqrt(-disc)
            c = math.cosh(bb * dt)
            sb = math.sinh(bb * dt) / bb
        else:  # critically damped limit
            c = 1.0
            sb = dt
        ex = math.exp(-a * dt)
        A2[0, 0] = ex * (c + a * sb)
        A2[0, 1] = ex * sb
        A2[1, 0] = -ex * w0 * w0 * sb
        A2[1, 1] = ex * (c - a * sb)
        p00, p11 = sig * sig, (w0 * sig) ** 2  # stationary covariance
        Q2[0, 0] = p00 - (A2[0, 0] ** 2 * p00 + A2[0, 1] ** 2 * p11)
        Q2[0, 1] = -(A2[0, 0] * A2[1, 0] * p00 + A2[0, 1] * A2[1, 1] * p11)
        Q2[1, 0] = Q2[0, 1]
        Q2[1, 1] = p11 - (A2[1, 0] ** 2 * p00 + A2[1, 1] ** 2 * p11)


@numba.njit(cache=True, nogil=True)
def kalman_loglik(t, y, R, kinds, starts, pars, P0, n_diffuse):
    """Log likelihood of a block-diagonal state-space GP, allocation-free.
    The prediction A P A' uses the block structure of A (blocks of 1 or 2)."""
    nb, d, n = len(kinds), P0.shape[0], len(t)
    size = np.empty(nb, np.int64)
    for k in range(nb):
        size[k] = 1 if kinds[k] == 0 else 2
    Ab = np.zeros((nb, 2, 2))
    Qb = np.zeros((nb, 2, 2))
    P = P0.copy()
    M = np.empty((d, d))
    PH = np.empty(d)
    x = np.zeros(d)
    xp = np.zeros(d)
    ll = 0.0
    dt_last = -1.0
    for i in range(n):
        dt = t[i] - t[i - 1] if i > 0 else 0.0
        if i > 0:
            if dt != dt_last:  # regular cadence: most steps reuse A, Q
                for k in range(nb):
                    _block_transition(kinds[k], pars[k], dt, Ab[k], Qb[k])
                dt_last = dt
            for k in range(nb):  # M = A P (rows of block k)
                s0, sz = starts[k], size[k]
                for r in range(sz):
                    for c in range(d):
                        acc = 0.0
                        for l in range(sz):
                            acc += Ab[k, r, l] * P[s0 + l, c]
                        M[s0 + r, c] = acc
            for k in range(nb):  # P = M A' + Q (columns of block k)
                s0, sz = starts[k], size[k]
                for r in range(d):
                    for c in range(sz):
                        acc = 0.0
                        for l in range(sz):
                            acc += M[r, s0 + l] * Ab[k, c, l]
                        P[r, s0 + c] = acc
                for r in range(sz):
                    for c in range(sz):
                        P[s0 + r, s0 + c] += Qb[k, r, c]
            for k in range(nb):  # x = A x
                s0, sz = starts[k], size[k]
                for r in range(sz):
                    acc = 0.0
                    for l in range(sz):
                        acc += Ab[k, r, l] * x[s0 + l]
                    xp[s0 + r] = acc
            for r in range(d):
                x[r] = xp[r]
        # update; H picks the first state of every block
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


def log_likelihood(model, t, y, yerr):
    kinds, starts, pars, P0 = model.blocks()
    R = yerr**2 + model.jitter**2
    return kalman_loglik(np.asarray(t, float), np.asarray(y, float), R, kinds, starts, pars, P0, model.n_diffuse)


def gp_mean(model, t, y, yerr):
    """Posterior mean of the baseline at t (RTS smoother)."""
    A, Q, P0, R = model.matrices(t, yerr)
    K, S, _, Pfilt = kalman_gains(A, Q, model.H, R, P0, False, True)
    _, xpred, xfilt = whiten(y, A, K, S, model.H)
    xs = rts_smooth(A, Q, xpred, xfilt, Pfilt)
    return xs @ model.H


# --------------------------------------------------------- model builders
def spline_model(h, sigma, cadence, jitter=0.0):
    return Model([Offset(), Spline(h, sigma, cadence)], jitter)


def rotation_model(p):
    """p: dict with sigma, period, Q0, dQ, f, sigma_slow, rho_slow, jitter."""
    s2 = p["sigma"] ** 2
    comps = [
        Offset(),
        SHO(math.sqrt(s2 / (1 + p["f"])), p["period"], 0.5 + p["Q0"] + p["dQ"]),
        SHO(math.sqrt(s2 * p["f"] / (1 + p["f"])), 0.5 * p["period"], 0.5 + p["Q0"]),
        SHO(p["sigma_slow"], p["rho_slow"], 1 / math.sqrt(2)),
    ]
    return Model(comps, p["jitter"])

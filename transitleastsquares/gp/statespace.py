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
    """Data-independent pass on dense per-point matrices (Model.matrices):
    gains K, innovation variances S, and the predicted / filtered
    covariances. store_pred, store_filt False: placeholders of length 1
    instead. The search and the smoother use the block-structured kernels
    (kalman_whiten, smoothed_mean); this generic filter is the reference
    of the tests. Explicit loops: no BLAS calls, which numba would take
    from SciPy."""
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
def _predict(kinds, starts, size, Ab, Qb, P, M, x, xp):
    """P = A P A' + Q and x = A x for block-diagonal A (blocks of 1 or 2)."""
    nb, d = len(kinds), P.shape[0]
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


@numba.njit(cache=True, nogil=True)
def _block_sizes(kinds):
    size = np.empty(len(kinds), np.int64)
    for k in range(len(kinds)):
        size[k] = 1 if kinds[k] == 0 else 2
    return size


@numba.njit(cache=True, nogil=True)
def _filter_segment(t, y, R, kinds, starts, pars, x, P, i0, i1, n_diffuse):
    """Kalman filter over the points i0..i1-1 from the filtered state (x, P)
    of point i0-1 (for i0 = 0: the prior; no prediction step). Updates x
    and P in place; returns the log likelihood of the segment."""
    nb, d = len(kinds), P.shape[0]
    size = _block_sizes(kinds)
    Ab = np.zeros((nb, 2, 2))
    Qb = np.zeros((nb, 2, 2))
    M = np.empty((d, d))
    PH = np.empty(d)
    xp = np.zeros(d)
    ll = 0.0
    dt_last = -1.0
    for i in range(i0, i1):
        dt = t[i] - t[i - 1] if i > 0 else 0.0
        if i > 0:
            if dt != dt_last:  # regular cadence: most steps reuse A, Q
                for k in range(nb):
                    _block_transition(kinds[k], pars[k], dt, Ab[k], Qb[k])
                dt_last = dt
            _predict(kinds, starts, size, Ab, Qb, P, M, x, xp)
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


@numba.njit(cache=True, nogil=True)
def kalman_loglik(t, y, R, kinds, starts, pars, P0, n_diffuse):
    """Log likelihood of a block-diagonal state-space GP, allocation-free.
    The prediction A P A' uses the block structure of A (blocks of 1 or 2)."""
    return _filter_segment(t, y, R, kinds, starts, pars, np.zeros(P0.shape[0]), P0.copy(), 0, len(t), n_diffuse)


@numba.njit(cache=True, nogil=True)
def _segment_element(t, y, R, kinds, starts, pars, i0, i1, F, c, P, J, eta):
    """The Kalman filter over the points i0..i1-1 (i0 > 0) conditioned on an
    exactly known filtered state s at point i0-1: the filtered state at
    i1-1 is N(F s + c, P), and the log likelihood of the segment is
    -const/2 - s'J s/2 + eta's. Fills F, c, P, J, eta; returns const.
    (The elements of the parallel-in-time Kalman filter of Sarkka &
    Garcia-Fernandez 2021, for one segment instead of one point.)"""
    nb, d = len(kinds), P.shape[0]
    size = _block_sizes(kinds)
    Ab = np.zeros((nb, 2, 2))
    Qb = np.zeros((nb, 2, 2))
    M = np.empty((d, d))
    PH = np.empty(d)
    hF = np.empty(d)
    xp = np.zeros(d)
    P[:] = 0.0
    F[:] = 0.0
    for r in range(d):
        F[r, r] = 1.0
    c[:] = 0.0
    J[:] = 0.0
    eta[:] = 0.0
    const = 0.0
    dt_last = -1.0
    for i in range(i0, i1):
        dt = t[i] - t[i - 1]
        if dt != dt_last:
            for k in range(nb):
                _block_transition(kinds[k], pars[k], dt, Ab[k], Qb[k])
            dt_last = dt
        _predict(kinds, starts, size, Ab, Qb, P, M, c, xp)
        for col in range(d):  # F = A F, column by column
            for r in range(d):
                M[r, 0] = F[r, col]
            for k in range(nb):
                s0, sz = starts[k], size[k]
                for r in range(sz):
                    acc = 0.0
                    for l in range(sz):
                        acc += Ab[k, r, l] * M[s0 + l, 0]
                    F[s0 + r, col] = acc
        s = R[i]
        v0 = y[i]
        for r in range(d):
            acc = 0.0
            for k in range(nb):
                acc += P[r, starts[k]]
            PH[r] = acc
        for col in range(d):
            acc = 0.0
            for k in range(nb):
                acc += F[starts[k], col]
            hF[col] = acc
        for k in range(nb):
            s += PH[starts[k]]
            v0 -= c[starts[k]]
        const += math.log(2 * math.pi * s) + v0 * v0 / s
        for r in range(d):
            u = hF[r] / s
            eta[r] += u * v0
            for col in range(r, d):  # upper triangle; mirrored after the loop
                J[r, col] += u * hF[col]
        for r in range(d):
            k_r = PH[r] / s
            c[r] += k_r * v0
            for col in range(d):
                F[r, col] -= k_r * hF[col]
        for r in range(d):
            for col in range(r, d):
                v = P[r, col] - PH[r] * PH[col] / s
                P[r, col] = v
                P[col, r] = v
    for r in range(d):
        for col in range(r + 1, d):
            J[col, r] = J[r, col]
    return const


@numba.njit(cache=True, nogil=True)
def _lu_solve(Am, B, X):
    """Solve Am X = B (d x d, Gaussian elimination with partial pivoting);
    returns log|det Am|. Am and B are overwritten."""
    d = Am.shape[0]
    logdet = 0.0
    for k in range(d):
        p = k
        for i in range(k + 1, d):
            if abs(Am[i, k]) > abs(Am[p, k]):
                p = i
        if p != k:
            for j in range(d):
                Am[k, j], Am[p, j] = Am[p, j], Am[k, j]
                B[k, j], B[p, j] = B[p, j], B[k, j]
        logdet += math.log(abs(Am[k, k]))
        for i in range(k + 1, d):
            f = Am[i, k] / Am[k, k]
            for j in range(k, d):
                Am[i, j] -= f * Am[k, j]
            for j in range(B.shape[1]):
                B[i, j] -= f * B[k, j]
    for j in range(B.shape[1]):
        for i in range(d - 1, -1, -1):
            acc = B[i, j]
            for k in range(i + 1, d):
                acc -= Am[i, k] * X[k, j]
            X[i, j] = acc / Am[i, i]
    return logdet


@numba.njit(cache=True, nogil=True)
def _combine(x, P, F, c, Pe, J, eta, const):
    """Condition the filtered state N(x, P) at the start of a segment on the
    segment's element (F, c, Pe, J, eta, const) and propagate it to the
    segment's end (in place); returns the segment's log likelihood given
    the data before it."""
    d = len(x)
    Am = np.eye(d)
    for i in range(d):  # Am = I + P J
        for j in range(d):
            acc = 0.0
            for l in range(d):
                acc += P[i, l] * J[l, j]
            Am[i, j] += acc
    B = P.copy()
    Pc = np.empty((d, d))
    logdet = _lu_solve(Am, B, Pc)  # Pc = (I + P J)^-1 P = (P^-1 + J)^-1
    g = eta.copy()
    quad = 0.0
    for i in range(d):
        acc = 0.0
        for j in range(d):
            acc += J[i, j] * x[j]
        g[i] -= acc
        quad += -0.5 * x[i] * acc + eta[i] * x[i]
    xc = x.copy()
    for i in range(d):
        acc = 0.0
        for j in range(d):
            acc += Pc[i, j] * g[j]
        quad += 0.5 * g[i] * acc
        xc[i] += acc
    for i in range(d):  # x = F xc + c, P = F Pc F' + Pe (symmetrised)
        acc = c[i]
        for j in range(d):
            acc += F[i, j] * xc[j]
        x[i] = acc
    M = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            acc = 0.0
            for l in range(d):
                acc += F[i, l] * Pc[l, j]
            M[i, j] = acc
    for i in range(d):
        for j in range(d):
            acc = Pe[i, j]
            for l in range(d):
                acc += M[i, l] * F[j, l]
            P[i, j] = acc
    for i in range(d):
        for j in range(i + 1, d):
            v = 0.5 * (P[i, j] + P[j, i])
            P[i, j] = v
            P[j, i] = v
    return -0.5 * const + quad - 0.5 * logdet


@numba.njit(cache=True, parallel=True)
def kalman_loglik_parallel(t, y, R, kinds, starts, pars, P0, n_diffuse, nseg):
    """kalman_loglik with the data split into nseg segments, filtered in
    parallel (the first one from the prior, the others conditioned on their
    unknown start state) and combined in order. Exact up to rounding; the
    result depends on nseg, not on the number of threads."""
    n, d = len(t), P0.shape[0]
    edges = np.empty(nseg + 1, np.int64)
    for k in range(nseg + 1):
        edges[k] = (k * n) // nseg
    F = np.empty((nseg, d, d))
    c = np.empty((nseg, d))
    P = np.empty((nseg, d, d))
    J = np.empty((nseg, d, d))
    eta = np.empty((nseg, d))
    const = np.empty(nseg)
    for k in numba.prange(nseg):
        if k == 0:
            c[0] = 0.0
            P[0] = P0
            const[0] = _filter_segment(t, y, R, kinds, starts, pars, c[0], P[0], 0, edges[1], n_diffuse)
        else:
            const[k] = _segment_element(t, y, R, kinds, starts, pars, edges[k], edges[k + 1],
                                        F[k], c[k], P[k], J[k], eta[k])
    x = c[0].copy()
    Pf = P[0].copy()
    ll = const[0]
    for k in range(1, nseg):
        ll += _combine(x, Pf, F[k], c[k], P[k], J[k], eta[k], const[k])
    return ll


@numba.njit(cache=True, nogil=True)
def _forward_store(t, y, R, kinds, starts, pars, x, P, i0, i1, xpred, xfilt, Pfilt):
    """_filter_segment, storing the predicted and filtered means and the
    filtered covariances of the points i0..i1-1 (global indices)."""
    nb, d = len(kinds), P.shape[0]
    size = _block_sizes(kinds)
    Ab = np.zeros((nb, 2, 2))
    Qb = np.zeros((nb, 2, 2))
    M = np.empty((d, d))
    PH = np.empty(d)
    xp = np.zeros(d)
    dt_last = -1.0
    for i in range(i0, i1):
        if i > 0:
            dt = t[i] - t[i - 1]
            if dt != dt_last:
                for k in range(nb):
                    _block_transition(kinds[k], pars[k], dt, Ab[k], Qb[k])
                dt_last = dt
            _predict(kinds, starts, size, Ab, Qb, P, M, x, xp)
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
        xpred[i] = x
        innov = y[i] - hx
        for r in range(d):
            x[r] += PH[r] * innov / s
        for r in range(d):
            for c in range(r, d):
                v = P[r, c] - PH[r] * PH[c] / s
                P[r, c] = v
                P[c, r] = v
        xfilt[i] = x
        Pfilt[i] = P


@numba.njit(cache=True, nogil=True)
def _backward_gains(t, kinds, starts, pars, i0, bnd, xpred, xfilt, Pfilt, Gam, gam):
    """RTS gains G_i = Pf_i A' Pp_{i+1}^-1 for i = bnd-1 .. i0, stored in
    place of Pfilt[i] (no longer needed), and the affine map of the
    smoothed mean from the boundary point bnd to i0:
    xs_i0 = Gam xs_bnd + gam."""
    nb, d = len(kinds), Pfilt.shape[1]
    size = _block_sizes(kinds)
    Ab = np.zeros((nb, 2, 2))
    Qb = np.zeros((nb, 2, 2))
    B = np.empty((d, d))
    Pp = np.empty((d, d))
    GT = np.empty((d, d))
    L = np.zeros((d, d))
    T = np.empty((d, d))
    h = np.empty(d)
    Gam[:] = 0.0
    for r in range(d):
        Gam[r, r] = 1.0
    gam[:] = 0.0
    dt_last = -1.0
    for i in range(bnd - 1, i0 - 1, -1):
        dt = t[i + 1] - t[i]
        if dt != dt_last:
            for k in range(nb):
                _block_transition(kinds[k], pars[k], dt, Ab[k], Qb[k])
            dt_last = dt
        for k in range(nb):  # B = A Pf (rows of block k)
            s0, sz = starts[k], size[k]
            for r in range(sz):
                for c in range(d):
                    acc = 0.0
                    for l in range(sz):
                        acc += Ab[k, r, l] * Pfilt[i, s0 + l, c]
                    B[s0 + r, c] = acc
        for k in range(nb):  # Pp = B A' + Q (columns of block k)
            s0, sz = starts[k], size[k]
            for r in range(d):
                for c in range(sz):
                    acc = 0.0
                    for l in range(sz):
                        acc += B[r, s0 + l] * Ab[k, c, l]
                    Pp[r, s0 + c] = acc
            for r in range(sz):
                for c in range(sz):
                    Pp[s0 + r, s0 + c] += Qb[k, r, c]
        _chol_solve(Pp, B, L, GT)  # GT = G'
        for r in range(d):
            for c in range(d):
                Pfilt[i, r, c] = GT[c, r]
        # xs_i = G xs_{i+1} + h, h = xf_i - G xp_{i+1}; compose with the map so far
        for r in range(d):
            acc = xfilt[i, r]
            for l in range(d):
                acc -= GT[l, r] * xpred[i + 1, l]
            h[r] = acc
        for r in range(d):
            acc = h[r]
            for l in range(d):
                acc += GT[l, r] * gam[l]
            T[r, 0] = acc
            for c in range(d):
                acc = 0.0
                for l in range(d):
                    acc += GT[l, r] * Gam[l, c]
                B[r, c] = acc
        for r in range(d):
            gam[r] = T[r, 0]
            for c in range(d):
                Gam[r, c] = B[r, c]


@numba.njit(cache=True, nogil=True)
def _backward_mean(kinds, starts, i0, bnd, xs_bnd, xpred, xfilt, G, out):
    """Smoothed means xs_i = xf_i + G_i (xs_{i+1} - xp_{i+1}) for
    i = bnd-1 .. i0 from the known xs at bnd; out[i] = H xs_i."""
    nb, d = len(kinds), G.shape[1]
    xs = xs_bnd.copy()
    tmp = np.empty(d)
    for i in range(bnd - 1, i0 - 1, -1):
        for r in range(d):
            acc = xfilt[i, r]
            for l in range(d):
                acc += G[i, r, l] * (xs[l] - xpred[i + 1, l])
            tmp[r] = acc
        hx = 0.0
        for r in range(d):
            xs[r] = tmp[r]
        for k in range(nb):
            hx += xs[starts[k]]
        out[i] = hx


@numba.njit(cache=True, parallel=True)
def smoothed_mean(t, y, R, kinds, starts, pars, P0, nseg):
    """Posterior mean H xs of a block-diagonal state-space GP (RTS
    smoother), in nseg segments in parallel: (1) segment elements, joined
    in order: the filtered state before each segment; (2) forward filter per
    segment; (3) backward gains per segment and the affine map of the
    smoothed mean across it, joined backwards in order; (4) the smoothed
    means per segment. Exact up to rounding for any nseg (nseg = 1: the
    sequential smoother)."""
    n, d = len(t), P0.shape[0]
    nb = len(kinds)
    edges = np.empty(nseg + 1, np.int64)
    for k in range(nseg + 1):
        edges[k] = (k * n) // nseg
    F = np.empty((nseg, d, d))
    c = np.empty((nseg, d))
    Pe = np.empty((nseg, d, d))
    J = np.empty((nseg, d, d))
    eta = np.empty((nseg, d))
    const = np.empty(nseg)
    xpred = np.empty((n, d))
    xfilt = np.empty((n, d))
    Pfilt = np.empty((n, d, d))
    x0 = np.zeros((nseg, d))  # filtered state before each segment (k >= 1)
    P0s = np.empty((nseg, d, d))
    for k in numba.prange(nseg):
        if k == 0:  # the first segment: forward filter from the prior
            c[0] = 0.0
            Pe[0] = P0
            _forward_store(t, y, R, kinds, starts, pars, c[0], Pe[0], 0, edges[1], xpred, xfilt, Pfilt)
        else:
            const[k] = _segment_element(t, y, R, kinds, starts, pars, edges[k], edges[k + 1],
                                        F[k], c[k], Pe[k], J[k], eta[k])
    x = c[0].copy()
    P = Pe[0].copy()
    for k in range(1, nseg):
        x0[k] = x
        P0s[k] = P
        _combine(x, P, F[k], c[k], Pe[k], J[k], eta[k], const[k])
    for k in numba.prange(1, nseg):
        xk = x0[k].copy()
        Pk = P0s[k].copy()
        _forward_store(t, y, R, kinds, starts, pars, xk, Pk, edges[k], edges[k + 1], xpred, xfilt, Pfilt)
    Gam = np.empty((nseg, d, d))
    gam = np.empty((nseg, d))
    bnd = np.empty(nseg, np.int64)
    for k in range(nseg):
        bnd[k] = min(edges[k + 1], n - 1)
    for k in numba.prange(nseg):
        _backward_gains(t, kinds, starts, pars, edges[k], bnd[k], xpred, xfilt, Pfilt, Gam[k], gam[k])
    xs_b = np.empty((nseg, d))  # smoothed mean at each segment's boundary point
    v = xfilt[n - 1].copy()
    for k in range(nseg - 1, -1, -1):
        xs_b[k] = v
        w = gam[k].copy()
        for r in range(d):
            for l in range(d):
                w[r] += Gam[k, r, l] * v[l]
        v = w
    out = np.empty(n)
    hx = 0.0
    for kk in range(nb):
        hx += xfilt[n - 1, starts[kk]]
    out[n - 1] = hx
    for k in numba.prange(nseg):
        _backward_mean(kinds, starts, edges[k], bnd[k], xs_b[k], xpred, xfilt, Pfilt, out)
    return out


TASK_POINTS = 4096  # points per parallel task when filling per-point arrays


@numba.njit(cache=True, parallel=True)
def transition_matrices(t, kinds, starts, pars, d):
    """Dense per-point transitions A_i (A_0 = I) of a block-diagonal model,
    filled in parallel (for the single events and tail operators)."""
    n, nb = len(t), len(kinds)
    size = _block_sizes(kinds)
    A = np.zeros((n, d, d))
    ntask = (n + TASK_POINTS - 1) // TASK_POINTS
    for task in numba.prange(ntask):
        Ab = np.zeros((nb, 2, 2))
        Qb = np.zeros((nb, 2, 2))
        dt_last = -1.0
        for i in range(max(1, task * TASK_POINTS), min(n, (task + 1) * TASK_POINTS)):
            dt = t[i] - t[i - 1]
            if dt != dt_last:
                for k in range(nb):
                    _block_transition(kinds[k], pars[k], dt, Ab[k], Qb[k])
                dt_last = dt
            for k in range(nb):
                s0, sz = starts[k], size[k]
                for r in range(sz):
                    for c in range(sz):
                        A[i, s0 + r, s0 + c] = Ab[k, r, c]
    for r in range(d):
        A[0, r, r] = 1.0
    return A


@numba.njit(cache=True, nogil=True)
def _gains_segment(t, y, R, kinds, starts, pars, x, P, i0, i1, K, S, ey):
    """_filter_segment, storing the gains K, innovation variances S and
    normalised innovations ey of the points i0..i1-1."""
    nb, d = len(kinds), P.shape[0]
    size = _block_sizes(kinds)
    Ab = np.zeros((nb, 2, 2))
    Qb = np.zeros((nb, 2, 2))
    M = np.empty((d, d))
    PH = np.empty(d)
    xp = np.zeros(d)
    dt_last = -1.0
    for i in range(i0, i1):
        if i > 0:
            dt = t[i] - t[i - 1]
            if dt != dt_last:
                for k in range(nb):
                    _block_transition(kinds[k], pars[k], dt, Ab[k], Qb[k])
                dt_last = dt
            _predict(kinds, starts, size, Ab, Qb, P, M, x, xp)
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
        S[i] = s
        ey[i] = innov / math.sqrt(s)
        for r in range(d):
            K[i, r] = PH[r] / s
            x[r] += K[i, r] * innov
        for r in range(d):
            for c in range(r, d):
                v = P[r, c] - PH[r] * PH[c] / s
                P[r, c] = v
                P[c, r] = v


@numba.njit(cache=True, parallel=True)
def kalman_whiten(t, y, R, kinds, starts, pars, P0, nseg):
    """Gains K, innovation variances S and normalised innovations ey
    (= L^-1 y) of a block-diagonal model, in nseg segments in parallel:
    segment elements joined in order give the filtered state before each
    segment, then every segment is filtered from it. Exact up to rounding
    (nseg = 1: the sequential filter)."""
    n, d = len(t), P0.shape[0]
    edges = np.empty(nseg + 1, np.int64)
    for k in range(nseg + 1):
        edges[k] = (k * n) // nseg
    K = np.empty((n, d))
    S = np.empty(n)
    ey = np.empty(n)
    F = np.empty((nseg, d, d))
    c = np.empty((nseg, d))
    Pe = np.empty((nseg, d, d))
    J = np.empty((nseg, d, d))
    eta = np.empty((nseg, d))
    const = np.empty(nseg)
    for k in numba.prange(nseg):
        if k == 0:
            c[0] = 0.0
            Pe[0] = P0
            _gains_segment(t, y, R, kinds, starts, pars, c[0], Pe[0], 0, edges[1], K, S, ey)
        else:
            const[k] = _segment_element(t, y, R, kinds, starts, pars, edges[k], edges[k + 1],
                                        F[k], c[k], Pe[k], J[k], eta[k])
    x0 = np.zeros((nseg, d))
    P0s = np.empty((nseg, d, d))
    x = c[0].copy()
    P = Pe[0].copy()
    for k in range(1, nseg):
        x0[k] = x
        P0s[k] = P
        _combine(x, P, F[k], c[k], Pe[k], J[k], eta[k], const[k])
    for k in numba.prange(1, nseg):
        xk = x0[k].copy()
        Pk = P0s[k].copy()
        _gains_segment(t, y, R, kinds, starts, pars, xk, Pk, edges[k], edges[k + 1], K, S, ey)
    return K, S, ey


DT_TOL = 1e-5  # time steps within this (relative) share their transition (transition_table)


@numba.njit(cache=True)
def transition_table(t, kinds, pars, tol=DT_TOL):
    """The transitions per distinct time step: (index per point, blocks A and
    Q per entry, shape (entries, blocks, 2, 2)). Entry 0 is the first point
    (A = I, Q = 0). Steps that differ by the rounding of stored times share an
    entry, so an even cadence costs a few transitions instead of one per point;
    A - I changes by tol * w0 dt (far below FP32 on the short cadences)."""
    n, nb = len(t), len(kinds)
    size = _block_sizes(kinds)
    idx = np.zeros(n, np.int32)
    cap = 1024
    At = np.zeros((cap, nb, 2, 2))
    Qt = np.zeros((cap, nb, 2, 2))
    for k in range(nb):
        for r in range(size[k]):
            At[0, k, r, r] = 1.0
    u, ref = 0, -1.0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if u == 0 or abs(dt - ref) > tol * ref:
            u += 1
            if u == cap:
                cap *= 2
                At2, Qt2 = np.zeros((cap, nb, 2, 2)), np.zeros((cap, nb, 2, 2))
                At2[:u] = At[:u]
                Qt2[:u] = Qt[:u]
                At, Qt = At2, Qt2
            for k in range(nb):
                _block_transition(kinds[k], pars[k], dt, At[u, k], Qt[u, k])
            ref = dt
        idx[i] = u
    return idx, At[: u + 1], Qt[: u + 1]


@numba.njit(cache=True)
def riccati_head(t, y, R, kinds, starts, pars, P0, n0, K, isS, ey):
    """The exact filter (FP64) over the first n0 points, which hold the
    diffuse start: K, 1 / sqrt(S), ey stored; returns the filtered x, P."""
    d = P0.shape[0]
    x = np.zeros(d)
    P = P0.copy()
    K0, S0 = np.empty((n0, d)), np.empty(n0)
    _gains_segment(t, y, R, kinds, starts, pars, x, P, 0, n0, K0, S0, ey)
    for i in range(n0):
        isS[i] = 1.0 / math.sqrt(S0[i])
        for r in range(d):
            K[i, r] = K0[i, r]
    return x, P


@numba.njit(cache=True)
def riccati_keys(t, R, kinds, starts, pars, P, n0, edges, kbin, gap):
    """Filtered covariance at each t[edges[s] - 1] from P at t[n0 - 1]: the
    Riccati recursion (FP64) over bins of up to kbin points with their summed
    information (R_b = 1 / sum(1 / R), at their weighted mean time, no bin
    across a gap >= `gap`), which needs no state and costs 1 / kbin."""
    nb, d = len(kinds), P.shape[0]
    size = _block_sizes(kinds)
    Ab, Qb = np.zeros((nb, 2, 2)), np.zeros((nb, 2, 2))
    M, PH, x, xp = np.empty((d, d)), np.empty(d), np.zeros(d), np.zeros(d)
    P = P.copy()
    out = np.empty((len(edges), d, d))
    tprev, i = t[n0 - 1], n0
    for s in range(len(edges)):
        hi = edges[s]
        while i < hi:
            j, iw, wt = i, 0.0, 0.0
            while j < hi and j - i < kbin and (j == i or t[j] - t[j - 1] < gap):
                iw += 1.0 / R[j]
                wt += t[j] / R[j]
                j += 1
            tb = wt / iw
            for k in range(nb):
                _block_transition(kinds[k], pars[k], tb - tprev, Ab[k], Qb[k])
            _predict(kinds, starts, size, Ab, Qb, P, M, x, xp)
            sv = 1.0 / iw
            for r in range(d):
                acc = 0.0
                for k in range(nb):
                    acc += P[r, starts[k]]
                PH[r] = acc
            for k in range(nb):
                sv += PH[starts[k]]
            for r in range(d):
                for c in range(r, d):
                    v = P[r, c] - PH[r] * PH[c] / sv
                    P[r, c] = v
                    P[c, r] = v
            tprev = tb
            i = j
        out[s] = P  # (filtered at the last bin's time)
        dt = t[hi - 1] - tprev
        if dt > 0:
            for k in range(nb):
                _block_transition(kinds[k], pars[k], dt, Ab[k], Qb[k])
            _predict(kinds, starts, size, Ab, Qb, out[s], M, x, xp)
    return out


@numba.njit(cache=True)
def mean_recursion(At, idx, y, starts, sizes, K, isS, x, n0, ey):
    """The state recursion (FP64, K and 1 / sqrt(S) as given) from x at point
    n0 - 1: ey of the points n0... ."""
    nb, d = starts.shape[0], K.shape[1]
    for i in range(n0, len(y)):
        u = idx[i]
        for k in range(nb):
            s0 = starts[k]
            if sizes[k] == 1:
                x[s0] = At[u, k, 0, 0] * x[s0]
            else:
                a, b = x[s0], x[s0 + 1]
                x[s0] = At[u, k, 0, 0] * a + At[u, k, 0, 1] * b
                x[s0 + 1] = At[u, k, 1, 0] * a + At[u, k, 1, 1] * b
        hx = 0.0
        for k in range(nb):
            hx += x[starts[k]]
        innov = y[i] - hx
        ey[i] = innov * isS[i]
        for r in range(d):
            x[r] += K[i, r] * innov


@numba.njit(cache=True)
def loglik_from_whitened(ey, isS, n_diffuse):
    """Log likelihood from the normalised innovations and 1 / sqrt(S), as
    _filter_segment."""
    ll = 0.0
    for i in range(n_diffuse, len(ey)):
        ll -= 0.5 * (math.log(2 * math.pi) - 2.0 * math.log(isS[i]) + ey[i] * ey[i])
    return ll


SEGMENT_MIN = 1024  # points per segment of the parallel passes, at least


def n_segments(n, threads):
    """Segments of the parallel passes: 2 per thread, >= SEGMENT_MIN points
    each; 1 (sequential) for one thread, as the segment elements cost 1.75x
    the sequential filter."""
    return max(1, min(2 * threads, n // SEGMENT_MIN)) if threads > 1 else 1


def log_likelihood(model, t, y, yerr, threads=1):
    """Gaussian log likelihood (data in ppt). threads > 1: in parallel
    segments (2 per thread, >= SEGMENT_MIN points each) if the data are long
    enough, exact up to rounding (~1e-15 relative), in numba's current
    number of threads; the segment work is 1.75x that of the sequential
    filter, so one thread always runs sequentially."""
    kinds, starts, pars, P0 = model.blocks()
    R = yerr**2 + model.jitter**2
    t, y = np.asarray(t, float), np.asarray(y, float)
    nseg = n_segments(len(t), threads)
    if nseg >= 2:
        return kalman_loglik_parallel(t, y, R, kinds, starts, pars, P0, model.n_diffuse, nseg)
    return kalman_loglik(t, y, R, kinds, starts, pars, P0, model.n_diffuse)


def gp_mean(model, t, y, yerr, threads=1):
    """Posterior mean of the baseline at t (RTS smoother). threads > 1: in
    parallel segments (as log_likelihood), exact up to rounding."""
    kinds, starts, pars, P0 = model.blocks()
    R = yerr**2 + model.jitter**2
    t, y = np.asarray(t, float), np.asarray(y, float)
    nseg = n_segments(len(t), threads)
    return smoothed_mean(t, y, R, kinds, starts, pars, P0, nseg)


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

"""The gradient of the rotation fit's log likelihood on the GPU (FP64), from
the RTS smoother in segments (gp.gradient_segments): the segment elements of
driver._Elements, joined on the host forwards and backwards (the smoothed
state at every segment boundary); per segment the filter (storing the
filtered states) and the smoother backwards (storing the smoothed states),
in scalars as gpu.generated; per point the predicted covariance, its
Cholesky factor and the smoother gain, and the adjoints of the step into
the point (gradient_segments' functions as device functions; atomic adds per
run of equal step-table entries); the first points (driver.FIT_HEAD and
more) on the host. Equal to the CPU's gradient.RotationGradient up to
rounding (~1e-9 relative)."""

import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import cuda, float32, float64
from numba.cuda.cudadrv import driver as _driver

from transitleastsquares.gp import gradient_segments as gs
from transitleastsquares.gpu import driver, generated as gen

FLUSH_CUDA = '''
def flush(ACC, u, loc):
    for k in range(3):
        for p in range(2):
            for q in range(2):
                cuda.atomic.add(ACC, (u, 0, k + 1, p, q), loc[k * 8 + p * 2 + q])
                cuda.atomic.add(ACC, (u, 1, k + 1, p, q), loc[k * 8 + 4 + p * 2 + q])
'''
_DEV = {dt: gs.build("cuda", FLUSH_CUDA, dict(cuda=cuda), dt) for dt in (np.float64, np.float32)}
SIZES = (1, 2, 2, 2)  # the rotation model's blocks
STORE_MAX = 1 << 27  # doubles of states on the device (1 GB): longer series use the CPU gradient
SEGMENT = 128  # points per segment of the sequential kernels (filter, smoother)
CHUNK = 16  # points per thread of the adjoints (one flush per run of equal entries; 32: 2.3 ms, 16: 2.1 ms, 4: 2.0 ms)
POINT_BLOCK = 128  # threads per CUDA block of the per-point kernels


@functools.lru_cache(maxsize=None)
def filter_store(sizes):
    """The filter per segment from the filtered state before it (x_, P_), storing the filtered states XF, PF
    (segment 0 also the state before it, at edges[0] - 1)."""
    starts, d = gen._layout(sizes)
    H = starts
    b = ["q = cuda.grid(1)", "if q >= x_.shape[0]:", "    return"]
    for r in range(d):
        b.append(f"x{r} = x_[q, {r}]")
        b += [f"P{r}_{c} = P_[q, {r}, {c}]" for c in range(r, d)]
    store = [f"XF[i, {r}] = x{r}" for r in range(d)]
    store += [f"PF[i, {r}, {c}] = {gen._sym('P', r, c)}" for r in range(d) for c in range(d)]
    b += ["if q == 0:", "    i = edges[0] - 1"] + gen._indent(store)
    b.append("for i in range(edges[q], edges[q + 1]):")
    s = gen._load_transitions(sizes) + gen._predict_cov(sizes) + gen._apply_block(sizes, [lambda r: f"x{r}"])
    for r in range(d):
        s.append(f"PH{r} = " + " + ".join(gen._sym("P", r, h) for h in H))
    s.append("s = R[i] + jit2 + " + " + ".join(f"PH{h}" for h in H))
    s.append("innov = y[i] - " + " - ".join(f"x{h}" for h in H))
    s.append("inv = F1 / s")  # (F1: FP32, so that FP32 arrays keep the arithmetic in FP32)
    for r in range(d):
        s += [f"t0 = PH{r} * inv", f"x{r} += t0 * innov"]
        s += [f"P{r}_{c} -= t0 * PH{c}" for c in range(r, d)]
    s += store
    return gen._build("filter_store", sizes, "T, idx, y, R, jit2, edges, x_, P_, XF, PF", b + gen._indent(s))


@functools.lru_cache(maxsize=None)
def smoother_back(sizes):
    """The smoother per segment from the smoothed state at its last point (MSB[q + 1], PSB[q + 1]), storing
    MS, PS (segment 0 also the state before it, MSB[0], at edges[0] - 1): ms = xf + G (ms' - xp'),
    Ps = Pf + G (Ps' - Pp') G' with G' = GTS[i]."""
    _, d = gen._layout(sizes)
    sym = gen._sym
    b = ["q = cuda.grid(1)", "nk = MSB.shape[0] - 1", "if q >= nk:", "    return", "a = edges[q]", "bq = edges[q + 1]"]
    for r in range(d):
        b.append(f"ms{r} = MSB[q + 1, {r}]")
        b += [f"Ps{r}_{c} = PSB[q + 1, {r}, {c}]" for c in range(r, d)]
    store = [f"MS[i, {r}] = ms{r}" for r in range(d)]
    store += [f"PS[i, {r}, {c}] = {sym('Ps', r, c)}" for r in range(d) for c in range(d)]
    b += ["i = bq - 1"] + store
    b += ["if q == 0:", "    for r in range(7):", "        MS[a - 1, r] = MSB[0, r]", "        for c in range(7):",
          "            PS[a - 1, r, c] = PSB[0, r, c]"]
    b.append("for i in range(bq - 2, a - 1, -1):")
    s = [f"g{l}_{r} = GTS[i, {l}, {r}]" for l in range(d) for r in range(d)]
    s += [f"dl{r} = ms{r} - XP[i + 1, {r}]" for r in range(d)]
    s += [f"dP{r}_{c} = Ps{r}_{c} - PP[i + 1, {r}, {c}]" for r in range(d) for c in range(r, d)]
    s += [f"ms{r} = XF[i, {r}] + " + " + ".join(f"g{l}_{r} * dl{l}" for l in range(d)) for r in range(d)]
    for c in range(d):  # column c of W = dP G', then Ps[:, c] = Pf[:, c] + G W[:, c]
        s += [f"w{l} = " + " + ".join(f"{sym('dP', l, m)} * g{m}_{c}" for m in range(d)) for l in range(d)]
        s += [f"Ps{r}_{c} = PF[i, {r}, {c}] + " + " + ".join(f"g{l}_{r} * w{l}" for l in range(d)) for r in range(c + 1)]
    s += store
    return gen._build("smoother_back", sizes, "edges, XF, PF, XP, PP, GTS, MSB, PSB, MS, PS", b + gen._indent(s))


@functools.lru_cache(maxsize=None)
def _kernels(dtype):
    """The per-point kernels in this precision (the adjoints are added in FP64)."""
    dev = _DEV[dtype]
    point_gains, point_grad, flush = dev["point_gains"], dev["point_grad"], dev["flush"]
    ft = float64 if dtype == np.float64 else float32

    @cuda.jit
    def gains_kernel(T, idx, n0, n, XF, PF, XP, PP, LL, GTS, OK):
        i = n0 - 1 + cuda.grid(1)
        if i >= n - 1:
            return
        Pf = cuda.local.array((7, 7), ft)
        B = cuda.local.array((7, 7), ft)
        Pp = cuda.local.array((7, 7), ft)
        L = cuda.local.array((7, 7), ft)
        GT = cuda.local.array((7, 7), ft)
        v = cuda.local.array(7, ft)
        if not point_gains(T, idx, i, XF, PF, XP, PP, LL, GTS, Pf, B, Pp, L, GT, v):
            OK[0] = 0

    @cuda.jit
    def grad_kernel(idx, y, R, jit2, n0, n, chunk, MS, PS, XP, PP, LL, GTS, ACC, RS):
        j = cuda.grid(1)
        i0 = n0 + j * chunk
        if i0 >= n:
            return
        Y = cuda.local.array((7, 7), ft)
        Z = cuda.local.array((7, 7), ft)
        dP = cuda.local.array((7, 7), ft)
        w = cuda.local.array(7, ft)
        dl = cuda.local.array(7, ft)
        L = cuda.local.array((7, 7), ft)
        loc = cuda.local.array(24, float64)
        rs = 0.0
        ucur = -1
        for i in range(i0, min(i0 + chunk, n)):
            u = idx[i]
            if u != ucur:
                if ucur >= 0:
                    flush(ACC, ucur, loc)
                for k in range(24):
                    loc[k] = 0.0
                ucur = u
            rs += point_grad(y, R, jit2, i, MS, PS, XP, PP, LL, GTS, Y, Z, dP, w, dl, L, loc)
        flush(ACC, ucur, loc)
        RS[j] = rs

    return gains_kernel, grad_kernel


def _h2d(dev, a):
    a = np.ascontiguousarray(a)
    _driver.host_to_device(dev, a, a.nbytes)  # (copy_to_device's checks cost 0.3 ms per call)


def _d2h(a, dev):
    _driver.device_to_host(a, dev, a.nbytes)


def _points_grid(m):
    return (m + POINT_BLOCK - 1) // POINT_BLOCK, POINT_BLOCK


class KalmanGradient:
    """f(lx) -> (log likelihood, gradient) of the rotation model for fixed data t, y, yerr (ppt) on the GPU,
    as gp.gradient.RotationGradient. segment: points per segment of the sequential kernels. precision:
    "fp64", or "fp32" for the filter, gains, smoother and adjoint kernels (the log likelihood, the elements and
    the segment boundaries stay FP64): on Talens light curves the gradient then differs by 1e-6 to 5e-4 of its
    largest component (absolute ~1e-2), far below what changes the fit, and the FP32 kernels run at the
    card's FP32 rate (64x FP64 on consumer GPUs)."""

    def __init__(self, t, y, yerr, segment=SEGMENT, precision="fp32"):
        t, y, yerr = (np.asarray(a, float) for a in (t, y, yerr))
        n = len(t)
        self.e = driver._Elements(t, y, yerr**2, segment)
        n0, nk = self.e.n0, self.e.nk
        if nk == 0 or n * 260 > STORE_MAX:
            raise ValueError("KalmanGradient: series too short or too long for the GPU")
        self.t, self.y, self.R, self.n0, self.nk = t, y, yerr**2, n0, nk
        self.dt = dt = {"fp64": np.float64, "fp32": np.float32}[precision]
        arr = functools.partial(cuda.device_array, dtype=dt)
        self.XF, self.XP, self.MS = arr((n, 7)), arr((n, 7)), arr((n, 7))
        self.PF, self.PP, self.LL, self.GTS, self.PS = (arr((n, 7, 7)) for _ in range(5))
        self.X0, self.P0S = arr((nk, 7)), arr((nk, 7, 7))
        self.MSB, self.PSB = arr((nk + 1, 7)), arr((nk + 1, 7, 7))
        self.dT = arr((len(self.e.steps), 2, 4, 2, 2))
        self.dy, self.dR = cuda.to_device(y.astype(dt)), cuda.to_device(self.R.astype(dt))
        self.k_gains, self.k_grad = (gains32(), grad32()) if dt == np.float32 else _kernels(dt)
        self.nchunk = (n - n0 + CHUNK - 1) // CHUNK
        self.RS, self.OK = cuda.device_array(self.nchunk), cuda.device_array(1, np.int32)
        ne = len(self.e.steps)
        self.ACC = cuda.device_array((ne, 2, 4, 2, 2))
        self.acc, self.rs, self.ok = np.zeros((ne, 2, 4, 2, 2)), np.empty(self.nchunk), np.ones(1, np.int32)
        self.zero = np.zeros((ne, 2, 4, 2, 2))
        self.k_filter, self.k_smooth = filter_store(SIZES), smoother_back(SIZES)
        self.pool = ThreadPoolExecutor(1)

    def __call__(self, lx):
        lx = np.asarray(lx, float)
        model, (kinds, starts, pars, P0), _, jit2 = gs.transitions(lx, self.e.steps)
        e, n0, n, nk = self.e, self.n0, len(self.t), self.nk
        h, _ = e(kinds, starts, pars, 7, jit2)  # elements on the device (uploads T to e.d["T"], e.T on the host)
        info = self.pool.submit(gs.info_backward, *h[:5])  # (in a second thread, with the forward join: GIL-free)
        ll, x0, P0s, xl, Pl = gs.join_forward(self.t[:n0], self.y[:n0], self.R[:n0] + jit2, kinds, starts, pars, P0,
                                              model.n_diffuse, n0, *h)
        dv, grid, dt = e.d, driver._grid(nk), self.dt
        T = dv["T"]
        if dt != np.float64:
            _h2d(self.dT, e.T.astype(dt))
            T = self.dT
        j2 = dt(jit2)
        _h2d(self.X0, x0.astype(dt))
        _h2d(self.P0S, P0s.astype(dt))
        _h2d(self.OK, self.ok)
        self.k_filter[grid](T, dv["idx"], self.dy, self.dR, j2, dv["edges"], self.X0, self.P0S, self.XF, self.PF)
        self.k_gains[_points_grid(n - n0)](T, dv["idx"], n0, n, self.XF, self.PF, self.XP, self.PP, self.LL, self.GTS,
                                           self.OK)
        ms_b, Ps_b = gs.boundary_smoothed(*info.result(), x0, P0s, xl, Pl)  # (on the host while the two kernels run)
        _h2d(self.MSB, ms_b.astype(dt))
        _h2d(self.PSB, Ps_b.astype(dt))
        self.k_smooth[grid](dv["edges"], self.XF, self.PF, self.XP, self.PP, self.GTS, self.MSB, self.PSB, self.MS,
                            self.PS)
        _h2d(self.ACC, self.zero)
        self.k_grad[_points_grid(self.nchunk)](dv["idx"], self.dy, self.dR, j2, n0, n, CHUNK, self.MS, self.PS, self.XP,
                                               self.PP, self.LL, self.GTS, self.ACC, self.RS)
        ACC = np.zeros(self.acc.shape)  # the first points and the derivative tables on the host while the kernels run
        P0b, rh = gs.head_and_prior(gs._cpu(), e.T, e.idx, self.y, self.R, jit2, P0, n0, ms_b[0].copy(),
                                    Ps_b[0].copy(), ACC)
        tables = gs.gradient.derivative_tables(lx, e.steps, starts)
        ok = np.empty(1, np.int32)
        _d2h(ok, self.OK)
        _d2h(self.acc, self.ACC)
        _d2h(self.rs, self.RS)
        if not ok[0]:
            raise np.linalg.LinAlgError("predicted covariance not positive definite")
        ACC += self.acc
        rs = self.rs.sum()
        if P0b is None or not np.isfinite(rs):
            raise np.linalg.LinAlgError("predicted covariance not positive definite")
        return ll, gs.gradient.contract(lx, e.steps, starts, ACC[:, 0], ACC[:, 1], P0b, rs + rh, model.jitter, tables)


# ---------------------------------------------------------------- FP32 per-point kernels in scalars
H_IDX = (0, 1, 3, 5)
BLOCKS = ((1, 1), (2, 3), (3, 5))  # (block, first state) of the three SHO blocks


def _chol_lines(P, L, fail):
    """Straight-line Cholesky of the symmetric scalars P(r, c) into L{i}_{j} (reciprocal diagonal)."""
    out = []
    for i in range(7):
        for j in range(i + 1):
            e = P(i, j) + "".join(f" - {L}{i}_{k} * {L}{j}_{k}" for k in range(j))
            if i == j:
                out += [f"acc = {e}", "if not acc > F0:"] + ["    " + x for x in fail] + [f"{L}{i}_{i} = F1 / math.sqrt(acc)"]
            else:
                out.append(f"{L}{i}_{j} = ({e}) * {L}{j}_{j}")
    return out


def _solve_lines(L, rhs, out, tmp="z"):
    """out(r) = (L L')^-1 rhs(r) for one column, straight-line (L with reciprocal diagonal)."""
    s = []
    for i in range(7):
        s.append(f"{tmp}{i} = ({rhs(i)}" + "".join(f" - {L}{i}_{k} * {tmp}{k}" for k in range(i)) + f") * {L}{i}_{i}")
    for i in range(6, -1, -1):
        s.append(f"{out(i)} = ({tmp}{i}" + "".join(f" - {L}{k}_{i} * {out(k)}" for k in range(i + 1, 7)) + f") * {L}{i}_{i}")
    return s


@functools.lru_cache(maxsize=None)
def gains32():
    """gradient_segments.point_gains per point in scalars (registers instead of local arrays)."""
    sizes = SIZES
    sym = gen._sym
    b = ["i = n0 - 1 + cuda.grid(1)", "if i >= n - 1:", "    return", "u = idx[i + 1]"]
    for k, sz in enumerate(sizes):
        for r in range(sz):
            for c in range(sz):
                b.append(f"a{k}{r}{c} = T[u, 0, {k}, {r}, {c}]")
                if r <= c:
                    b.append(f"q{k}{r}{c} = T[u, 1, {k}, {r}, {c}]")
    b += [f"P{r}_{c} = PF[i, {r}, {c}]" for r in range(7) for c in range(r, 7)]
    b += [f"x{r} = XF[i, {r}]" for r in range(7)]
    b += gen._predict_cov(sizes)  # M = A Pf (= B), P = A Pf A' + Q
    b += gen._apply_block(sizes, [lambda r: f"x{r}"])
    b += _chol_lines(lambda r, c: sym("P", r, c), "L", ["OK[0] = 0", "return"])
    for c in range(7):
        b += _solve_lines("L", lambda r, c=c: f"M{r}_{c}", lambda r, c=c: f"g{r}_{c}")
    for r in range(7):
        b.append(f"XP[i + 1, {r}] = x{r}")
        for c in range(7):
            b += [f"PP[i + 1, {r}, {c}] = {sym('P', r, c)}", f"GTS[i, {r}, {c}] = g{r}_{c}",
                  f"LL[i + 1, {r}, {c}] = {f'L{r}_{c}' if c <= r else 'F0'}"]
    return gen._build("gains32", sizes, "T, idx, n0, n, XF, PF, XP, PP, LL, GTS, OK", b)


@functools.lru_cache(maxsize=None)
def grad32():
    """gradient_segments.point_grad per point in scalars, chunk points per thread, the adjoints per run of equal
    entries in FP64 scalars, flushed with atomic adds."""
    sizes = SIZES
    sym = gen._sym
    names = [(blk, s0, p, q) for blk, s0 in BLOCKS for p in range(2) for q in range(2)]
    b = ["j = cuda.grid(1)", "i0 = n0 + j * chunk", "if i0 >= n:", "    return", "rs = 0.0", "ucur = -1"]
    b += [f"cA{m} = 0.0" for m in range(12)] + [f"cQ{m} = 0.0" for m in range(12)]
    flush = []
    for m, (blk, s0, p, q) in enumerate(names):
        flush += [f"cuda.atomic.add(ACC, (ucur, 0, {blk}, {p}, {q}), cA{m})", f"cuda.atomic.add(ACC, (ucur, 1, {blk}, {p}, {q}), cQ{m})"]
    b.append("for i in range(i0, min(i0 + chunk, n)):")
    s = ["u = idx[i]", "if u != ucur:", "    if ucur >= 0:"] + ["        " + x for x in flush]
    s += ["    " + x for x in [f"cA{m} = 0.0" for m in range(12)] + [f"cQ{m} = 0.0" for m in range(12)] + ["ucur = u"]]
    s += [f"L{r}_{c} = LL[i, {r}, {c}]" for r in range(7) for c in range(r + 1)]
    s += [f"dl{r} = MS[i, {r}] - XP[i, {r}]" for r in range(7)]
    s += [f"dP{r}_{c} = PS[i, {r}, {c}] - PP[i, {r}, {c}]" for r in range(7) for c in range(r, 7)]
    for c in range(7):  # Y = Pp^-1 dP
        s += _solve_lines("L", lambda r, c=c: sym("dP", r, c), lambda r, c=c: f"y{r}_{c}")
    s += _solve_lines("L", lambda r: f"dl{r}", lambda r: f"w{r}")  # w = Pp^-1 d
    zs = {}
    for blk, s0 in BLOCKS:  # the needed entries of Pp^-1 dP Pp^-1 = Pp^-1 Y': columns s0, s0 + 1
        for q in range(2):
            c = s0 + q
            s += _solve_lines("L", lambda r, c=c: f"y{c}_{r}", lambda r, c=c: f"zc{r}", tmp="t")
            for p in range(2):
                s.append(f"z{s0 + p}_{c} = zc{s0 + p}")
    for m, (blk, s0, p, q) in enumerate(names):
        a, c = s0 + p, s0 + q
        e = f"w{a} * MS[i - 1, {c}]" + "".join(f" + y{a}_{l} * GTS[i - 1, {l}, {c}]" for l in range(7))
        s += [f"cA{m} += {e}", f"cQ{m} += float32(0.5) * (z{a}_{c} + w{a} * w{c})"]
    hm = " + ".join(f"MS[i, {h}]" for h in H_IDX)
    hph = " + ".join(f"PS[i, {a}, {c}]" for a in H_IDX for c in H_IDX)
    s += [f"res = y[i] - ({hm})", "Ri = R[i] + jit2", f"rs += -0.5 * (1.0 / Ri - (res * res + {hph}) / (Ri * Ri))"]
    b += gen._indent(s)
    b += flush + ["RS[j] = rs"]
    return gen._build("grad32", sizes, "idx, y, R, jit2, n0, n, chunk, MS, PS, XP, PP, LL, GTS, ACC, RS", b)

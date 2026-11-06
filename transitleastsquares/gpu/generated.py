"""CUDA kernels of the Kalman pass (FP64) and the single events (FP32), written out in
scalars for a model's block structure (sizes: 1 or 2 states per block; H picks the first
state of each).

Everything here could be written with local arrays and loops, as the other kernels are,
but those arrays live in stack memory, where every access is a load or a store; scalars
stay in registers. On the RTX 5060 this makes these kernels about 4x faster. A step takes about 4.7 us of a thread's time, so the length of a segment sets a kernel's time
(driver._segment_length)."""

import functools
import math

from numba import cuda, float32, float64, int32

from transitleastsquares.gpu.kernels import F0, F1, _first, _template


def _layout(sizes):
    starts = [sum(sizes[:k]) for k in range(len(sizes))]
    return starts, sum(sizes)


def _indent(lines, n=1):
    return ["    " * n + x for x in lines]


def _load_transitions(sizes, q=True):
    """a{k}{r}{c} and q{k}{r}{c} (upper triangle) of the step table's entry u (T: A and Q in one array,
    as one transfer is what costs on the host)."""
    out = ["u = idx[i]"]
    for k, sz in enumerate(sizes):
        for r in range(sz):
            for c in range(sz):
                out.append(f"a{k}{r}{c} = T[u, 0, {k}, {r}, {c}]")
                if q and r <= c:
                    out.append(f"q{k}{r}{c} = T[u, 1, {k}, {r}, {c}]")
    return out


def _load_at(sizes):
    out = ["u = idx[i]"]
    for k, sz in enumerate(sizes):
        out += [f"a{k}{r}{c} = At[u, {k}, {r}, {c}]" for r in range(sz) for c in range(sz)]
    return out


def _sym(name, a, b):
    return f"{name}{min(a, b)}_{max(a, b)}"


def _predict_cov(sizes):
    """P = A P A' + Q on the upper triangle of the scalars P{r}_{c}."""
    starts, d = _layout(sizes)
    where = {}
    for k, sz in enumerate(sizes):
        for i in range(sz):
            where[starts[k] + i] = (k, i, starts[k], sz)
    out = []
    for r in range(d):
        k, i, s0, sz = where[r]
        for c in range(d):
            out.append(f"M{r}_{c} = " + " + ".join(f"a{k}{i}{l} * {_sym('P', s0 + l, c)}" for l in range(sz)))
    for r in range(d):
        for c in range(r, d):
            k, i, s0, sz = where[c]
            e = " + ".join(f"M{r}_{s0 + l} * a{k}{i}{l}" for l in range(sz))
            kr, ir, _, _ = where[r]
            if kr == k:
                e += f" + q{k}{min(ir, i)}{max(ir, i)}"
            out.append(f"P{r}_{c} = {e}")
    return out


def _apply_block(sizes, names):
    """v = A v, for each of the column-like scalar families in names (a function of the state's index)."""
    starts, _ = _layout(sizes)
    out = []
    for name in names:
        for k, sz in enumerate(sizes):
            s0 = starts[k]
            if sz == 2:
                out += [f"t0 = {name(s0)}", f"t1 = {name(s0 + 1)}", f"{name(s0)} = a{k}00 * t0 + a{k}01 * t1",
                        f"{name(s0 + 1)} = a{k}10 * t0 + a{k}11 * t1"]
            else:
                out.append(f"{name(s0)} = a{k}00 * {name(s0)}")
    return out


def _build(name, sizes, signature, body):
    """The kernel, cached on disk under its name and this file's time stamp (the file must exist for numba's cache)."""
    name = name + "_" + "".join(str(x) for x in sizes)
    src = f"@cuda.jit(cache=True)\ndef {name}({signature}):\n" + "\n".join(_indent(body)) + "\n"
    ns = dict(cuda=cuda, math=math, TWO_PI=2 * math.pi, float32=float32, float64=float64, int32=int32, F0=F0, F1=F1,
              _first=_first, _template=_template)
    exec(compile(src, __file__, "exec"), ns)  # noqa: S102 (source generated above from the block sizes only)
    return ns[name]


@functools.lru_cache(maxsize=None)
def elements(sizes):
    """gp.statespace._segment_element per segment, with the transitions of a step table.
    R + jit2: the noise variance. E: all elements in one array, as driver._Elements unpacks them
    (F, Pe, J, then c, eta, then const), for one transfer."""
    starts, d = _layout(sizes)
    H = starts
    b = ["q = cuda.grid(1)", "if q >= nk:", "    return"]
    for r in range(d):
        b += [f"c{r} = 0.0", f"e{r} = 0.0"]
        for c in range(d):
            b.append(f"F{r}_{c} = {1.0 if r == c else 0.0}")
            if r <= c:
                b += [f"P{r}_{c} = 0.0", f"J{r}_{c} = 0.0"]
    b += ["const = 0.0", "for i in range(edges[q], edges[q + 1]):"]
    s = _load_transitions(sizes) + _predict_cov(sizes)
    s += _apply_block(sizes, [lambda r, c=c: f"F{r}_{c}" for c in range(d)] + [lambda r: f"c{r}"])
    for r in range(d):
        s.append(f"PH{r} = " + " + ".join(_sym("P", r, h) for h in H))
        s.append(f"hF{r} = " + " + ".join(f"F{h}_{r}" for h in H))
    s.append("s = R[i] + jit2 + " + " + ".join(f"PH{h}" for h in H))
    s.append("v0 = y[i] - " + " - ".join(f"c{h}" for h in H))
    s += ["inv = 1.0 / s", "const += math.log(TWO_PI * s) + v0 * v0 * inv"]
    for r in range(d):
        s += [f"t0 = hF{r} * inv", f"e{r} += t0 * v0"]
        s += [f"J{r}_{c} += t0 * hF{c}" for c in range(r, d)]
    for r in range(d):
        s += [f"t0 = PH{r} * inv", f"c{r} += t0 * v0"]
        s += [f"F{r}_{c} -= t0 * hF{c}" for c in range(d)]
        s += [f"P{r}_{c} -= t0 * PH{c}" for c in range(r, d)]
    b += _indent(s)
    b += [f"o0 = q * {d * d}", f"o1 = nk * {d * d} + o0", f"o2 = nk * {2 * d * d} + o0", f"o3 = nk * {3 * d * d} + q * {d}",
          f"o4 = nk * {3 * d * d + d} + q * {d}", f"E[nk * {3 * d * d + 2 * d} + q] = const"]
    for r in range(d):
        b += [f"E[o3 + {r}] = c{r}", f"E[o4 + {r}] = e{r}"]
        for c in range(d):
            i = r * d + c
            b += [f"E[o0 + {i}] = F{r}_{c}", f"E[o1 + {i}] = {_sym('P', r, c)}", f"E[o2 + {i}] = {_sym('J', r, c)}"]
    return _build("elements", sizes, "T, idx, y, R, jit2, edges, nk, E", b)


@functools.lru_cache(maxsize=None)
def filter_segments(sizes):
    """The Kalman filter per segment, from the filtered state before it (the
    segment elements joined; x_, P_): K and 1 / sqrt(S) in the type of the arrays given, ey in FP64."""
    starts, d = _layout(sizes)
    H = starts
    b = ["q = cuda.grid(1)", "if q >= x_.shape[0]:", "    return"]
    for r in range(d):
        b.append(f"x{r} = x_[q, {r}]")
        b += [f"P{r}_{c} = P_[q, {r}, {c}]" for c in range(r, d)]
    b.append("for i in range(edges[q], edges[q + 1]):")
    s = _load_transitions(sizes) + _predict_cov(sizes) + _apply_block(sizes, [lambda r: f"x{r}"])
    for r in range(d):
        s.append(f"PH{r} = " + " + ".join(_sym("P", r, h) for h in H))
    s.append("s = R[i] + " + " + ".join(f"PH{h}" for h in H))
    s.append("innov = y[i] - " + " - ".join(f"x{h}" for h in H))
    s += ["inv = 1.0 / s", "rt = math.sqrt(inv)", "isS[i] = rt", "ey[i] = innov * rt"]
    for r in range(d):
        s += [f"t0 = PH{r} * inv", f"K[i, {r}] = t0", f"x{r} += t0 * innov"]
        s += [f"P{r}_{c} -= t0 * PH{c}" for c in range(r, d)]
    return _build("filter_segments", sizes, "T, idx, y, R, edges, x_, P_, K, isS, ey", b + _indent(s))


def _adjoint_step(sizes, value, others):
    """w = A' (w - H' z) for the scalars value(state) (with z in zname) and the families in others (with their own z)."""
    starts, _ = _layout(sizes)
    out = []
    for name, z in [(value[0], value[1])] + others:
        for k, sz in enumerate(sizes):
            s0 = starts[k]
            if sz == 2:
                out += [f"t0 = {name(s0)} - {z}", f"t1 = {name(s0 + 1)}", f"{name(s0)} = a{k}00 * t0 + a{k}10 * t1",
                        f"{name(s0 + 1)} = a{k}01 * t0 + a{k}11 * t1"]
            else:
                out.append(f"{name(s0)} = a{k}00 * ({name(s0)} - {z})")
    return out


@functools.lru_cache(maxsize=None)
def adjoint_maps(sizes):
    """statespace.mean_adjoint's backward recursion over a segment as an affine map:
    w before its first point = W_ w at its last point + v_."""
    _, d = _layout(sizes)
    b = ["q = cuda.grid(1)", "if q >= v_.shape[0]:", "    return"]
    for r in range(d):
        b.append(f"v{r} = 0.0")
        b += [f"W{r}_{c} = {1.0 if r == c else 0.0}" for c in range(d)]
    b.append("for i in range(edges[q + 1] - 1, edges[q] - 1, -1):")
    s = [f"k{r} = K[i, {r}]" for r in range(d)]
    s.append("zv = ey[i] * isS[i] + " + " + ".join(f"k{r} * v{r}" for r in range(d)))
    s += [f"kw{c} = " + " + ".join(f"k{r} * W{r}_{c}" for r in range(d)) for c in range(d)]
    s += _load_transitions(sizes, False)
    s += _adjoint_step(sizes, (lambda r: f"v{r}", "zv"), [(lambda r, c=c: f"W{r}_{c}", f"kw{c}") for c in range(d)])
    b += _indent(s)
    for r in range(d):
        b.append(f"v_[q, {r}] = v{r}")
        b += [f"W_[q, {r}, {c}] = W{r}_{c}" for c in range(d)]
    return _build("adjoint_maps", sizes, "T, idx, K, isS, ey, edges, W_, v_", b)


@functools.lru_cache(maxsize=None)
def adjoint_fill(sizes):
    """z = C^-1 y per segment (statespace.mean_adjoint), from w at its last point."""
    _, d = _layout(sizes)
    b = ["q = cuda.grid(1)", "if q >= w_.shape[0]:", "    return"]
    b += [f"w{r} = w_[q, {r}]" for r in range(d)]
    b.append("for i in range(edges[q + 1] - 1, edges[q] - 1, -1):")
    s = ["zz = ey[i] * isS[i] + " + " + ".join(f"K[i, {r}] * w{r}" for r in range(d)), "z[i] = zz"]
    s += _load_transitions(sizes, False) + _adjoint_step(sizes, (lambda r: f"w{r}", "zz"), [])
    return _build("adjoint_fill", sizes, "T, idx, K, isS, ey, edges, w_, z", b + _indent(s))


@functools.lru_cache(maxsize=None)
def single_gp(sizes):
    """gp.events.single_events_split in FP32 (one thread per row): a = sum s z
    minus the diffuse-start terms, b = |L^-1 s|^2 by the filter with the precomputed gains,
    and the tail x' W x (W symmetric). The transitions are read when the entry changes."""
    starts, d = _layout(sizes)
    b = ["i = cuda.grid(1)", "if i >= ab.shape[0]:", "    return", "nd = eyd.shape[0]", "j = row_j[i]",
         "tcm = t0 + float64(i - off[j]) * float64(mstep[j]) * delta", "D = durs[j]", "invD = F1 / D",
         "half = 0.5 * float64(D) + 0.5 * float64(texp)", "n = t.shape[0]", "nsub = nsubs[j]", "s = _first(t, tcm - half)",
         "aa = F0", "bb = F0", "last = -1", "ulast = int32(-1)"]
    b += [f"x{r} = F0" for r in range(d)]
    for k, sz in enumerate(sizes):
        b += [f"a{k}{r}{c} = F0" for r in range(sz) for c in range(sz)]
    b.append("while s < n and t[s] < tcm + half:")
    w = ["u = idx[s]", "if u != ulast:", "    ulast = u"]
    for k, sz in enumerate(sizes):
        w += [f"    a{k}{r}{c} = At[u, {k}, {r}, {c}]" for r in range(sz) for c in range(sz)]
    w += _apply_block(sizes, [lambda r: f"x{r}"])
    w.append("hx = " + " + ".join(f"x{h}" for h in starts))
    w += ["v = _template(float32(t[s] - tcm), invD, table, texp, nsub)", "aa += v * z[s]", "innov = v - hx", "e = innov * isS[s]",
          "if s < nd:", "    aa -= e * eyd[s]", "bb += e * e"]
    w += [f"x{r} += K[s, {r}] * innov" for r in range(d)]
    w += ["last = s", "s += 1"]
    b += _indent(w)
    b.append("if last >= 0:")
    t = []
    for r in range(d):
        t.append(f"bb += x{r} * x{r} * W[last, {r}, {r}]")
        t += [f"bb += (x{r} + x{r}) * x{c} * W[last, {r}, {c}]" for c in range(r + 1, d)]
    b += _indent(t)
    b += ["ab[i, 0] = aa", "ab[i, 1] = bb"]
    return _build("single_gp", sizes, "t, z, eyd, At, idx, K, isS, W, t0, delta, durs, nsubs, table, texp, mstep, off, row_j, ab", b)


def _tail_step(sizes, d, starts):
    """One backward step of gp.events.tail_operators with F = (I - K H) A: Y = W - H' (K' W) - (W K) H + H' (K' W K + 1 / S) H,
    W = A' Y A (upper triangle), and w = A' (w - H' z)."""
    where = {}
    for k, sz in enumerate(sizes):
        for i in range(sz):
            where[starts[k] + i] = (k, i, starts[k], sz)
    s = [f"k{r} = K[i, {r}]" for r in range(d)]
    s += ["sq = isS[i]", "zz = ey[i] * sq + " + " + ".join(f"k{r} * w{r}" for r in range(d))]
    s += [f"wk{r} = " + " + ".join(f"{_sym('W', r, c)} * k{c}" for c in range(d)) for r in range(d)]
    s.append("kwk = sq * sq + " + " + ".join(f"k{r} * wk{r}" for r in range(d)))
    for r in range(d):
        for c in range(r, d):
            e = f"W{r}_{c}"
            if r in starts:
                e += f" - wk{c}"
            if c in starts:
                e += f" - wk{r}"
            if r in starts and c in starts:
                e += " + kwk"
            s.append(f"Y{r}_{c} = {e}")
    s += _load_at(sizes)
    for r in range(d):
        for c in range(d):
            k, i, s0, sz = where[c]
            s.append(f"M{r}_{c} = " + " + ".join(f"{_sym('Y', r, s0 + l)} * a{k}{l}{i}" for l in range(sz)))
    for r in range(d):
        k, i, s0, sz = where[r]
        for c in range(r, d):
            s.append(f"W{r}_{c} = " + " + ".join(f"a{k}{l}{i} * M{s0 + l}_{c}" for l in range(sz)))
    s += _adjoint_step(sizes, (lambda r: f"w{r}", "zz"), [])
    return s


@functools.lru_cache(maxsize=None)
def tail_maps(sizes):
    """Phase 1 of gp.events.tail_operators (one thread per segment k >= 1), FP32: the segment's affine map
    W_lo = Phi' W_b Phi + C, w_lo = Phi' w_b + cc, composed backwards from its boundary edges[k + 1]."""
    starts, d = _layout(sizes)
    b = ["k_ = cuda.grid(1) + 1", "if k_ >= Phi.shape[0]:", "    return"]
    for r in range(d):
        b.append(f"w{r} = F0")
        b += [f"W{r}_{c} = F0" for c in range(r, d)]
        b += [f"Ph{r}_{c} = {'F1' if r == c else 'F0'}" for c in range(d)]
    b.append("for m in range(edges[k_ + 1] - 1, edges[k_] - 1, -1):")
    s = ["i = m + 1"] + _tail_step(sizes, d, starts)
    for r in range(d):  # Ph = Ph F = (Ph - (Ph K) H) A
        s.append(f"pk = " + " + ".join(f"Ph{r}_{c} * k{c}" for c in range(d)))
        s += [f"Ph{r}_{h} -= pk" for h in starts]
        for k, sz in enumerate(sizes):
            s0 = starts[k]
            if sz == 2:
                s += [f"t0 = Ph{r}_{s0}", f"t1 = Ph{r}_{s0 + 1}", f"Ph{r}_{s0} = t0 * a{k}00 + t1 * a{k}10", f"Ph{r}_{s0 + 1} = t0 * a{k}01 + t1 * a{k}11"]
            else:
                s.append(f"Ph{r}_{s0} = Ph{r}_{s0} * a{k}00")
    b += _indent(s)
    for r in range(d):
        b.append(f"cc[k_, {r}] = w{r}")
        for c in range(d):
            b += [f"C[k_, {r}, {c}] = {_sym('W', r, c)}", f"Phi[k_, {r}, {c}] = Ph{r}_{c}"]
    return _build("tail_maps", sizes, "At, idx, K, isS, ey, edges, Phi, C, cc", b)


@functools.lru_cache(maxsize=None)
def tail_fill(sizes):
    """Phase 3 of gp.events.tail_operators (one thread per segment), FP32: W (upper triangle) and w of every
    point of the segment from the known boundary Wb[k], wb[k]."""
    starts, d = _layout(sizes)
    b = ["k_ = cuda.grid(1)", "if k_ >= Wb.shape[0]:", "    return"]
    for r in range(d):
        b.append(f"w{r} = wb[k_, {r}]")
        b += [f"W{r}_{c} = Wb[k_, {r}, {c}]" for c in range(r, d)]
    b.append("for m in range(edges[k_ + 1] - 1, edges[k_] - 1, -1):")
    s = ["i = m + 1"] + _tail_step(sizes, d, starts)
    for r in range(d):
        s.append(f"w_[m, {r}] = w{r}")
        s += [f"W_[m, {r}, {c}] = W{r}_{c}" for c in range(r, d)]
    b += _indent(s)
    return _build("tail_fill", sizes, "At, idx, K, isS, ey, edges, Wb, wb, W_, w_", b)

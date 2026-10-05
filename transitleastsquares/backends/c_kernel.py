"""Backend "c": the fused kernel in C (csrc/tls_kernel.c), with OpenMP threads.

The shared library is compiled on first use with the system C compiler
(``cc -O3 -march=native -ffast-math -fopenmp``) and cached in
$TLS_CACHE_DIR (default: ~/.cache/transitleastsquares, or
$XDG_CACHE_HOME/transitleastsquares). Threads share the data, so there is no
process pool and no pickling.
"""

import ctypes
import hashlib
import os
import subprocess
import sys

import numpy

from transitleastsquares import tls_constants
from transitleastsquares.backends import SearchBackend, SearchResult
from transitleastsquares.core_fused import FusedProblem

_SRC = os.path.join(os.path.dirname(__file__), "csrc", "tls_kernel.c")
_LIB = None
CFLAGS = ["-O3", "-march=native", "-ffast-math", "-fopenmp", "-shared", "-fPIC"]


def _cache_dir():
    if os.environ.get("TLS_CACHE_DIR"):
        return os.environ["TLS_CACHE_DIR"]
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "transitleastsquares")


def load_library():
    """Compile (if needed) and load the C kernel."""
    global _LIB
    if _LIB is not None:
        return _LIB
    with open(_SRC, "rb") as f:
        source = f.read()
    cc = os.environ.get("CC", "cc")
    key = hashlib.sha256(source + " ".join([cc] + CFLAGS).encode()).hexdigest()[:16]
    os.makedirs(_cache_dir(), exist_ok=True)
    path = os.path.join(
        _cache_dir(), f"tls_kernel_{key}{'.dll' if sys.platform == 'win32' else '.so'}"
    )
    if not os.path.exists(path):
        tmp = path + f".{os.getpid()}.tmp"
        cmd = [cc, *CFLAGS, _SRC, "-o", tmp, "-lm"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Compiling the TLS C kernel failed:\n{' '.join(cmd)}\n{result.stderr}"
            )
        os.replace(tmp, path)
    lib = ctypes.CDLL(path)
    D = numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags="C_CONTIGUOUS")
    I = numpy.ctypeslib.ndpointer(dtype=numpy.int64, flags="C_CONTIGUOUS")
    i64, f64, cint = ctypes.c_int64, ctypes.c_double, ctypes.c_int
    lib.tls_search.restype = cint
    lib.tls_search.argtypes = [
        D,
        i64,
        D,
        D,
        D,
        i64,
        cint,
        f64,  # periods, n_periods, t, y, inv_dy2, n, uniform, span
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,  # depth_min, R_min, R_max, M_min, M_max, upper
        I,
        I,
        I,
        I,
        i64,
        D,
        D,
        D,  # widths, rows, offsets, lengths, n_widths, profile, ov, a2
        f64,
        f64,
        cint,
        cint,  # margin, signal_depth, prune, n_threads
        D,
        I,
        D,  # outputs
    ]
    _LIB = lib
    return lib


class CBackend(SearchBackend):
    name = "c"
    prune = True
    block = 512  # periods per call (progress bar granularity)

    def search(self, problem, periods, use_threads=1, progress=None):
        lib = load_library()
        fp = FusedProblem(problem)
        widths, rows, offsets, lengths, profile, overshoot, sum_a2 = fp.templates
        periods = numpy.ascontiguousarray(periods, dtype=numpy.float64)
        n_p = len(periods)
        chi2 = numpy.empty(n_p)
        row = numpy.empty(n_p, dtype=numpy.int64)
        depth = numpy.empty(n_p)
        for start in range(0, n_p, self.block):
            sl = slice(start, min(n_p, start + self.block))
            p_blk = numpy.ascontiguousarray(periods[sl])
            c_blk, r_blk, d_blk = (
                numpy.empty(len(p_blk)),
                numpy.empty(len(p_blk), numpy.int64),
                numpy.empty(len(p_blk)),
            )
            status = lib.tls_search(
                p_blk,
                len(p_blk),
                fp.t,
                fp.y,
                fp.inv_dy2,
                len(fp.t),
                int(fp.uniform_weights),
                fp.time_span,
                problem.transit_depth_min,
                problem.R_star_min,
                problem.R_star_max,
                problem.M_star_min,
                problem.M_star_max,
                tls_constants.FRACTIONAL_TRANSIT_DURATION_MAX,
                widths,
                rows,
                offsets,
                lengths,
                len(widths),
                profile,
                overshoot,
                sum_a2,
                float(problem.T0_search_margin),
                float(tls_constants.SIGNAL_DEPTH),
                int(self.prune),
                int(use_threads),
                c_blk,
                r_blk,
                d_blk,
            )
            if status != 0:
                raise MemoryError("TLS C kernel: allocation failed")
            chi2[sl], row[sl], depth[sl] = c_blk, r_blk, d_blk
            if progress is not None:
                progress(len(p_blk))
        return SearchResult.from_unsorted(periods, chi2, row, depth)

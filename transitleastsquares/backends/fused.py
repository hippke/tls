"""Backend "fused": allocation-free fused period kernel (core_fused.py), numba."""

import os

import numpy

from transitleastsquares.backends.pool import ProcessPoolBackend
from transitleastsquares.core_fused import FusedProblem


class FusedBackend(ProcessPoolBackend):
    name = "fused"
    min_stride = 0  # 0: exact correlation; see FusedBinnedBackend
    pieces = 0  # piecewise-constant templates (B1b); 0: off
    dtype = "float64"  # dot-product precision
    pl = (0,)  # piecewise-linear templates (L3): (min_length, max_stride, eps)
    a2_spread_max = 0.0  # U2 (A2 from the window-mean weight) if spread <= this
    t0_coarsen = 1  # L5: coarse T0 grid with local refinement (1: off)
    scout_every = 0  # L8: every n-th duration scans all phases (0: off)
    ls_depth = False  # B6/L9: least-squares depth per shift (statistic change)

    def prepare(self, problem):
        fp = FusedProblem(problem)
        fp.pieces = self.pieces
        fp.set_binning(self.min_stride)
        fp.set_precision(self.dtype)
        fp.set_pl(
            *self.pl_config(),
            a2_approx=self.use_a2_approx(fp),
            t0_coarsen=self.coarsen(),
            scout_every=self.scouts(),
            ls_depth=self.ls_depth or os.environ.get("TLS_LS_DEPTH") == "1",
        )
        if self.exact:
            fp.set_screen()
        else:
            fp.set_storage(numpy.float32)  # approximate backends (step 50)
        return fp

    def plan(self, state, periods):
        """Walk fold (idea L12): crossover period for the searched range."""
        state.set_walk(periods=periods)

    def coarsen(self):
        """Coarse T0 grid factor (idea L5); environment TLS_T0_COARSEN."""
        import os

        return int(os.environ.get("TLS_T0_COARSEN", self.t0_coarsen))

    def scouts(self):
        """Scout durations (idea L8); environment TLS_SCOUT_EVERY."""
        import os

        return int(os.environ.get("TLS_SCOUT_EVERY", self.scout_every))

    def pl_config(self):
        """(min_length, max_stride, eps); eps can be overridden for
        experiments with the environment variable TLS_PL_EPS."""
        import os

        if len(self.pl) < 3 or "TLS_PL_EPS" not in os.environ:
            return self.pl
        return (*self.pl[:2], float(os.environ["TLS_PL_EPS"]))

    def use_a2_approx(self, fp):
        """Idea U2: A2 from the window-mean weight if the weights are nearly
        uniform (relative std <= a2_spread_max; env TLS_PL_A2_SPREAD_MAX)."""
        import os

        import numpy

        limit = float(os.environ.get("TLS_PL_A2_SPREAD_MAX", self.a2_spread_max))
        w = fp.inv_dy2
        return bool(limit > 0 and numpy.std(w) <= limit * numpy.mean(w))

    def evaluate(self, state, period):
        return state.search(period)


class FusedThreadsBackend(FusedBackend):
    """Same kernel, parallelised with numba threads (prange) instead of processes."""

    name = "fused-threads"
    block = 1024  # periods per call (progress bar granularity)

    def search(self, problem, periods, use_threads=1, progress=None):
        import numba
        import numpy

        from transitleastsquares import tls_constants
        from transitleastsquares.backends import SearchResult
        from transitleastsquares.core_fused import search_periods_fused_parallel

        fp = FusedProblem(problem)
        fp.pieces = self.pieces
        fp.set_binning(self.min_stride)
        fp.set_precision(self.dtype)
        fp.set_pl(
            *self.pl_config(),
            a2_approx=self.use_a2_approx(fp),
            t0_coarsen=self.coarsen(),
            scout_every=self.scouts(),
            ls_depth=self.ls_depth or os.environ.get("TLS_LS_DEPTH") == "1",
        )
        if self.exact:
            fp.set_screen()
        else:
            fp.set_storage(numpy.float32)
        p = problem
        periods = numpy.ascontiguousarray(periods, dtype=float)
        fp.set_walk(periods=periods)
        old = numba.get_num_threads()
        numba.set_num_threads(min(use_threads, numba.config.NUMBA_NUM_THREADS))
        try:
            parts = []
            for start in range(0, len(periods), self.block):
                blk = periods[start : start + self.block]
                parts.append(
                    search_periods_fused_parallel(
                        blk,
                        fp.t,
                        fp.r_in,
                        fp.w_in,
                        fp.uniform_weights,
                        fp.time_span,
                        p.transit_depth_min,
                        p.R_star_min,
                        p.R_star_max,
                        p.M_star_min,
                        p.M_star_max,
                        *fp.templates_kernel,
                        float(p.T0_search_margin),
                        float(tls_constants.SIGNAL_DEPTH),
                        fp.prune,
                        *fp.binning,
                        *fp.pl,
                        int(min(use_threads, numba.config.NUMBA_NUM_THREADS)),
                        fp.invariants,
                        fp.index_dtype,
                        fp.screen,
                        fp.walk,
                    )
                )
                if progress is not None:
                    progress(len(blk))
        finally:
            numba.set_num_threads(old)
        chi2 = numpy.concatenate([x[0] for x in parts])
        rows = numpy.concatenate([x[1] for x in parts])
        depths = numpy.concatenate([x[2] for x in parts])
        return SearchResult.from_unsorted(periods, chi2, rows, depths)


class FusedBinnedBackend(FusedBackend):
    """Approximate: stride-binned correlation for templates with shift stride
    >= 4 (idea B1, see PERFORMANCE_LOG.md)."""

    name = "fused-binned"
    min_stride = 4
    exact = False


class FusedBinned32Backend(FusedBinnedBackend):
    """As fused-binned, with float32 dot products. Experimental, not
    registered: no speed gain in the full kernel (PERFORMANCE_LOG.md step 11)."""

    name = "fused-binned32"
    dtype = "float32"


class FusedPiecesBackend(FusedBinnedBackend):
    """As fused-binned, plus piecewise-constant templates (about 100 pieces)
    for wide templates with shift stride < 4 (idea B1b, experimental)."""

    name = "fused-pieces"
    pieces = 100


class FusedPLBackend(FusedBackend):
    """Approximate: piecewise-linear templates (idea L3, PERFORMANCE_LOG.md
    step 17) for all templates with >= 64 samples, any shift stride; no
    stride bins. AR from about 20 terms on double prefix sums."""

    name = "fused-pl"
    exact = False
    pl = (64, 1 << 62, 2e-2)  # eps 2e-2: PERFORMANCE_LOG step 23
    a2_spread_max = 0.1  # U2 for nearly uniform weights (PERFORMANCE_LOG step 21)
    t0_coarsen = 3  # L5: 3x coarser T0 grid + local refinement (step 24)
    scout_every = 4  # L8: scout durations (step 28)

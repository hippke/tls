"""Backend "fused": allocation-free fused period kernel (core_fused.py), numba."""

from transitleastsquares.backends.pool import ProcessPoolBackend
from transitleastsquares.core_fused import FusedProblem


class FusedBackend(ProcessPoolBackend):
    name = "fused"
    min_stride = 0  # 0: exact correlation; see FusedBinnedBackend
    pieces = 0  # piecewise-constant templates (B1b); 0: off
    dtype = "float64"  # dot-product precision
    pl = (0,)  # piecewise-linear templates (L3): (min_length, max_stride, eps)

    def prepare(self, problem):
        fp = FusedProblem(problem)
        fp.pieces = self.pieces
        fp.set_binning(self.min_stride)
        fp.set_precision(self.dtype)
        fp.set_pl(*self.pl)
        return fp

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
        fp.set_pl(*self.pl)
        p = problem
        periods = numpy.ascontiguousarray(periods, dtype=float)
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
                        fp.y,
                        fp.inv_dy2,
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
    pl = (64, 1 << 62, 1e-2)

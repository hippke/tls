"""Backend "fused": allocation-free fused period kernel (core_fused.py), numba."""

from transitleastsquares.backends.pool import ProcessPoolBackend
from transitleastsquares.core_fused import FusedProblem


class FusedBackend(ProcessPoolBackend):
    name = "fused"

    def prepare(self, problem):
        return FusedProblem(problem)

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
                        *fp.templates,
                        float(p.T0_search_margin),
                        float(tls_constants.SIGNAL_DEPTH),
                        fp.prune,
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

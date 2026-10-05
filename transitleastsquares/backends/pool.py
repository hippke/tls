"""Process-pool parallelisation shared by the CPU backends.

The search data are handed to each worker once (pool initializer; with the
default "fork" start method they are inherited without pickling), and the
periods are sent in chunks. TLS <= 1.33 pickled all arrays for every period.
"""

import multiprocessing
import os

import numpy

from transitleastsquares.backends import SearchBackend, SearchResult

_WORKER = None


def _init_worker(backend, state):
    global _WORKER
    _WORKER = (backend, state)


def _search_chunk(periods):
    backend, state = _WORKER
    return [backend.evaluate(state, p) for p in periods]


def chunks(periods, use_threads, per_thread=16):
    """Split the periods into about use_threads * per_thread chunks."""
    n_chunks = max(1, min(len(periods), use_threads * per_thread))
    return [c for c in numpy.array_split(numpy.asarray(periods), n_chunks) if len(c)]


class ProcessPoolBackend(SearchBackend):
    """Backend skeleton: subclasses implement prepare() and evaluate()."""

    def prepare(self, problem):
        """Per-search precomputation (done once, in the parent process)."""
        return problem

    def evaluate(self, state, period):
        """Return (period, chi2, row, depth) for one trial period."""
        raise NotImplementedError

    def search(self, problem, periods, use_threads=1, progress=None):
        state = self.prepare(problem)
        results = []

        def collect(block):
            results.extend(block)
            if progress is not None:
                progress(len(block))

        if use_threads > 1:
            ctx = multiprocessing.get_context(
                os.environ.get("TLS_MP_START_METHOD") or None
            )
            pool = ctx.Pool(
                processes=use_threads, initializer=_init_worker, initargs=(self, state)
            )
            try:
                for block in pool.imap_unordered(
                    _search_chunk, chunks(periods, use_threads)
                ):
                    collect(block)
                pool.close()  # close + join is much faster than terminate (0.6 s)
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
        else:
            for period in periods:
                collect([self.evaluate(state, period)])

        periods_out, chi2, rows, depths = zip(*results)
        return SearchResult.from_unsorted(periods_out, chi2, rows, depths)

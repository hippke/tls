"""Process-pool parallelisation shared by the CPU backends.

The search data are handed to each worker once (pool initializer; with the
default "fork" start method they are inherited without pickling), and the
periods are sent in chunks. TLS <= 1.33 pickled all arrays for every period.

One pool serves all search tasks of a power() call (the unbinned light curve
and its pre-binned copies, see main._search_plan): starting and stopping a
pool costs ~20-35 ms, and pre-binning used to create one pool per copy. The
chunks shrink towards the end of the search so that no worker idles while
another finishes a large last chunk. The parent may defer the final join of
the workers until it has computed the statistics (power(): finish()).
"""

import multiprocessing
import os

import numpy

from transitleastsquares.backends import SearchBackend, SearchResult

_WORKER = None


def _init_worker(backend, states):
    global _WORKER
    _WORKER = (backend, states)


def _search_chunk(periods):
    """Single-task chunk (kept for callers of the pre-round-10 interface)."""
    backend, state = _WORKER
    return [backend.evaluate(state, p) for p in periods]


def _search_task_chunk(item):
    task, periods = item
    backend, states = _WORKER
    state = states[task]
    return task, [backend.evaluate(state, p) for p in periods]


def chunks(periods, use_threads, per_thread=16):
    """Split the periods into about use_threads * per_thread chunks."""
    n_chunks = max(1, min(len(periods), use_threads * per_thread))
    return [c for c in numpy.array_split(numpy.asarray(periods), n_chunks) if len(c)]


def schedule(tasks, use_threads, per_thread=16, tail=8, guided=4):
    """Chunks [(task index, periods)] for all tasks [(periods, cost per period)].

    Chunk sizes (in units of estimated work) are the equal chunks of
    `chunks` (total / (use_threads * per_thread)) while much work remains,
    then shrink as remaining / (guided * use_threads), down to 1/tail of the
    regular size: the last chunks are small, so the workers finish together
    (load balance). Every period appears exactly once; the order within a
    task is kept. Tasks with the highest cost per period come first."""
    total = sum(len(p) * c for p, c in tasks)
    if total <= 0:
        return []
    regular = total / max(1, use_threads * per_thread)
    smallest = regular / tail
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i][1])
    out = []
    remaining = total
    for i in order:
        periods, cost = tasks[i]
        periods = numpy.asarray(periods)
        start = 0
        while start < len(periods):
            size = min(regular, max(smallest, remaining / (guided * use_threads)))
            count = max(1, int(round(size / cost)))
            stop = min(len(periods), start + count)
            out.append((i, periods[start:stop]))
            remaining -= (stop - start) * cost
            start = stop
    return out


class ProcessPoolBackend(SearchBackend):
    """Backend skeleton: subclasses implement prepare() and evaluate()."""

    def prepare(self, problem):
        """Per-search precomputation (done once, in the parent process)."""
        return problem

    def plan(self, state, periods):
        """Optional set-up that depends on the searched periods (in the parent
        process, after prepare)."""

    def evaluate(self, state, period):
        """Return (period, chi2, row, depth) for one trial period."""
        raise NotImplementedError

    def cost(self, state):
        """Relative cost per period of a task (only used for scheduling)."""
        return float(len(getattr(state, "t", ()))) or 1.0

    def search(self, problem, periods, use_threads=1, progress=None):
        return self._search_tasks([(problem, periods)], use_threads, progress)[0]

    def search_many(self, tasks, use_threads=1, progress=None, defer_join=False):
        """[(problem, periods)] -> [SearchResult], one process pool for all.
        defer_join: keep the closed pool until finish() (the workers exit
        while the caller continues). Subclasses that override search() keep
        their per-task semantics."""
        if type(self).search is not ProcessPoolBackend.search:
            return [
                self.search(
                    problem, periods, use_threads=use_threads, progress=progress
                )
                for problem, periods in tasks
            ]
        return self._search_tasks(tasks, use_threads, progress, defer_join)

    def finish(self):
        """Join pools whose join was deferred (search_many(defer_join=True))."""
        pending = getattr(self, "_pending", None) or []
        self._pending = []
        for pool in pending:
            pool.join()

    def _search_tasks(self, tasks, use_threads=1, progress=None, defer_join=False):
        states = []
        for problem, periods in tasks:
            state = self.prepare(problem)
            self.plan(state, periods)
            states.append(state)
        results = [[] for _ in tasks]

        if use_threads > 1:
            work = schedule(
                [(periods, self.cost(s)) for (_, periods), s in zip(tasks, states)],
                use_threads,
            )
            ctx = multiprocessing.get_context(
                os.environ.get("TLS_MP_START_METHOD") or None
            )
            pool = ctx.Pool(
                processes=use_threads,
                initializer=_init_worker,
                initargs=(self, states),
            )
            joined = False
            try:
                for task, block in pool.imap_unordered(_search_task_chunk, work):
                    results[task].extend(block)
                    if progress is not None:
                        progress(len(block))
                pool.close()  # close + join is much faster than terminate (0.6 s)
                if defer_join:
                    self._pending = getattr(self, "_pending", None) or []
                    self._pending.append(pool)
                    joined = True
            except BaseException:
                pool.terminate()
                raise
            finally:
                if not joined:
                    pool.join()
        else:
            for i, ((_, periods), state) in enumerate(zip(tasks, states)):
                for period in periods:
                    results[i].append(self.evaluate(state, period))
                    if progress is not None:
                        progress(1)

        out = []
        for block in results:
            periods_out, chi2, rows, depths = zip(*block)
            out.append(SearchResult.from_unsorted(periods_out, chi2, rows, depths))
        return out

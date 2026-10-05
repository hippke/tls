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

Persistent workers (round 11): for the built-in backends the pool is kept
alive between power() calls (tls_constants.WORKER_IDLE_TIMEOUT seconds of
idleness, and until interpreter exit; TLS_PERSISTENT_POOL=0 disables it).
Each call then ships its (backend, states) once through a temporary pickle
file that every worker loads on its first chunk of that call, together with
a snapshot of tls_constants and the TLS_* environment variables, so the
workers see the same configuration as the parent. A new pool costs ~15 ms
with "fork" but ~3 s with "spawn" (macOS, Windows) or "forkserver" (Linux
from Python 3.14), where every worker imports TLS and loads the kernels.
"""

import atexit
import mmap
import multiprocessing
import os
import pickle
import tempfile
import threading
import uuid

import numpy

from transitleastsquares import tls_constants
from transitleastsquares.backends import SearchBackend, SearchResult

_WORKER = None
_CACHE = None  # in a persistent worker: (token, backend, states)


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


def _config_snapshot():
    """Upper-case constants of tls_constants and TLS_* environment variables,
    applied in persistent workers so they match the parent at call time."""
    consts = {}
    for k, v in vars(tls_constants).items():
        if k.isupper() and isinstance(v, (bool, int, float, str, tuple, list)):
            consts[k] = v
    env = {k: v for k, v in os.environ.items() if k.startswith("TLS_")}
    return consts, env


def _apply_config(config):
    consts, env = config
    for k, v in consts.items():
        setattr(tls_constants, k, v)
    for k in [k for k in os.environ if k.startswith("TLS_") and k not in env]:
        del os.environ[k]
    os.environ.update(env)


_ALIGN = 64


def _write_payload(fh, payload):
    """Pickle protocol 5 with the array buffers out of band, each at an
    aligned offset of the file: workers map the file copy-on-write, so all of
    them share one copy of the input arrays in the page cache (as a forked
    pool shares the parent's pages) instead of unpickling private copies."""
    buffers = []
    data = pickle.dumps(payload, protocol=5, buffer_callback=buffers.append)
    raws = [b.raw() for b in buffers]
    offsets = []
    pos = 0
    for r in raws:
        pos = (pos + _ALIGN - 1) // _ALIGN * _ALIGN
        offsets.append((pos, r.nbytes))
        pos += r.nbytes
    header = pickle.dumps((data, offsets), protocol=5)
    start = (8 + len(header) + _ALIGN - 1) // _ALIGN * _ALIGN
    fh.write(len(header).to_bytes(8, "little"))
    fh.write(header)
    fh.write(b"\0" * (start - 8 - len(header)))
    written = 0
    for (off, nbytes), r in zip(offsets, raws):
        fh.write(b"\0" * (off - written))
        fh.write(r)
        written = off + nbytes


def _read_payload(path):
    with open(path, "rb") as fh:
        if os.name == "nt":  # mapped files cannot be deleted there: plain copy
            content = fh.read()
            mm = None
        else:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
            content = mm
    n = int.from_bytes(content[:8], "little")
    data, offsets = pickle.loads(content[8 : 8 + n])
    start = (8 + n + _ALIGN - 1) // _ALIGN * _ALIGN
    view = memoryview(content)
    buffers = [view[start + off : start + off + nb] for off, nb in offsets]
    return pickle.loads(data, buffers=buffers)


def _persistent_chunk(item):
    """Chunk of a persistent pool: load this call's payload once per worker."""
    global _CACHE
    token, path, task, periods = item
    if _CACHE is None or _CACHE[0] != token:
        _CACHE = None  # release the previous call's states first
        config, backend, states = _read_payload(path)
        _apply_config(config)
        _CACHE = (token, backend, states)
    _, backend, states = _CACHE
    return task, [backend.evaluate(states[task], p) for p in periods]


class _Persistent:
    """The one persistent pool of this process (key: start method, size,
    backend class), with an idle timer."""

    lock = threading.Lock()
    pool = None
    key = None
    timer = None
    busy = 0
    registered = False

    @classmethod
    def acquire(cls, ctx, processes, backend):
        key = (ctx.get_start_method(), processes, type(backend))
        with cls.lock:
            if cls.timer is not None:
                cls.timer.cancel()
                cls.timer = None
            if cls.pool is not None and (cls.key != key or cls.busy):
                if cls.busy:  # concurrent call from another thread: own pool
                    return None
                cls._close_locked()
            if cls.pool is None:
                cls.pool = ctx.Pool(processes=processes)
                cls.key = key
                if not cls.registered:
                    atexit.register(shutdown_workers)
                    cls.registered = True
            cls.busy += 1
            return cls.pool

    @classmethod
    def release(cls, broken=False):
        with cls.lock:
            cls.busy -= 1
            if broken:
                if cls.pool is not None:
                    cls.pool.terminate()
                    cls.pool.join()
                cls.pool = cls.key = None
                return
            timeout = float(getattr(tls_constants, "WORKER_IDLE_TIMEOUT", 0) or 0)
            if cls.pool is not None and timeout > 0 and not cls.busy:
                cls.timer = threading.Timer(timeout, cls.idle)
                cls.timer.daemon = True
                cls.timer.start()

    @classmethod
    def idle(cls):
        with cls.lock:
            if not cls.busy:
                cls._close_locked()

    @classmethod
    def _close_locked(cls):
        if cls.pool is not None:
            cls.pool.close()
            cls.pool.join()
        cls.pool = cls.key = None


def shutdown_workers():
    """Stop the persistent worker processes now (they also stop after
    tls_constants.WORKER_IDLE_TIMEOUT idle seconds and at exit)."""
    with _Persistent.lock:
        if _Persistent.timer is not None:
            _Persistent.timer.cancel()
            _Persistent.timer = None
        if not _Persistent.busy:
            _Persistent._close_locked()


def persistent_enabled(backend):
    """Persistent workers for the built-in backends (their classes and
    configuration are known to the workers), unless disabled."""
    if os.environ.get("TLS_PERSISTENT_POOL", "1") == "0":
        return False
    if float(getattr(tls_constants, "WORKER_IDLE_TIMEOUT", 0) or 0) <= 0:
        return False
    return type(backend).__module__.startswith("transitleastsquares.")


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

    def __getstate__(self):  # pending pools stay in the parent
        state = dict(self.__dict__)
        state.pop("_pending", None)
        return state

    def _search_persistent(self, pool, work, states, results, progress):
        token = uuid.uuid4().hex
        fd, path = tempfile.mkstemp(prefix="tls_search_", suffix=".pkl")
        broken = True
        try:
            with os.fdopen(fd, "wb") as fh:
                _write_payload(fh, (_config_snapshot(), self, states))
            items = [(token, path, task, periods) for task, periods in work]
            for task, block in pool.imap_unordered(_persistent_chunk, items):
                results[task].extend(block)
                if progress is not None:
                    progress(len(block))
            broken = False
        finally:
            _Persistent.release(broken=broken)
            try:
                os.unlink(path)
            except OSError:
                pass

    def _search_tasks(self, tasks, use_threads=1, progress=None, defer_join=False):
        states = []
        for problem, periods in tasks:
            state = self.prepare(problem)
            self.plan(state, periods)
            states.append(state)
        results = [[] for _ in tasks]

        pool = None
        if use_threads > 1:
            work = schedule(
                [(periods, self.cost(s)) for (_, periods), s in zip(tasks, states)],
                use_threads,
            )
            ctx = multiprocessing.get_context(
                os.environ.get("TLS_MP_START_METHOD") or None
            )
            if persistent_enabled(self):
                pool = _Persistent.acquire(ctx, use_threads, self)
        if pool is not None:
            self._search_persistent(pool, work, states, results, progress)
        elif use_threads > 1:
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

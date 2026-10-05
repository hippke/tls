"""First-run compilation of the numba kernels.

numba compiles the search kernels on their first use and caches the machine
code on disk (``cache=True``), so this happens once per installation (and
after updates of TLS or numba). ``ensure_compiled`` runs a tiny search in the
calling process before the real search: on a cold cache it tells the user
what is going on; on a warm cache it only loads the cached code (~0.1 s), so
the worker processes forked afterwards inherit it instead of each loading or
compiling it themselves.
"""

import time

import numpy

_DONE = set()
# built-in numba backends (None = the default, possibly set via TLS_BACKEND)
NUMBA_BACKENDS = (
    "numba",
    "fused",
    "fused-threads",
    "fused-binned",
    "fused-pieces",
    "fused-pl",
)

FIRST_RUN_MESSAGE = (
    "Welcome to TLS! This is its first run on this system (or the first after "
    "an update).\nThe numba just-in-time compiler is now translating the search "
    "code into machine code\nfor this computer. This takes up to about 10 "
    "seconds and happens only once;\nlater runs start right away. Compiling..."
)


def _kernel_cached():
    """True if the main search kernel is in numba's on-disk cache (best effort;
    uses numba internals, so any failure counts as "cached" = no message)."""
    try:
        from transitleastsquares.core_fused import search_period_fused

        index = search_period_fused._cache._cache_file._load_index()
        return len(index) > 0
    except Exception:
        return True


def ensure_compiled(backend=None, verbose=True):
    """Compile (cold cache) or load (warm cache) the kernels of `backend` in
    this process, once. Prints FIRST_RUN_MESSAGE on a cold cache if verbose."""
    if backend is not None and backend not in NUMBA_BACKENDS:
        return  # other or user-supplied backends: do not call them behind their back
    key = backend
    if key in _DONE:
        return
    _DONE.add(key)  # also guards against recursion through power() below
    cold = not _kernel_cached()
    if cold and verbose:
        print(FIRST_RUN_MESSAGE, flush=True)
    t_start = time.perf_counter()
    from transitleastsquares.main import transitleastsquares

    rng = numpy.random.default_rng(0)
    t = numpy.linspace(0, 20, 3000)
    y = 1 + rng.normal(0, 1e-3, len(t))
    y[(t % 5.0) < 0.1] -= 3e-3
    transitleastsquares(t, y, verbose=False).power(
        backend=backend,
        use_threads=1,
        period_min=4.5,
        period_max=5.5,
        show_progress_bar=False,
        verbose=False,
    )
    # statistics helpers not reached by the tiny search above (short spectrum)
    from transitleastsquares.helpers import running_median

    running_median(numpy.linspace(0.0, 1.0, 9), 3)
    if cold and verbose:
        print(f"...done ({time.perf_counter() - t_start:.1f} s).", flush=True)


def warmup(verbose=True):
    """Compile the kernels now (e.g. right after installation, or when
    building a container image)."""
    ensure_compiled(None, verbose=verbose)

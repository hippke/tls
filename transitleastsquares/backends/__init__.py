"""Pluggable compute backends for the performance-critical parts of TLS.

A backend implements two operations:

* ``search(problem, periods, use_threads, progress)`` -- the period search:
  for every trial period, the lowest chi2 over all trial durations and phase
  shifts, plus the template row and depth of that best fit;
* ``fit_T0(...)`` -- the final mid-transit time fit at the best period.

The reference backend ``"numba"`` (``backends/numba_reference.py``) is the
original TLS implementation. New backends (other libraries, C extensions,
different parallelisation, ...) subclass :class:`SearchBackend`, register via
:func:`register_backend`, and are selected with ``power(backend="name")`` or
the environment variable ``TLS_BACKEND``. They must reproduce the reference
(tests/test_core_oracle.py, tests/test_golden.py, tests/test_backends.py).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy

from transitleastsquares import tls_constants


@dataclass
class SearchProblem:
    """Everything a backend needs to search a light curve (all arrays float64).

    t, y, dy            cleaned time series (dy normalised to mean 1)
    lc_arr              object array: in-transit template per row (SIGNAL_DEPTH deep)
    lc_cache_overview   structured array (duration, width_in_samples, overshoot)
    transit_depth_min   shallowest depth that is fit
    R/M_star_min/max    stellar limits that restrict the durations per period
    T0_search_margin    phase-shift margin during the search (fraction of width)
    """

    t: numpy.ndarray
    y: numpy.ndarray
    dy: numpy.ndarray
    lc_arr: numpy.ndarray
    lc_cache_overview: numpy.ndarray
    transit_depth_min: float
    R_star_min: float
    R_star_max: float
    M_star_min: float
    M_star_max: float
    T0_search_margin: float


@dataclass
class SearchResult:
    """Per-period results, sorted by ascending period."""

    periods: numpy.ndarray
    chi2: numpy.ndarray
    rows: numpy.ndarray
    depths: numpy.ndarray

    @classmethod
    def from_unsorted(cls, periods, chi2, rows, depths):
        periods = numpy.asarray(periods)
        order = numpy.argsort(periods)
        return cls(
            periods=periods[order],
            chi2=numpy.asarray(chi2)[order],
            rows=numpy.asarray(rows)[order],
            depths=numpy.asarray(depths)[order],
        )


class SearchBackend:
    """Base class. Subclasses must implement :meth:`search`."""

    name = "base"
    exact = True  # reproduces the reference statistic up to rounding

    def search(
        self,
        problem: SearchProblem,
        periods: numpy.ndarray,
        use_threads: int = 1,
        progress: Optional[Callable[[int], None]] = None,
    ) -> SearchResult:
        raise NotImplementedError

    def fit_T0(
        self, signal, depth, t, y, dy, period, T0_fit_margin, show_progress_bar, verbose
    ) -> float:
        from transitleastsquares.stats import final_T0_fit

        return final_T0_fit(
            signal=signal,
            depth=depth,
            t=t,
            y=y,
            dy=dy,
            period=period,
            T0_fit_margin=T0_fit_margin,
            show_progress_bar=show_progress_bar,
            verbose=verbose,
        )

    def __repr__(self):
        return f"<TLS backend {self.name!r}>"


# name -> backend class / factory, or "module:attribute" string (lazy import)
_REGISTRY: Dict[str, Union[str, Callable[[], SearchBackend]]] = {
    "numba": "transitleastsquares.backends.numba_reference:NumbaBackend",
    "fused": "transitleastsquares.backends.fused:FusedBackend",
    "fused-threads": "transitleastsquares.backends.fused:FusedThreadsBackend",
    "c": "transitleastsquares.backends.c_kernel:CBackend",
    "fused-binned": "transitleastsquares.backends.fused:FusedBinnedBackend",
    "fused-pieces": "transitleastsquares.backends.fused:FusedPiecesBackend",
}


def register_backend(name: str, factory) -> None:
    """Register a backend class/factory (or a lazy "module:attr" string)."""
    _REGISTRY[name] = factory


def available_backends():
    return sorted(_REGISTRY)


def get_backend(backend=None) -> SearchBackend:
    """Return a backend instance from an instance, a name, or the default
    (environment variable TLS_BACKEND, else tls_constants.DEFAULT_BACKEND)."""
    if isinstance(backend, SearchBackend):
        return backend
    name = backend or os.environ.get("TLS_BACKEND") or tls_constants.DEFAULT_BACKEND
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown TLS backend {name!r}. "
            f"Available: {', '.join(available_backends())}"
        )
    factory = _REGISTRY[name]
    if isinstance(factory, str):
        module, attr = factory.split(":")
        factory = getattr(importlib.import_module(module), attr)
    return factory()

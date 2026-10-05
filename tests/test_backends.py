"""Backend registry and interface."""

import numpy
import pytest

from transitleastsquares import (
    SearchBackend,
    available_backends,
    get_backend,
    register_backend,
    transitleastsquares,
)
from transitleastsquares.backends import _REGISTRY
from transitleastsquares.backends.fused import FusedBackend
from transitleastsquares.backends.numba_reference import NumbaBackend

Q = dict(show_progress_bar=False, verbose=False)


def lc():
    rng = numpy.random.default_rng(7)
    t = numpy.arange(0, 30, 0.01)
    y = 1 + rng.normal(0, 3e-4, len(t))
    ph = (t - 1.1) % 3.3
    y[(ph < 0.06) | (ph > 3.3 - 0.06)] -= 2e-3
    return t, y


def test_default_backend():
    assert "numba" in available_backends()
    assert isinstance(get_backend(), FusedBackend)  # default
    assert isinstance(get_backend("numba"), NumbaBackend)
    b = NumbaBackend()
    assert get_backend(b) is b


def test_env_var(monkeypatch):
    monkeypatch.setenv("TLS_BACKEND", "nonexistent")
    with pytest.raises(ValueError):
        get_backend()
    monkeypatch.setenv("TLS_BACKEND", "numba")
    assert isinstance(get_backend(), NumbaBackend)
    monkeypatch.delenv("TLS_BACKEND")
    assert isinstance(get_backend(), FusedBackend)


class CountingBackend(NumbaBackend):
    """Example custom backend: delegates to the reference and counts calls."""

    name = "counting"
    calls = {"search": 0, "fit_T0": 0}

    def search(self, problem, periods, use_threads=1, progress=None):
        CountingBackend.calls["search"] += 1
        return super().search(problem, periods, use_threads, progress)

    def fit_T0(self, *args, **kwargs):
        CountingBackend.calls["fit_T0"] += 1
        return super().fit_T0(*args, **kwargs)


def test_custom_backend_identical_results():
    register_backend("counting", CountingBackend)
    try:
        t, y = lc()
        kw = dict(period_min=3, period_max=3.6, use_threads=1, **Q)
        ref = transitleastsquares(t, y).power(backend="numba", **kw)
        cur = transitleastsquares(t, y).power(backend="counting", **kw)
        assert CountingBackend.calls == {"search": 1, "fit_T0": 1}
        numpy.testing.assert_array_equal(ref.power, cur.power)
        assert ref.T0 == cur.T0 and ref.period == cur.period
        cur2 = transitleastsquares(t, y).power(backend=CountingBackend(), **kw)
        numpy.testing.assert_array_equal(ref.power, cur2.power)
    finally:
        _REGISTRY.pop("counting", None)


def test_threads_identical():
    t, y = lc()
    kw = dict(period_min=3, period_max=3.6, **Q)
    a = transitleastsquares(t, y).power(use_threads=1, **kw)
    b = transitleastsquares(t, y).power(use_threads=4, **kw)
    numpy.testing.assert_array_equal(a.power, b.power)
    numpy.testing.assert_array_equal(a.chi2, b.chi2)
    assert a.T0 == b.T0


def test_base_class_is_abstract():
    with pytest.raises(NotImplementedError):
        SearchBackend().search(None, numpy.array([1.0]))


def test_global_random_state_untouched():
    t, y = lc()
    numpy.random.seed(123)
    expected = numpy.random.random()
    numpy.random.seed(123)
    transitleastsquares(t, y).power(period_min=3, period_max=3.6, use_threads=1, **Q)
    assert numpy.random.random() == expected


def test_T0_search_margin():
    t, y = lc()
    kw = dict(period_min=3, period_max=3.6, use_threads=1, **Q)
    ref = transitleastsquares(t, y).power(**kw)
    same = transitleastsquares(t, y).power(T0_search_margin=0.01, **kw)
    numpy.testing.assert_array_equal(ref.chi2, same.chi2)
    fine = transitleastsquares(t, y).power(T0_search_margin=0, **kw)
    assert numpy.all(fine.chi2 <= ref.chi2 + 1e-9)  # denser search: never worse

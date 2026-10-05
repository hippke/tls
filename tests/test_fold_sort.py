"""core_fused.fold_sort must give exactly the stable (mergesort) phase order."""

import numpy
import pytest

from transitleastsquares.core import foldfast
from transitleastsquares.core_fused import fold_sort


def reference(t, y, w, period):
    ph = foldfast(t, period)  # as in core.search_period
    order = numpy.argsort(ph, kind="mergesort")
    return y[order], w[order]


CASES = {
    "random_times": lambda rng: numpy.sort(rng.uniform(0, 100, 5000)),
    "regular_cadence": lambda rng: numpy.arange(20000) * 0.02,
    "duplicates": lambda rng: numpy.repeat(numpy.arange(3000) * 0.03, 3),
    "unsorted": lambda rng: rng.uniform(-50, 50, 4000),
    "gaps": lambda rng: numpy.concatenate(
        [numpy.arange(0, 10, 0.01), numpy.arange(50, 60, 0.01)]
    ),
    "clustered": lambda rng: rng.uniform(0, 1e-5, 5000),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("period", [0.37, 1.0, 2.0, 3.3, 1.0 + 1e-9, 41.0])
def test_fold_sort_matches_mergesort(case, period):
    rng = numpy.random.default_rng(0)
    t = CASES[case](rng)
    y = rng.normal(1, 1e-3, len(t))
    w = rng.uniform(0.5, 2, len(t))
    f1, w1 = fold_sort(t, y, w, period)
    f2, w2 = reference(t, y, w, period)
    numpy.testing.assert_array_equal(f1, f2)
    numpy.testing.assert_array_equal(w1, w2)


@pytest.mark.parametrize("case", list(CASES))
def test_fold_sort_without_weights(case):
    """move_w=False (uniform weights): same flux order, weights untouched."""
    from transitleastsquares.core_fused import fold_sort_into

    rng = numpy.random.default_rng(1)
    t = CASES[case](rng)
    n = len(t)
    y = rng.normal(1, 1e-3, n)
    w = numpy.ones(n)
    flux = numpy.empty(n)
    w_out = numpy.full(n, -7.0)
    fold_sort_into(
        t,
        y,
        w,
        1.7,
        numpy.empty(n),
        numpy.empty(n + 1, dtype=numpy.int64),
        numpy.empty(n, dtype=numpy.int64),
        numpy.empty(n),
        flux,
        w_out,
        False,
    )
    numpy.testing.assert_array_equal(flux, reference(t, y, w, 1.7)[0])
    assert numpy.all(w_out == -7.0)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("move_w", [True, False])
@pytest.mark.parametrize("dtype", [numpy.int32, numpy.int64])
def test_fold_sort_integer_workspaces(case, move_w, dtype):
    """Compact and wide storage preserve stable ties and the clustered fallback."""
    from transitleastsquares.core_fused import fold_sort_into

    rng = numpy.random.default_rng(2)
    t = CASES[case](rng)
    n = len(t)
    y = rng.normal(1, 1e-3, n)
    weights = rng.uniform(0.5, 2.0, n)
    flux = numpy.empty(n)
    w_out = numpy.full(n, -7.0)
    phases, keys = numpy.empty(n), numpy.empty(n)
    counts, bucket = numpy.empty(n + 1, dtype=dtype), numpy.empty(n, dtype=dtype)
    for period in [0.37, 3.3, 41.0]:
        fold_sort_into(
            t, y, weights, period, phases, counts, bucket, keys, flux, w_out, move_w
        )
        expected_f, expected_w = reference(t, y, weights, period)
        numpy.testing.assert_array_equal(flux, expected_f)
        if move_w:
            numpy.testing.assert_array_equal(w_out, expected_w)
        else:
            assert numpy.all(w_out == -7.0)


def test_sort_workspace_capacity_boundary():
    """The final count can equal N, including at the signed-32-bit boundary."""
    from transitleastsquares.core_fused import sort_index_dtype

    limit = numpy.iinfo(numpy.int32).max
    for n in [1, 100_000, limit, limit + 1]:
        dtype = sort_index_dtype(n)
        assert int(dtype(n)) == n
        assert numpy.dtype(dtype).itemsize == (4 if n <= limit else 8)

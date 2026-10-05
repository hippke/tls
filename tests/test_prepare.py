"""Merged preparation must equal stable phase sorting plus literal prefixes."""

import numpy
import pytest
from test_core_oracle import cache

from transitleastsquares import tls_constants as C
from transitleastsquares.backends import SearchProblem
from transitleastsquares.core import foldfast
from transitleastsquares.core_fused import FusedProblem


@pytest.mark.parametrize("kind", ["regular", "shifted", "duplicates", "unsorted"])
@pytest.mark.parametrize("ratio", [0.03, 0.35, 0.5, 0.8, 1.5])
@pytest.mark.parametrize("weight_mode", ["uniform", "u2", "weighted"])
def test_preparation_matches_stable_prefixes(kind, ratio, weight_mode):
    rng = numpy.random.default_rng(404)
    t = numpy.arange(1200) * 0.025
    if kind == "shifted":
        t += 13.0
    elif kind == "duplicates":
        t = numpy.repeat(t[::2], 2)
    elif kind == "unsorted":
        t = rng.permutation(t)
    y = 1 + rng.normal(0, 1e-3, len(t))
    dy = numpy.ones(len(t))
    if weight_mode != "uniform":
        dy += rng.uniform(-0.03, 0.03, len(t))
    ov, lc = cache(t, y, "default")
    fp = FusedProblem(
        SearchProblem(
            t, y, dy, lc, ov, 1e-5,
            C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX, 0.01,
        )
    )
    fp.set_pl(64, 1 << 62, 0.02, a2_approx=weight_mode == "u2")
    period = ratio * numpy.ptp(t)
    fp.search(period)
    order = numpy.argsort(foldfast(t, period), kind="mergesort")
    maxw = int(fp.templates[0][-1])
    maxw += maxw % 2
    indices = numpy.concatenate((order, order[:maxw]))
    f = y[indices]
    w = fp.inv_dy2[indices]
    rw = (1 - f) * w

    def prefix(values):
        return numpy.concatenate(([0.0], numpy.cumsum(values)))

    expected = [
        prefix(f),
        prefix((1 - f) ** 2 * w),
        prefix(w),
        prefix(rw),
        prefix(prefix(rw - fp.input_means[0])),
        prefix(prefix(w - fp.input_means[1])),
    ]
    active = [True, True, weight_mode != "uniform", True, True,
              weight_mode == "weighted"]
    pos = 4 * len(t)
    for values, used in zip(expected, active):
        if used:
            numpy.testing.assert_allclose(
                fp.workspace()[0][pos : pos + len(values)], values,
                rtol=1e-12, atol=1e-12,
            )
        pos += len(values)

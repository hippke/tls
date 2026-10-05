"""Certificates checked against direct correlations, including cancellation."""

import numpy
import pytest
from test_core_oracle import cache, make_case

from transitleastsquares import tls_constants as C
from transitleastsquares.backends import SearchProblem
from transitleastsquares.core_fused import (
    SCREEN_PROXY_EPS,
    FusedProblem,
    pl_fit,
    pl_templates,
    proxy_tolerance,
    screen_round_budget,
    screen_templates,
)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("shape", ["curved", "box"])
@pytest.mark.parametrize("length", [128, 1024])
@pytest.mark.parametrize("spike", [False, True])
@pytest.mark.parametrize("tol", [0.1, 0.01, SCREEN_PROXY_EPS])
def test_correlation_certificate(weighted, shape, length, spike, tol):
    rng = numpy.random.default_rng(443)
    start = 50000
    a = (
        numpy.full(length, 0.5) if shape == "box"
        else 0.5 * numpy.sin(numpy.linspace(0, numpy.pi, length)) ** 0.3
    )
    templates = (
        numpy.array([length]), numpy.array([0]), numpy.array([0]),
        numpy.array([length]), a, numpy.ones(1), numpy.array([numpy.dot(a, a)]),
    )
    pl = pl_templates(templates, 0.01, 1 << 62, 1 << 62, 0.02)
    y = 1 - rng.normal(1e-4, 1e-10 if spike else 1e-3, start + length)
    if spike:
        y[17] = 1001.0
    r = 1 - y
    w = numpy.exp(rng.normal(0, 1, len(y))) if weighted else numpy.ones(len(y))
    rw = r * w
    mu, muw = numpy.mean(rw), numpy.mean(w)
    budget = screen_round_budget(y, w, (mu, muw), length)
    s = screen_templates(templates, pl, 0.02, weighted, tol, round_budget=budget)
    assert s is not None

    def double_prefix(x):
        first = numpy.concatenate(([0.0], numpy.cumsum(x)))
        return numpy.concatenate(([0.0], numpy.cumsum(first)))

    dd, dw = double_prefix(rw - mu), double_prefix(w - muw)
    count, off = s[0][0], s[1][0]
    proxy = (
        numpy.dot(s[5][off : off + count], dd[start + s[4][off : off + count]])
        + mu * s[6][0]
    )
    energy = numpy.dot(r[start:] ** 2, w[start:])
    upper_ar = proxy + s[10][0] + numpy.sqrt(s[8][0] * w.max() * (energy + s[12]))
    ar = numpy.dot(a, rw[start:])
    assert ar <= upper_ar
    a2 = numpy.dot(a * a, w[start:])
    lower_a2 = a2
    if weighted:
        count, off = s[2][0], s[3][0]
        lower_a2 = (
            numpy.dot(s[5][off : off + count], dw[start + s[4][off : off + count]])
            + muw * s[7][0] - s[9][0] * w[start:].sum() - s[11][0]
        )
        assert lower_a2 <= a2
    for k in [1e-4, 0.01, 1.0]:
        assert 2 * k * ar - k * k * a2 <= 2 * k * upper_ar - k * k * lower_a2


@pytest.mark.parametrize("with_dy", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_screen_preserves_scouts_and_refinement(seed, with_dy):
    t, y, dy = make_case(seed, n=3000, span=30, with_dy=with_dy)
    ov, lc = cache(t, y, "ecc_sqrt")
    fp = FusedProblem(
        SearchProblem(
            t, y, dy, lc, ov, 1e-5,
            C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX, 0.01,
        )
    )
    fp.set_pl(64, 1 << 62, 0.02, a2_approx=with_dy, t0_coarsen=3, scout_every=4)
    periods = [0.7, 1.2, 2.3, 4.2, 10.0]
    expected = numpy.array([fp.search(p) for p in periods])
    fp.set_screen()
    assert fp.screen is not None
    actual = numpy.array([fp.search(p) for p in periods])
    numpy.testing.assert_array_equal(actual[:, 2:], expected[:, 2:])
    numpy.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-12)


def test_pl_target_certificate_uses_evaluated_shapes():
    a = 0.5 * numpy.sin(numpy.linspace(0, numpy.pi, 257)) ** 0.3
    templates = (
        numpy.array([len(a)]), numpy.array([0]), numpy.array([0]),
        numpy.array([len(a)]), a, numpy.ones(1), numpy.array([numpy.dot(a, a)]),
    )
    pl = pl_templates(templates, 0.01, 64, 1 << 62, 0.02)
    s = screen_templates(templates, pl, 0.02, True)
    assert s is not None
    target = pl_fit(a, 0.02)[3]
    target2 = pl_fit(a * a, 0.02)[3]
    proxy = pl_fit(target, 0.1)[3]
    proxy2 = pl_fit(target2, 0.1)[3]
    # The second fine fit is of a², not the square of the first fine fit.
    assert not numpy.allclose(target2, target * target, rtol=1e-5)
    assert numpy.linalg.norm(target - proxy) ** 2 <= s[8][0]
    assert numpy.max(numpy.abs(target2 - proxy2)) <= s[9][0]


def test_nonpositive_scale_keeps_original_path():
    """The AR upper bound is used only for positive scaling, not its reverse."""
    t, y, dy = make_case(7, n=3000, span=30, with_dy=True)
    ov, lc = cache(t, y, "default")
    fp = FusedProblem(
        SearchProblem(
            t, y, dy, lc, ov, 1e-5,
            C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX, 0.01,
        )
    )
    fp.templates[4][:] *= -1
    fp.templates[5][:] *= -1
    fp.set_binning(0)
    periods = [1.2, 2.3, 4.2]
    expected = numpy.array([fp.search(p) for p in periods])
    assert numpy.any(expected[:, 3] > 1)
    fp.set_screen(min_length=48)
    assert fp.screen is not None
    actual = numpy.array([fp.search(p) for p in periods])
    numpy.testing.assert_array_equal(actual[:, 2:], expected[:, 2:])
    numpy.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-12)


def test_exact_screen_builds_weights_with_inactive_u2_flag():
    """U2 is inactive without PL targets; exact screening still needs dd_w."""
    from transitleastsquares.core import foldfast

    t, y, dy = make_case(8, n=3000, span=30, with_dy=True)
    ov, lc = cache(t, y, "default")
    fp = FusedProblem(
        SearchProblem(
            t, y, dy, lc, ov, 1e-5,
            C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX, 0.01,
        )
    )
    fp.set_pl(0, a2_approx=True)
    expected = fp.search(2.3)
    fp.set_screen(min_length=48)
    actual = fp.search(2.3)
    order = numpy.argsort(foldfast(t, 2.3), kind="mergesort")
    maxw = int(fp.templates[0][-1])
    maxw += maxw % 2
    w = fp.inv_dy2[numpy.concatenate((order, order[:maxw]))]
    first = numpy.concatenate(([0.0], numpy.cumsum(w - fp.input_means[1])))
    expected_dd = numpy.concatenate(([0.0], numpy.cumsum(first)))
    m = len(t) + maxw
    offset = 4 * len(t) + 4 * (m + 1) + m + 2
    actual_dd = fp.workspace()[0][offset : offset + m + 2].copy()
    # Reassociation in the two scans can differ from numpy near a zero crossing.
    # Check the certificate's global rounding allowance, then demand bitwise
    # equality to the same exact computation with the irrelevant flag cleared.
    budget = screen_round_budget(y, fp.inv_dy2, fp.input_means, maxw)
    assert numpy.max(numpy.abs(actual_dd - expected_dd)) <= budget[1]
    fp.set_pl(0, a2_approx=False)
    fp.set_screen(min_length=48)
    fp.search(2.3)
    numpy.testing.assert_array_equal(
        actual_dd, fp.workspace()[0][offset : offset + m + 2]
    )
    numpy.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_proxy_tolerance_rule():
    rule = ((0, 0.1), (512, 0.01))
    assert proxy_tolerance(0.06, 4096) == 0.06
    assert proxy_tolerance(rule, 64) == 0.1
    assert proxy_tolerance(rule, 511) == 0.1
    assert proxy_tolerance(rule, 512) == 0.01
    assert proxy_tolerance(rule, 10**6) == 0.01


@pytest.mark.parametrize("with_dy", [False, True])
def test_length_rule_matches_unscreened(with_dy):
    """Both proxy tolerances (forced onto short templates) keep every result."""
    t, y, dy = make_case(9, n=3000, span=30, with_dy=with_dy)
    ov, lc = cache(t, y, "ecc_sqrt")
    fp = FusedProblem(
        SearchProblem(
            t, y, dy, lc, ov, 1e-5,
            C.R_STAR_MIN, C.R_STAR_MAX, C.M_STAR_MIN, C.M_STAR_MAX, 0.01,
        )
    )
    periods = [0.7, 1.2, 2.3, 4.2, 10.0]
    expected = numpy.array([fp.search(p) for p in periods])
    lengths = fp.templates[3]
    split = int(numpy.median(lengths[lengths >= 48]))
    fp.set_screen(proxy_eps=((0, 0.1), (split, 0.01)), min_length=48)
    s = fp.screen
    assert s is not None and numpy.sum(s[0][lengths >= split] > 0) > 0
    assert numpy.sum(s[0][(lengths >= 48) & (lengths < split)] > 0) > 0
    actual = numpy.array([fp.search(p) for p in periods])
    numpy.testing.assert_array_equal(actual[:, 2:], expected[:, 2:])
    numpy.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-12)

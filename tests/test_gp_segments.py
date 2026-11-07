"""gp.search: light curve segments with their own filters, and the options of
the search grid (stellar limits, shortest duration, longest period)."""

import numpy as np
import pytest

from gp_cases import ROT_SYN, synthetic
from transitleastsquares import gp, gpu
from transitleastsquares.gp.api import _flatten_segments, _polynomial_flatten, _segments

GRID = dict(period_min=2.0, period_max=8.0, trend=False)


def _quarters(seed=3):
    """Two segments with drifts and an offset, white noise and a box transit (P = 7.3 d)."""
    rng = np.random.default_rng(seed)
    t = np.r_[np.arange(0.0, 40.0, 0.02), np.arange(42.0, 80.0, 0.02)]
    q2 = t > 41
    f = rng.normal(1.0, 3e-4, len(t)) + np.where(q2, 4e-3 - 6e-3 * (t - 42) / 38, 6e-3 * t / 40)
    f[np.abs((t - 1.5 + 3.65) % 7.3 - 3.65) < 0.08] -= 1.5e-3
    return t, f, np.full(len(t), 3e-4)


def test_one_segment_is_the_plain_search():
    t, f, e = synthetic()
    f = f / np.median(f)
    plain = gp.search(t, f, e, baseline=ROT_SYN, **GRID)
    seg = gp.search(t, f, e, baseline=ROT_SYN, segment_gap=1e9, segment_degree=None, **GRID)
    np.testing.assert_allclose(seg.chi2, plain.chi2, rtol=1e-10)
    assert (seg.period, seg.T0, seg.duration) == (plain.period, plain.T0, plain.duration)


def test_segments_have_their_own_filters():
    """chi2_0 is the sum over the segments searched alone."""
    t, f, e = _quarters()
    r = gp.search(t, f, e, baseline=ROT_SYN, segment_gap=0.5, segment_degree=None, **GRID)
    alone = 0.0
    for a, b in _segments(t, 0.5):
        m = np.median(f[a:b])
        alone += gp.search(t[a:b], f[a:b] / m, e[a:b] / m, baseline=ROT_SYN, **GRID).chi2_0
    assert r.chi2_0 == pytest.approx(alone, rel=1e-10)


def test_segments_recover_the_signal_and_return_the_flat_flux():
    t, f, e = _quarters()
    r = gp.search(t, f, e, baseline=ROT_SYN, segment_gap=0.5, period_min=2.0, period_max=20.0)
    assert r.period == pytest.approx(7.3, rel=0.01)
    assert abs((r.T0 - 1.5 + 3.65) % 7.3 - 3.65) < 0.08
    assert r.flux.std() < 1.5 * 3e-4 + 1.5e-3 * np.sqrt(0.16 / 7.3)
    assert len(r.trend) == len(t) and np.max(np.abs(r.trend - 1)) < 1e-3


def test_flatten_with_outliers():
    t = np.linspace(10.0, 100.0, 2000)
    x = (t - 55.0) / 45.0
    p = 1 + 2e-3 * x - 3e-3 * x**2 + 1e-3 * x**4
    y = p.copy()
    y[100:110] -= 0.05
    np.testing.assert_allclose(np.delete(_polynomial_flatten(t, y, 4), np.s_[100:110]), 1.0, atol=1e-6)
    assert np.array_equal(_polynomial_flatten(t[:4], y[:4], 4), y[:4])  # too few points: unchanged


def test_flatten_errors_follow_the_median():
    t = np.r_[np.arange(0.0, 10.0, 0.1), np.arange(20.0, 30.0, 0.1)]
    y = np.where(t < 15, 2.0, 4.0)
    y2, e2 = _flatten_segments(t, y, np.full(len(t), 0.02), _segments(t, 0.5), None)
    np.testing.assert_allclose(y2, 1.0)
    np.testing.assert_allclose(e2, np.where(t < 15, 0.01, 0.005))


def test_short_segments_are_not_searched():
    t, f, e = _quarters()
    t, f, e = np.r_[t, 90.0, 90.1, 90.2], np.r_[f, 1.0, 1.0, 1.0], np.r_[e, e[:3]]
    r = gp.search(t, f, e, baseline=ROT_SYN, segment_gap=0.5, **GRID)
    assert not r.use[-3:].any() and r.use[:-3].all() and r.n_points == len(t) - 3


@pytest.mark.parametrize("kw", [dict(bayes_factor=True), dict(bin_fraction=1 / 30), dict(t0_step_duration=None)])
def test_segments_refuse(kw):
    t, f, e = _quarters()
    with pytest.raises(ValueError, match="segments"):
        gp.search(t, f, e, baseline=ROT_SYN, segment_gap=0.5, **GRID, **kw)


def test_longest_period():
    t, f, e = _quarters()
    span = t[-1] - t[0]
    kw = dict(baseline=ROT_SYN, period_min=20.0, trend=False)
    assert gp.search(t, f, e, **kw).periods.max() == pytest.approx(span / 2, rel=0.02)
    assert gp.search(t, f, e, n_transits_min=1, **kw).periods.max() == pytest.approx(span, rel=0.02)
    assert gp.search(t, f, e, grid_transits_min=1, **kw).periods.max() == pytest.approx(span, rel=0.02)


def test_duration_limits():
    t, f, e = _quarters()
    kw = dict(baseline=ROT_SYN, **GRID)
    wide = gp.search(t, f, e, **kw).durations
    assert gp.search(t, f, e, duration_min_scale=0.44, **kw).durations[0] == pytest.approx(0.44 * wide[0])
    near = gp.search(t, f, e, R_star_min=0.8, R_star_max=1.2, M_star_min=0.8, M_star_max=1.2, **kw).durations
    assert near[0] > wide[0] and near[-1] < wide[-1]


@pytest.mark.skipif(not gpu.available(), reason="no CUDA device")
def test_segments_gpu():
    t, f, e = _quarters()
    kw = dict(baseline=ROT_SYN, segment_gap=0.5, **GRID)
    cpu, dev = gp.search(t, f, e, **kw), gp.search(t, f, e, device="gpu", **kw)
    assert dev.chi2_0 == pytest.approx(cpu.chi2_0, rel=1e-6)
    np.testing.assert_allclose(dev.delta_chi2, cpu.delta_chi2, atol=1e-2 * max(1.0, cpu.delta_chi2.max() * 1e-2))
    assert dev.period == pytest.approx(cpu.period, rel=1e-3)

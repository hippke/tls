"""transitleastsquares.gp: the public interface (gp.search, baselines,
results). Experiments use this interface, so its contract is tested here."""

import os
import subprocess
import sys

import numpy as np
import pytest

from gp_cases import ROT_SYN, k2_25, synthetic
from transitleastsquares import gp
from transitleastsquares.gp.baselines import rotation_period_guess

KEYS = ["period", "T0", "duration", "depth", "depth_err", "snr", "transit_times", "periods", "power", "SDE",
        "SDE_raw", "delta_chi2", "chi2", "chi2_0", "durations", "log_bf", "log_bf_global", "baseline",
        "baseline_log_likelihood", "trend", "time", "flux", "flux_err", "use", "n_points"]


def test_tls_does_not_import_gp():
    """`import transitleastsquares` (the TLS search) does not load gp."""
    code = "import sys, transitleastsquares; print(any(m.startswith('transitleastsquares.gp') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ))
    assert out.stdout.strip() == "False", out.stderr


@pytest.fixture(scope="module")
def fitted():
    """End-to-end: rotation GP fitted with flare clipping, then the search."""
    t, f, e = synthetic()
    return (t, f, e), gp.search(t, f, e, period_min=1.0, period_max=8.0, rotation_period=2.3)


def test_results_contract(fitted):
    (t, f, e), r = fitted
    assert isinstance(r, gp.GPSearchResults) and set(KEYS) <= set(r)
    for k in KEYS:
        assert getattr(r, k) is r[k]
    with pytest.raises(AttributeError):
        r.no_such_key
    n = len(t)
    for k in ("time", "flux", "flux_err", "use", "trend"):
        assert len(r[k]) == n
    assert len(r.periods) == len(r.power) == len(r.delta_chi2) == len(r.chi2)
    np.testing.assert_allclose(r.chi2, r.chi2_0 - r.delta_chi2)
    assert r.n_points == r.use.sum()
    assert isinstance(r.baseline, gp.RotationBaseline) and np.isfinite(r.baseline_log_likelihood)
    assert r.log_bf is None and r.log_bf_global is None  # bayes_factor=False
    assert np.all(np.diff(r.transit_times) > 0) and r.transit_times[0] >= t.min() and r.transit_times[-1] <= t.max()


def test_finds_injected_transit(fitted):
    """synthetic(): P = 3.1 d, duration 0.11 d, template depth 2 ppt."""
    _, r = fitted
    assert abs(r.period - 3.1) / 3.1 < 0.01
    assert abs(r.depth - 0.002) < 4 * r.depth_err
    assert abs(r.baseline.period - 2.3) / 2.3 < 0.05
    assert np.std(r.flux / r.trend) < 0.5 * np.std(r.flux)  # the trend follows the variability


def test_spline_baseline():
    t, f, e = synthetic()
    r = gp.search(t, f, e, baseline="spline", period_min=1.0, period_max=8.0)
    assert isinstance(r.baseline, gp.SplineBaseline) and 0.05 <= r.baseline.h <= 5.0


def test_fixed_baseline_is_used_as_given():
    t, f, e = synthetic()
    r = gp.search(t, f, e, baseline=ROT_SYN, period_min=1.0, period_max=8.0)
    assert r.baseline is ROT_SYN and r.baseline_log_likelihood is None and r.use.all()


def test_input_cleaning():
    """Unsorted input with NaNs gives the same result as clean sorted input;
    dy=None estimates the errors from the point-to-point scatter."""
    t, f, e = synthetic(days=12.0)
    rng = np.random.default_rng(0)
    p = rng.permutation(len(t))
    tb, fb, eb = t[p].copy(), f[p].copy(), e[p].copy()
    fb[3] = np.nan
    keep = np.isfinite(f[p])
    keep[3] = False
    ref = gp.search(np.sort(t[p][keep]), f[p][keep][np.argsort(t[p][keep])], e[p][keep][np.argsort(t[p][keep])],
                    baseline=ROT_SYN, period_min=1.0, period_max=4.0)
    r = gp.search(tb, fb, eb, baseline=ROT_SYN, period_min=1.0, period_max=4.0)
    np.testing.assert_array_equal(r.delta_chi2, ref.delta_chi2)
    np.testing.assert_array_equal(r.time, ref.time)
    r0 = gp.search(t, f, None, baseline=ROT_SYN, period_min=1.0, period_max=4.0)
    assert np.all(r0.flux_err == r0.flux_err[0]) and 0.5 * np.median(e) < r0.flux_err[0] < 2 * np.median(e)


def test_reproducible():
    t, f, e = synthetic(days=12.0)
    a = gp.search(t, f, e, baseline=ROT_SYN, period_min=1.0, period_max=4.0)
    b = gp.search(t, f, e, baseline=ROT_SYN, period_min=1.0, period_max=4.0)
    np.testing.assert_array_equal(a.delta_chi2, b.delta_chi2)
    assert a.period == b.period and a.depth == b.depth


@pytest.mark.parametrize("kwargs, error", [
    (dict(baseline="no_such_model"), ValueError),
    (dict(baseline=3), TypeError),
    (dict(n_starts=5), ValueError),
])
def test_invalid_arguments(kwargs, error):
    t, f, e = synthetic(days=12.0)
    with pytest.raises(error):
        gp.search(t, f, e, period_min=1.0, period_max=4.0, **kwargs)


def test_invalid_errors():
    t, f, e = synthetic(days=12.0)
    with pytest.raises(ValueError):
        gp.search(t, f, -e, baseline=ROT_SYN)


def test_baseline_parameters_round_trip():
    d = ROT_SYN.as_dict()
    assert set(d) == set(gp.RotationBaseline.KEYS)
    assert d["sigma"] == pytest.approx(5.0e-3) and d["period"] == 2.3
    again = gp.RotationBaseline(**d)
    for k in d:
        assert getattr(again, k) == pytest.approx(d[k], rel=1e-15)
    s = gp.SplineBaseline(h=0.2, jitter=1e-4)
    assert s.as_dict() == dict(h=0.2, jitter=pytest.approx(1e-4), sigma_ref=None, cadence=None)
    assert "RotationBaseline(" in repr(ROT_SYN)


def test_rotation_period_guess():
    t, f, e = k2_25()
    assert abs(rotation_period_guess(t, f) - 1.8764) < 0.01  # astropy LombScargle: 1.8764 d


# --- degenerate hyperparameters (found in the validation runs) ------------
def test_sho_critical_damping():
    """Q = 1/2 (critically damped) is continuous with its neighbours (the
    eigendecomposition used before was singular there)."""
    from transitleastsquares.gp.statespace import SHO

    dt = np.array([0.0, 0.0204, 0.5, 3.0])
    A0, Q0, _ = SHO(2.0, 3.0, 0.5).discretize(dt)
    for q in (0.5 - 1e-6, 0.5 + 1e-6):
        A1, Q1, _ = SHO(2.0, 3.0, q).discretize(dt)
        np.testing.assert_allclose(A1, A0, atol=1e-5)
        np.testing.assert_allclose(Q1, Q0, atol=1e-4)
    assert np.all(np.isfinite(A0)) and np.all(np.isfinite(Q0))


def test_zero_variance_component_trend():
    """A component with zero amplitude (f = 0: no P/2 term) makes the
    predicted covariance singular; the smoother must still work."""
    t, f, e = synthetic(days=12.0)
    d = ROT_SYN.as_dict()
    d["f"] = 0.0
    trend = gp.RotationBaseline(**d).trend(t, f, e)
    assert np.all(np.isfinite(trend)) and np.std(f / trend) < np.std(f)


def test_flare_clipping_guards(monkeypatch):
    """A degenerate trend (NaN) or a round that would clip > 20 % of the
    points ends the clipping instead of removing everything."""
    from transitleastsquares.gp import baselines

    t, f, e = synthetic(days=12.0)
    monkeypatch.setattr(baselines.RotationBaseline, "trend", lambda self, t, y, dy, use=None, device="cpu": np.full(len(t), np.nan))
    b, ll, use = baselines.fit_with_flare_clipping("rotation", t, f, e, period_guess=2.3, n_starts=1)
    assert use.all()
    monkeypatch.setattr(baselines.RotationBaseline, "trend", lambda self, t, y, dy, use=None, device="cpu": y - np.arange(len(y)))
    b, ll, use = baselines.fit_with_flare_clipping("rotation", t, f, e, period_guess=2.3, n_starts=1)
    assert use.sum() >= 0.8 * len(t)


def test_search_without_trend():
    from gp_cases import ROT_SYN, synthetic

    t, f, e = synthetic()
    r = gp.search(t, f, e, baseline=ROT_SYN, period_min=1.0, period_max=8.0, trend=False)
    ref = gp.search(t, f, e, baseline=ROT_SYN, period_min=1.0, period_max=8.0)
    assert r.trend is None and ref.trend is not None
    assert r.period == ref.period and r.SDE == ref.SDE

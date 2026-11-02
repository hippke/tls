"""gp.search results must match the golden files (fixed baselines).

Exact refactorings keep dchi2 to ~1e-9. Approximations (if any) must stay
within DCHI2_RTOL of the largest dchi2 and must not change the best signal.
"""

import os

import numpy as np
import pytest

from gp_cases import CASES, GOLDEN, loglik_ppt, run

DCHI2_RTOL = 1e-6  # relative to max(dchi2)
SCALAR_RTOL = 1e-5


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    gold = np.load(os.path.join(GOLDEN, f"{request.param}.npz"))
    return request.param, run(request.param), gold


def test_periods_and_dchi2(case):
    name, r, gold = case
    np.testing.assert_array_equal(r["periods"], gold["periods"])
    scale = gold["dchi2"].max()
    err = np.max(np.abs(r["delta_chi2"] - gold["dchi2"])) / scale
    assert err < DCHI2_RTOL, f"{name}: max dchi2 error {err:.2e} of max"


def test_best_signal(case):
    name, r, gold = case
    for k in ("period", "T0", "duration"):
        assert r[k] == gold[k], f"{name}: {k} {r[k]} != {gold[k]}"
    for k in ("depth", "depth_err", "snr", "SDE", "chi2_0"):
        np.testing.assert_allclose(r[k], gold[k], rtol=SCALAR_RTOL, err_msg=f"{name}: {k}")


def test_log_bayes_factor(case):
    name, r, gold = case
    np.testing.assert_allclose(r["log_bf_global"], gold["log_bf_global"], rtol=SCALAR_RTOL, atol=1e-6)
    ok = np.isfinite(gold["log_bf_per_period"])
    np.testing.assert_array_equal(np.isfinite(r["log_bf"]), ok)
    np.testing.assert_allclose(r["log_bf"][ok], gold["log_bf_per_period"][ok], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", list(CASES))
def test_loglik(name):
    gold = np.load(os.path.join(GOLDEN, f"{name}.npz"))
    np.testing.assert_allclose(loglik_ppt(name), gold["loglik"], rtol=1e-9)  # diffuse prior: rounding


def test_bayes_factor_off_same_detection():
    """bayes_factor=False (the default) must not change the search itself."""
    gold = np.load(os.path.join(GOLDEN, "syn_rot.npz"))
    r = run("syn_rot", bayes_factor=False)
    assert r["log_bf_global"] is None and r["log_bf"] is None
    np.testing.assert_allclose(r["delta_chi2"], gold["dchi2"], rtol=0, atol=DCHI2_RTOL * gold["dchi2"].max())
    assert r["period"] == gold["period"] and r["duration"] == gold["duration"]


@pytest.mark.parametrize("name", list(CASES))
def test_uniform_grid_unchanged(name):
    """t0_step_duration=None (uniform cadence/4 grid) must reproduce the
    golden files frozen before the duration-dependent default (*_cad4)."""
    gold = np.load(os.path.join(GOLDEN, f"{name}_cad4.npz"))
    r = run(name, t0_step_duration=None)
    np.testing.assert_array_equal(r["periods"], gold["periods"])
    assert np.max(np.abs(r["delta_chi2"] - gold["dchi2"])) / gold["dchi2"].max() < DCHI2_RTOL
    for k in ("period", "T0", "duration"):
        assert r[k] == gold[k]
    np.testing.assert_allclose(r["log_bf_global"], gold["log_bf_global"], rtol=SCALAR_RTOL)


def test_strided_fold_matches_standard():
    """t0_step_duration so small that every step multiple is 1: the strided
    fold (durations outside) must give the uniform-grid result, Bayes factor
    included."""
    gold = np.load(os.path.join(GOLDEN, "syn_rot_cad4.npz"))
    r = run("syn_rot", t0_step_duration=1e-9)
    np.testing.assert_allclose(r["delta_chi2"], gold["dchi2"], rtol=0, atol=DCHI2_RTOL * gold["dchi2"].max())
    assert r["period"] == gold["period"] and r["duration"] == gold["duration"]
    ok = np.isfinite(gold["log_bf_per_period"])
    np.testing.assert_allclose(r["log_bf"][ok], gold["log_bf_per_period"][ok], rtol=1e-6, atol=1e-6)

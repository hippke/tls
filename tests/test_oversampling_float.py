"""float / fractional oversampling_factor (issue #105).

The SDE running-median kernel is oversampling_factor * 30 samples. It used to
stay a float, which crashed the median filter for fractional factors (TLS
1.33: IndexError for 3.14) and, with the faster median of TLS 2.0, even for
whole-number floats such as 3.0 (TypeError)."""

import warnings

import numpy
import pytest

from transitleastsquares import tls_constants, transitleastsquares
from transitleastsquares.stats import _sde_kernel, sde_trend, spectra

KERNEL_SIZE = tls_constants.SDE_MEDIAN_KERNEL_SIZE


def old_kernel(oversampling_factor, factor=1):
    """The kernel of TLS <= 1.33 / 2.0 for the inputs that worked."""
    kernel = int(oversampling_factor * KERNEL_SIZE * factor)
    if kernel % 2 == 0:
        kernel += 1
    return kernel


@pytest.mark.parametrize("oversampling_factor", range(1, 11))
@pytest.mark.parametrize("factor", [1, tls_constants.SDE_HYBRID_KERNEL_FACTOR])
def test_kernel_unchanged_for_integer_factors(oversampling_factor, factor):
    assert _sde_kernel(oversampling_factor, factor) == old_kernel(
        oversampling_factor, factor
    )


@pytest.mark.parametrize("oversampling_factor", [0.01, 0.5, 1.5, 2.0, 2.5, 3.0, 3.14, 4.7])
@pytest.mark.parametrize("factor", [1, tls_constants.SDE_HYBRID_KERNEL_FACTOR])
def test_kernel_is_odd_int(oversampling_factor, factor):
    kernel = _sde_kernel(oversampling_factor, factor)
    assert isinstance(kernel, int)
    assert kernel >= 1 and kernel % 2 == 1


def chi2_curve(n=500):
    return numpy.linspace(2, 1, n) + numpy.random.default_rng(3).normal(0, 1e-3, n)


@pytest.mark.parametrize("oversampling_factor", [1, 2, 3, 5])
def test_whole_number_float_equals_int(oversampling_factor):
    chi2 = chi2_curve()
    for a, b in zip(
        spectra(chi2, oversampling_factor), spectra(chi2, float(oversampling_factor))
    ):
        numpy.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("oversampling_factor", [0.5, 2.5, 3.14])
def test_spectra_fractional(oversampling_factor):
    chi2 = chi2_curve()
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    assert len(power) == len(chi2)
    assert numpy.all(numpy.isfinite(power)) and numpy.isfinite(SDE)


@pytest.mark.parametrize("oversampling_factor", [3, 3.0, 3.14])
def test_hybrid_detrend_fractional(oversampling_factor):
    chi2 = chi2_curve(2000)
    periods = numpy.geomspace(0.5, 100, len(chi2))
    power_raw = spectra(chi2, 3)[1]
    trend = sde_trend(power_raw, periods, oversampling_factor, detrend="hybrid")
    assert len(trend) == len(power_raw) and numpy.all(numpy.isfinite(trend))


def noise_curve():
    # the reproducer of issue #105
    t = numpy.linspace(0, 30, num=300)
    y = numpy.random.default_rng(1).normal(size=t.shape) * 0.1 + 1
    return t, y


def search(oversampling_factor):
    t, y = noise_curve()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "no transit were fit"
        return transitleastsquares(t, y, verbose=False).power(
            oversampling_factor=oversampling_factor,
            use_threads=1,
            show_progress_bar=False,
            verbose=False,
        )


def test_issue_105_reproducer_fractional():
    results = search(3.14)
    assert len(results.power) == len(results.periods)
    assert numpy.all(numpy.isfinite(results.power))


def test_search_whole_number_float_equals_int():
    a, b = search(3), search(3.0)
    numpy.testing.assert_array_equal(a.periods, b.periods)
    numpy.testing.assert_array_equal(a.power, b.power)
    assert a.SDE == b.SDE

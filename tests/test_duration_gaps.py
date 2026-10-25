"""Regression test for #83: the transit duration must not shrink when the
light curve has long gaps (e.g. TESS sectors observed years apart)."""

import numpy
import pytest

from transitleastsquares import transitleastsquares

PERIOD, T0, T14_DAYS, DEPTH = 5.66, 1.3, 1.4 / 24, 3000e-6
CADENCE = 10 / 1440


def _box_lightcurve(blocks):
    t = numpy.concatenate([numpy.arange(a, b, CADENCE) for a, b in blocks])
    rng = numpy.random.default_rng(42)
    y = 1 + rng.normal(0, 300e-6, t.size)
    phase = ((t - T0 + 0.5 * PERIOD) % PERIOD) - 0.5 * PERIOD
    y[numpy.abs(phase) < T14_DAYS / 2] -= DEPTH
    return t, y


@pytest.mark.parametrize(
    "blocks",
    [
        [(0, 54)],  # contiguous
        [(0, 27), (1100, 1127)],  # two sectors ~3 years apart (fill factor ~0.05)
    ],
)
def test_duration_independent_of_gaps(blocks):
    t, y = _box_lightcurve(blocks)
    results = transitleastsquares(t, y).power(
        period_min=5,
        period_max=6.5,
        oversampling_factor=5,  # fine grid: long baselines need it
        duration_grid_step=1.05,
        use_threads=1,
        show_progress_bar=False,
    )
    assert results.period == pytest.approx(PERIOD, abs=0.01)
    assert results.duration == pytest.approx(T14_DAYS, rel=0.15)

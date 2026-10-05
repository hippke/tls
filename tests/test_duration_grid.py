import os

import numpy

from transitleastsquares import duration_grid, period_grid

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_duration_grid():
    print("Starting test: duration_grid...", end="")
    periods = period_grid(
        R_star=1,  # R_sun
        M_star=1,  # M_sun
        time_span=20,  # days
        period_min=0,
        period_max=999,
        oversampling_factor=3,
    )
    durations = duration_grid(periods, log_step=1.05, shortest=2)
    numpy.testing.assert_almost_equal(max(durations), 0.12)
    numpy.testing.assert_almost_equal(min(durations), 0.004562690993268325)
    numpy.testing.assert_equal(len(durations), 69)
    print("passed")


if __name__ == "__main__":
    test_duration_grid()

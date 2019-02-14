from __future__ import division, print_function
import numpy
import scipy
import scipy.signal
from transitleastsquares import (
    transitleastsquares,
    transit_mask,
    cleaned_array
    )


def loadfile(filename):
    data = numpy.genfromtxt(
            filename,
            delimiter=",",
            dtype="f8, f8",
            names=["t", "y"]
        )
    return data["t"], data["y"]


if __name__ == "__main__":
    print("Starting test: Multi-planet...", end='')
    t, y = loadfile("EPIC201367065.csv")
    trend = scipy.signal.medfilt(y, 25)
    y_filt = y / trend

    model = transitleastsquares(t, y_filt)
    results = model.power()
    
    numpy.testing.assert_almost_equal(max(results.power), 45.934646920004326, decimal=5)
    numpy.testing.assert_almost_equal(max(results.power_raw), 44.00867236551441, decimal=5)
    numpy.testing.assert_almost_equal(min(results.power), -0.620153987656165, decimal=5)
    numpy.testing.assert_almost_equal(min(results.power_raw), -0.29015390864908414, decimal=5)
    print('Detrending of power spectrum from power_raw passed')

    # Mask of the first planet
    intransit = transit_mask(
        t,
        results.period,
        2 * results.duration,
        results.T0
        )
    y_second_run = y_filt[~intransit]
    t_second_run = t[~intransit]
    t_second_run, y_second_run = cleaned_array(t_second_run, y_second_run)

    # Search for second planet
    model_second_run = transitleastsquares(t_second_run, y_second_run)
    results_second_run = model_second_run.power()
    numpy.testing.assert_almost_equal(
        results_second_run.duration, 0.1478628403227008, decimal=5
    )
    numpy.testing.assert_almost_equal(
        results_second_run.SDE, 34.98291056410117, decimal=5
    )
    numpy.testing.assert_almost_equal(
        results_second_run.rp_rs, 0.025852178872027086, decimal=5
    )

    print('Passed')

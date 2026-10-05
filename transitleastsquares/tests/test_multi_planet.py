from __future__ import division, print_function
import numpy
import scipy
import scipy.signal
from transitleastsquares import transitleastsquares, transit_mask, cleaned_array


def loadfile(filename):
    data = numpy.genfromtxt(filename, delimiter=",", dtype="f8, f8", names=["t", "y"])
    return data["t"], data["y"]


if __name__ == "__main__":
    # Reference values updated for TLS 1.33: PR #112 changed the chi2 of
    # "no fit" periods from len(y) to sum((y-1)^2/dy^2), which shifts power/SDE
    # slightly. The test was not updated upstream and failed on 1.33.
    print("Starting test: Multi-planet...", end="")
    t, y = loadfile("EPIC201367065.csv")
    trend = scipy.signal.medfilt(y, 25)
    y_filt = y / trend

    model = transitleastsquares(t, y_filt)
    results = model.power()

    numpy.testing.assert_almost_equal(max(results.power), 45.185902962942706, decimal=3)
    numpy.testing.assert_almost_equal(
        max(results.power_raw), 42.10612241872027, decimal=3
    )
    numpy.testing.assert_almost_equal(min(results.power), -0.6113463157599826, decimal=3)
    numpy.testing.assert_almost_equal(
        min(results.power_raw), -0.4961035632387514, decimal=3
    )
    print("Detrending of power spectrum from power_raw passed")

    # Mask of the first planet
    intransit = transit_mask(t, results.period, 2 * results.duration, results.T0)
    y_second_run = y_filt[~intransit]
    t_second_run = t[~intransit]
    t_second_run, y_second_run = cleaned_array(t_second_run, y_second_run)

    # Search for second planet
    model_second_run = transitleastsquares(t_second_run, y_second_run)
    results_second_run = model_second_run.power()
    numpy.testing.assert_almost_equal(
        results_second_run.duration, 0.15061016994013998, decimal=3
    )
    numpy.testing.assert_almost_equal(
        results_second_run.SDE, 34.987371359756565, decimal=3
    )
    numpy.testing.assert_almost_equal(
        results_second_run.rp_rs, 0.025480893577558485, decimal=3
    )

    print("Passed")

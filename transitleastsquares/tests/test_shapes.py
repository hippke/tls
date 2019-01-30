import numpy
import batman
import scipy
import scipy.signal
from transitleastsquares import transitleastsquares


def loadfile(filename):
    data = numpy.genfromtxt(
            filename,
            delimiter=",",
            dtype="f8, f8",
            names=["t", "y"]
        )
    return data["t"], data["y"]


if __name__ == "__main__":
    print("Starting test: transit shapes...", end='')
    # Testing transit shapes
    t, y = loadfile("EPIC206154641.csv")
    trend = scipy.signal.medfilt(y, 25)
    y_filt = y / trend

    # grazing

    model_grazing = transitleastsquares(t, y_filt)
    results_grazing = model_grazing.power(transit_template="grazing")

    numpy.testing.assert_almost_equal(
        results_grazing.duration, 0.08785037229975422, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results_grazing.chi2red), 0.06683059525866272, decimal=5
    )
    numpy.testing.assert_almost_equal(results_grazing.SDE, 64.59390167350149, decimal=5)
    numpy.testing.assert_almost_equal(
        results_grazing.rp_rs, 0.0848188816853949, decimal=5
    )
    print("Test passed: Grazing-shaped")

    # box
    model_box = transitleastsquares(t, y_filt)
    results_box = model_box.power(transit_template="box")

    numpy.testing.assert_almost_equal(
        results_box.duration, 0.0660032849735193, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results_box.chi2red), 0.12358085916803863, decimal=5
    )
    numpy.testing.assert_almost_equal(results_box.SDE, 56.748626429853424, decimal=5)
    numpy.testing.assert_almost_equal(results_box.rp_rs, 0.0861904513547099, decimal=5)

    print("Test passed: Box-shaped")
    print('All tests passed')
    
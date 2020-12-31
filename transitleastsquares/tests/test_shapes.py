from __future__ import division, print_function
import numpy
import scipy
import scipy.signal
from transitleastsquares import transitleastsquares
from transitleastsquares.template_generator.default_transit_template_generator import DefaultTransitTemplateGenerator


def loadfile(filename):
    data = numpy.genfromtxt(filename, delimiter=",", dtype="f8, f8", names=["t", "y"])
    return data["t"], data["y"]

class TestTransitTemplateGenerator(DefaultTransitTemplateGenerator):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    print("Starting test: transit shapes...", end="")

    # Testing transit shapes
    t, y = loadfile("EPIC206154641.csv")
    trend = scipy.signal.medfilt(y, 25)
    y_filt = y / trend

    # box
    model_box = transitleastsquares(t, y_filt)
    results_box = model_box.power(transit_template="box")
    numpy.testing.assert_almost_equal(
        results_box.duration, 0.06111785726416931, decimal=5)
    numpy.testing.assert_almost_equal(results_box.rp_rs, 0.08836981203437415, decimal=5)
    print("Test passed: Box-shaped")

    # grazing
    model_grazing = transitleastsquares(t, y_filt)
    results_grazing = model_grazing.power(transit_template="grazing")

    numpy.testing.assert_almost_equal(
        results_grazing.duration, 0.08948265482047034, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results_grazing.chi2red), 0.06759475703796078, decimal=5)
    print("Test passed: Grazing-shaped")

    # comet
    model_comet = transitleastsquares(t, y_filt)
    results_comet = model_comet.power(transit_template="comet")

    numpy.testing.assert_almost_equal(
        results_comet.duration, 0.23209496125032572, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results_comet.chi2red), 0.0980794344892094, decimal=5)
    print("Test passed: Comet-shaped")

    model_custom = transitleastsquares(t, y_filt)
    try:
        results_custom = model_custom.power(transit_template="custom",
                                            transit_template_generator="wrongTransitTemplateGenerator")
        assert False
    except ValueError as e:
        if e.args[0] == "The custom transit_template_generator does not implement TransitTemplateGenerator.":
            print("Test passed: Wrong custom transit template generator.")
        else:
            assert False

    # custom
    model_custom = transitleastsquares(t, y_filt)
    results_custom = model_custom.power(transit_template="custom",
                                        transit_template_generator=TestTransitTemplateGenerator())

    numpy.testing.assert_almost_equal(
        results_custom.duration, 0.06722964299058617, decimal=5
    )
    numpy.testing.assert_almost_equal(
        min(results_custom.chi2red), 0.09977336183179186, decimal=5)
    print("Test passed: Custom-shaped")

    print("All tests passed")

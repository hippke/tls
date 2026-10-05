import os

import numpy

from transitleastsquares import FAP

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_FAP():
    print("Starting test FAP...", end="")
    numpy.testing.assert_equal(FAP(SDE=2), numpy.nan)
    numpy.testing.assert_equal(FAP(SDE=7), 0.009443778)
    numpy.testing.assert_equal(FAP(SDE=99), 8.0032e-05)
    print("passed")


if __name__ == "__main__":
    test_FAP()

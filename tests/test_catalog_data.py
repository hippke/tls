import os

import numpy
import pytest

from transitleastsquares import catalog_info

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.network
def test_catalog_data():
    if True:
        print("Starting test: catalog_data...")
        (a, b), mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(
            EPIC_ID=204341806
        )
        # EPIC 204341806 has no Teff/logg in the EPIC catalog (masked values).
        # TLS <= 1.33 did not detect masked values and returned the LD of the
        # first table row (Teff=1500 K, logg=2.5): (1.1605, -0.1704). Now the
        # documented fallback Teff=6000 K, logg=4 is used.
        numpy.testing.assert_almost_equal((a, b), (0.4804, 0.1867))
        numpy.testing.assert_almost_equal(mass, numpy.nan)
        numpy.testing.assert_almost_equal(mass_min, numpy.nan)
        numpy.testing.assert_almost_equal(mass_max, numpy.nan)
        numpy.testing.assert_almost_equal(radius, numpy.nan)
        numpy.testing.assert_almost_equal(radius_min, numpy.nan)
        numpy.testing.assert_almost_equal(radius_max, numpy.nan)

        (a, b), mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(
            EPIC_ID=204099713
        )
        numpy.testing.assert_almost_equal((a, b), (0.4804, 0.1867))
        numpy.testing.assert_almost_equal(mass, 1.046)
        numpy.testing.assert_almost_equal(mass_min, 0.898)
        numpy.testing.assert_almost_equal(mass_max, 0.642)
        numpy.testing.assert_almost_equal(radius, 1.261)
        numpy.testing.assert_almost_equal(radius_min, 1.044)
        numpy.testing.assert_almost_equal(radius_max, 0.925)
        print("Test passed: EPIC catalog pull from Vizier using astroquery")

        (a, b), mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(
            KIC_ID=757076
        )
        numpy.testing.assert_almost_equal((a, b), (0.5819, 0.137))
        numpy.testing.assert_almost_equal(mass, 1.357)
        numpy.testing.assert_almost_equal(mass_max, 0.204)
        numpy.testing.assert_almost_equal(mass_min, 0.475)
        numpy.testing.assert_almost_equal(radius, 3.128)
        numpy.testing.assert_almost_equal(radius_max, 0.987)
        numpy.testing.assert_almost_equal(radius_min, 2.304)

        (a, b), mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(
            KIC_ID=12557548
        )
        numpy.testing.assert_almost_equal((a, b), (0.701, 0.0462))
        numpy.testing.assert_almost_equal(mass, 0.6666, decimal=3)
        numpy.testing.assert_almost_equal(mass_max, 0.067, decimal=3)
        numpy.testing.assert_almost_equal(mass_min, 0.055, decimal=3)
        numpy.testing.assert_almost_equal(radius, 0.660, decimal=3)
        numpy.testing.assert_almost_equal(radius_max, 0.054, decimal=3)
        numpy.testing.assert_almost_equal(radius_min, 0.054, decimal=3)
        print("Test passed: KIC catalog pull from Vizier using astroquery")

        (a, b), mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(
            TIC_ID=279741377
        )
        numpy.testing.assert_equal((a, b), (0.354, 0.2321))
        numpy.testing.assert_equal(mass, 1.12)
        numpy.testing.assert_equal(mass_min, 0.148162)
        numpy.testing.assert_equal(mass_max, 0.148162)
        numpy.testing.assert_equal(radius, 3.08104)
        numpy.testing.assert_equal(radius_min, 0.167789)
        numpy.testing.assert_equal(radius_max, 0.167789)
        print("Test passed: TESS Input Catalog (TIC) pull from Vizier using astroquery")

        print("All tests passed")


if __name__ == "__main__":
    test_catalog_data()

"""Stellar parameters and limb darkening from online catalogs (needs astroquery)."""

import warnings
from os import path

import numpy

from transitleastsquares import tls_constants


def _vizier():
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        raise ImportError("Package astroquery required but failed to import") from None
    return Vizier


def catalog_info_KIC(KIC_ID):
    """Takes KIC_ID, returns stellar information from online catalog using Vizier"""
    if type(KIC_ID) is not int:
        raise TypeError('KIC_ID ID must be of type "int"')
    Vizier = _vizier()
    columns = ["Teff", "log(g)", "Rad", "E_Rad", "e_Rad", "Mass", "E_Mass", "e_Mass"]
    catalog = "J/ApJS/229/30/catalog"
    result = (
        Vizier(columns=columns)
        .query_constraints(KIC=KIC_ID, catalog=catalog)[0]
        .as_array()
    )
    # Columns in the requested order: (E_ = upper, e_ = lower error)
    Teff, logg, radius, radius_max, radius_min, mass, mass_max, mass_min = result[0]
    return Teff, logg, radius, radius_min, radius_max, mass, mass_min, mass_max


def catalog_info_EPIC(EPIC_ID):
    """Takes EPIC_ID, returns stellar information from online catalog using Vizier"""
    if type(EPIC_ID) is not int:
        raise TypeError('EPIC_ID ID must be of type "int"')
    if (EPIC_ID < 201000001) or (EPIC_ID > 251813738):
        raise TypeError("EPIC_ID ID must be in range 201000001 to 251813738")
    Vizier = _vizier()
    columns = ["Teff", "logg", "Rad", "E_Rad", "e_Rad", "Mass", "E_Mass", "e_Mass"]
    catalog = "IV/34/epic"
    result = (
        Vizier(columns=columns)
        .query_constraints(ID=EPIC_ID, catalog=catalog)[0]
        .as_array()
    )
    Teff, logg, radius, radius_max, radius_min, mass, mass_max, mass_min = result[0]
    return Teff, logg, radius, radius_min, radius_max, mass, mass_min, mass_max


def catalog_info_TIC(TIC_ID):
    """Takes TIC_ID, returns stellar information from the TESS Input Catalog (MAST).

    Columns are selected by name (not by position, which silently breaks when
    MAST changes the column order). The TIC gives symmetric errors (e_rad,
    e_mass); they are returned as both lower and upper error.
    """
    if type(TIC_ID) is not int:
        raise TypeError('TIC_ID ID must be of type "int"')
    try:
        from astroquery.mast import Catalogs
    except ImportError:
        raise ImportError("Package astroquery required but failed to import") from None

    row = Catalogs.query_criteria(catalog="Tic", ID=TIC_ID)[0]
    Teff = row["Teff"]
    logg = row["logg"]
    radius = row["rad"]
    radius_max = radius_min = row["e_rad"]
    mass = row["mass"]
    mass_max = mass_min = row["e_mass"]
    return Teff, logg, radius, radius_min, radius_max, mass, mass_min, mass_max


def _masked_to_none(value):
    return None if numpy.ma.is_masked(value) else value


def catalog_info(EPIC_ID=None, TIC_ID=None, KIC_ID=None):
    """Takes EPIC ID, returns limb darkening parameters u (linear) and
    a,b (quadratic), and stellar parameters. Values are pulled for minimum
    absolute deviation between given/catalog Teff and logg. Data are from:
    - K2 Ecliptic Plane Input Catalog, Huber+ 2016, 2016ApJS..224....2H
    - New limb-darkening coefficients, Claret+ 2012, 2013,
      2012A&A...546A..14C, 2013A&A...552A..16C"""

    given = [x is not None for x in (EPIC_ID, TIC_ID, KIC_ID)]
    if not any(given):
        raise ValueError("No ID was given")
    if sum(given) > 1:
        raise ValueError("Only one ID allowed")

    if KIC_ID is not None:  # Kepler K1
        info = catalog_info_KIC(KIC_ID)
    elif EPIC_ID is not None:  # Kepler K2
        info = catalog_info_EPIC(EPIC_ID)
    else:  # TESS
        info = catalog_info_TIC(TIC_ID)
    Teff, logg, radius, radius_min, radius_max, mass, mass_min, mass_max = info

    if TIC_ID is not None:
        ld = numpy.genfromtxt(
            path.join(tls_constants.resources_dir, "ld_claret_tess.csv"),
            skip_header=1,
            delimiter=",",
            dtype="f8, int32, f8, f8",
            names=["logg", "Teff", "a", "b"],
        )
    else:  # Limb darkening is the same for K1 (KIC) and K2 (EPIC)
        ld = numpy.genfromtxt(
            path.join(tls_constants.resources_dir, "JAA546A14limb1-4.csv"),
            skip_header=1,
            delimiter=",",
            dtype="f8, int32, f8, f8, f8",
            names=["logg", "Teff", "u", "a", "b"],
        )

    logg = _masked_to_none(logg)
    Teff = _masked_to_none(Teff)
    if logg is None:
        logg = 4
        warnings.warn("No logg in catalog. Proceeding with logg=4")
    if Teff is None:
        Teff = 6000
        warnings.warn("No Teff in catalog. Proceeding with Teff=6000")

    # From here on, K2 and TESS catalogs work the same:
    # - Take Teff from star catalog and find nearest entry in LD catalog
    # - Same for logg, but only for the Teff values returned before
    # - Return stellar parameters and best-match LD
    nearest_Teff = ld["Teff"][(numpy.abs(ld["Teff"] - Teff)).argmin()]
    idx_all_Teffs = numpy.where(ld["Teff"] == nearest_Teff)
    relevant_lds = numpy.copy(ld[idx_all_Teffs])
    idx_nearest = numpy.abs(relevant_lds["logg"] - logg).argmin()
    a = relevant_lds["a"][idx_nearest]
    b = relevant_lds["b"][idx_nearest]

    def clean(value):
        value = numpy.array(value)
        return numpy.nan if value == 0.0 else value

    return (
        (a, b),
        clean(mass),
        clean(mass_min),
        clean(mass_max),
        clean(radius),
        clean(radius_min),
        clean(radius_max),
    )

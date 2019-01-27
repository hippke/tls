import numpy
from os import path
import tls_constants


def catalog_info(EPIC_ID=None, TIC_ID=None, KIC_ID=None):

    """Takes EPIC ID, returns limb darkening parameters u (linear) and
        a,b (quadratic), and stellar parameters. Values are pulled for minimum
        absolute deviation between given/catalog Teff and logg. Data are from:
        - K2 Ecliptic Plane Input Catalog, Huber+ 2016, 2016ApJS..224....2H
        - New limb-darkening coefficients, Claret+ 2012, 2013,
          2012A&A...546A..14C, 2013A&A...552A..16C"""

    if (EPIC_ID is None) and (TIC_ID is None) and (KIC_ID is None):
        raise ValueError("No ID was given")
    if (EPIC_ID is not None) and (TIC_ID is not None):
        raise ValueError("Only one ID allowed")
    if (EPIC_ID is not None) and (KIC_ID is not None):
        raise ValueError("Only one ID allowed")
    if (TIC_ID is not None) and (KIC_ID is not None):
        raise ValueError("Only one ID allowed")

    # KOI CASE (Kepler K1)
    if KIC_ID is not None:
        try:
            from astroquery.vizier import Vizier
        except:
            raise ImportError(
                'Package astroquery.vizier required for KIC_ but failed to import'
            )
        if type(KIC_ID) is not int:
            raise TypeError(
                'KIC_ID ID must be of type "int"'
            )
        columns = ["Teff", "log(g)", "Rad", "E_Rad", "e_Rad", "Mass", "E_Mass", "e_Mass"]
        catalog = "J/ApJS/229/30/catalog"
        result = Vizier(columns=columns).query_constraints(KIC=KIC_ID, catalog=catalog)[0].as_array()
        Teff = result[0][0]
        logg = result[0][1]
        radius = result[0][2]
        radius_max = result[0][3]
        radius_min = result[0][4]
        mass = result[0][5]
        mass_max = result[0][6]
        mass_min = result[0][7]

    # EPIC CASE (Kepler K2)
    if EPIC_ID is not None:
        if type(EPIC_ID) is not int:
            raise TypeError(
                'EPIC_ID ID must be of type "int"'
            )
        if (EPIC_ID < 201000001) or (EPIC_ID > 251813738):
            raise TypeError(
                "EPIC_ID ID must be in range 201000001 to 251813738"
            )

        try:
            from astroquery.vizier import Vizier
        except:
            raise ImportError(
                'Package astroquery.vizier required for EPIC_ID but failed to import'
            )

        columns = ["Teff", "logg", "Rad", "E_Rad", "e_Rad", "Mass", "E_Mass", "e_Mass"]
        catalog = "IV/34/epic"
        result = Vizier(columns=columns).query_constraints(ID=EPIC_ID, catalog=catalog)[0].as_array()
        Teff = result[0][0]
        logg = result[0][1]
        radius = result[0][2]
        radius_max = result[0][3]
        radius_min = result[0][4]
        mass = result[0][5]
        mass_max = result[0][6]
        mass_min = result[0][7]

    # Kepler limb darkening for EPIC and KIC, load from locally saved CSV file
    if EPIC_ID is not None or KIC_ID is not None:
        ld = numpy.genfromtxt(
            path.join(
                tls_constants.resources_dir, "JAA546A14limb1-4.csv"
            ),
            skip_header=1,
            delimiter=",",
            dtype="f8, int32, f8, f8, f8",
            names=["logg", "Teff", "u", "a", "b"],
        )

    # TESS CASE
    if TIC_ID is not None:
        if type(TIC_ID) is not int:
            raise TypeError(
                'TIC_ID ID must be of type "int"'
            )

        try:
            from astroquery.mast import Catalogs
        except:
            raise ImportError(
                'Package astroquery.mast required for EPIC_ID but failed to import'
            )

        result = Catalogs.query_criteria(catalog="Tic", ID=TIC_ID).as_array()
        Teff = result[0][64]
        logg = result[0][66]
        radius = result[0][70]
        radius_max = result[0][71]
        radius_min = result[0][71]
        mass = result[0][72]
        mass_max = result[0][73]
        mass_min = result[0][73]

        ld = numpy.genfromtxt(
            path.join(tls_constants.resources_dir, "ld_claret_tess.csv"),
            skip_header=1,
            delimiter=";",
            dtype="f8, int32, f8, f8",
            names=["logg", "Teff", "a", "b"],
        )

        if logg is None:
            logg = 4
            warnings.warn(
                "No logg in catalog. Proceeding with logg=4"
            )

    """From here on, K2 and TESS catalogs work the same:
        - Take Teff from star catalog and find nearest entry in LD catalog
        - Same for logg, but only for the Teff values returned before
        - Return stellar parameters and best-match LD
    """
    #if KIC_ID is None:
    # Find nearest Teff and logg
    nearest_Teff = ld["Teff"][
        (numpy.abs(ld["Teff"] - Teff)).argmin()
    ]
    idx_all_Teffs = numpy.where(ld["Teff"] == nearest_Teff)
    relevant_lds = numpy.copy(ld[idx_all_Teffs])
    idx_nearest = numpy.abs(
        relevant_lds["logg"] - logg
    ).argmin()
    a = relevant_lds["a"][idx_nearest]
    b = relevant_lds["b"][idx_nearest]


    #p

    return (
        (a, b),
        float(mass),
        float(mass_min),
        float(mass_max),
        float(radius),
        float(radius_min),
        float(radius_max),
    )

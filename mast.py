import kplr
import k2plr
import numpy
import json
import sys
import warnings
try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
    
try: # Python 3.x
    import http.client as httplib 
except ImportError:  # Python 2.x
    import httplib


def mastQuery(request):
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+".".join(map(str, sys.version_info[:3]))}
    conn = httplib.HTTPSConnection('mast.stsci.edu')
    conn.request("POST", "/api/v0/invoke", "request="+urlencode(json.dumps(request)), headers)
    response = conn.getresponse()
    header = response.getheaders()
    content = response.read().decode('utf-8')
    conn.close()
    return header, content


def get_tic_data(TIC_ID):
    adv_filters=[{"paramName":"ID", "values":[{"min":TIC_ID,"max":TIC_ID}]}]
    headers, outString = mastQuery({
        "service":"Mast.Catalogs.Filtered.Tic",
        "format":"json",
        "params":{"columns":"c.*", "filters":adv_filters}})
    return json.loads(outString)['data']


def catalog_info(EPIC_ID=None, TIC_ID=None):
    """Takes EPIC ID, returns limb darkening parameters u (linear) and
        a,b (quadratic), and stellar parameters. Values are pulled for minimum
        absolute deviation between given/catalog Teff and logg. Data are from:
        - K2 Ecliptic Plane Input Catalog, Huber+ 2016, 2016ApJS..224....2H
        - New limb-darkening coefficients, Claret+ 2012, 2013,
          2012A&A...546A..14C, 2013A&A...552A..16C"""

    if (EPIC_ID is None) and (TIC_ID is None):
        raise ValueError('No ID was given')
    if (EPIC_ID is not None) and (TIC_ID is not None):
        raise ValueError('Only one ID allowed')

    # EPIC CASE
    if (EPIC_ID is not None):
        if type(EPIC_ID) is not int:
            raise TypeError('EPIC_ID ID must be of type "int"')
        if (EPIC_ID<201000001) or (EPIC_ID>251813738):
            raise TypeError('EPIC_ID ID must be in range 201000001 to 251813738')

        # EPIC K2 catalog, load from locally saved CSV file
        star = numpy.genfromtxt(
            "k2cat.tsv",
            skip_header=1,
            delimiter=";",
            dtype="int32, int32, f8, f8, f8, f8, f8, f8, f8",
            names=[
                "EPIC_ID",
                "Teff",
                "logg",
                "radius",
                "E_radius",
                "e_radius",
                "mass",
                "E_mass",
                "e_mass",
            ],
        )

        # Kepler limb darkening, load from locally saved CSV file
        ld = numpy.genfromtxt(
            "JAA546A14limb1-4.csv",
            skip_header=1,
            delimiter=",",
            dtype="f8, int32, f8, f8, f8",
            names=["logg", "Teff", "u", "a", "b"],
        )

        # Find row in EPIC catalog
        idx = numpy.where(star["EPIC_ID"] == EPIC_ID)
        if numpy.size(idx) == 0:
            raise ValueError("EPIC_ID not in catalog")

        Teff = star["Teff"][idx]
        logg = star["logg"][idx]
        radius = star["radius"][idx]
        radius_max = star["E_radius"][idx]
        radius_min = star["e_radius"][idx]
        mass = star["mass"][idx]
        mass_max = star["E_mass"][idx]
        mass_min = star["e_mass"][idx]

    # TESS CASE
    if (TIC_ID is not None):
        if type(TIC_ID) is not int:
            raise TypeError('TIC_ID ID must be of type "int"')

        # Load entry for TESS Input Catalog from MAST
        tic_data = get_tic_data(TIC_ID)

        if len(tic_data)!=1:
            raise TypeError('TIC_ID not in catalog')

        star = tic_data[0]
        ld = numpy.genfromtxt(
            "ld_claret_tess.csv",
            skip_header=1,
            delimiter=";",
            dtype="f8, int32, f8, f8",
            names=["logg", "Teff", "a", "b"],
        )
        Teff = star["Teff"]
        logg = star["logg"]
        radius = star["rad"]
        radius_max = star["e_rad"]  # only one uncertainty is provided
        radius_min = star["e_rad"]
        mass = star["mass"]
        mass_max = star["e_mass"]  # only one uncertainty is provided
        mass_min = star["e_mass"]

        if logg is None:
            logg = 4
            warnings.warn("No logg in catalog. Proceeding with logg=4")

    """From here on, all catalogs should work the same:
        - Take Teff from star catalog and find nearest entry in LD catalog
        - Same for logg, but only for the Teff values returned before
        - Return stellar parameters and best-match LD
    """

    # Find nearest Teff and logg
    nearest_Teff = ld["Teff"][(numpy.abs(ld["Teff"] - Teff)).argmin()]
    idx_all_Teffs = numpy.where(ld["Teff"] == nearest_Teff)
    relevant_lds = numpy.copy(ld[idx_all_Teffs])
    idx_nearest = numpy.abs(relevant_lds["logg"] - logg).argmin()
    nearest_logg = relevant_lds["logg"][idx_nearest]
    a = relevant_lds["a"][idx_nearest]
    b = relevant_lds["b"][idx_nearest]

    # The EPIC catalog was reduced from an array. The return values shall be floats.
    if (EPIC_ID is not None):
        mass = mass[0]
        mass_min = mass_min[0]
        mass_max = mass_max[0]
        radius = radius[0]
        radius_min = radius_min[0]
        radius_max = radius_max[0]

    return (
        (a, b),
        mass,
        mass_min,
        mass_max,
        radius,
        radius_min,
        radius_max,
    )




print(catalog_info(TIC_ID=290131778))
print(catalog_info(EPIC_ID=211611158))

ab, R_star, R_star_min, R_star_max, M_star, M_star_min, M_star_max = catalog_info(EPIC_ID=211611158)
print('Quadratic limb darkening a, b', ab[0], ab[1])
print('Stellar radius', R_star, '+', R_star_max, '-', R_star_min)
print('Stellar mass', M_star, '+', M_star_max, '-', M_star_min)

#  Optimized algorithm to search for transits of small extrasolar planets
#                                                                            /
#       ,        AUTHORS                                                   O/
#    \  :  /     Michael Hippke (1) [michael@hippke.org]                /\/|
# `. __/ \__ .'  Rene' Heller (2) [heller@mps.mpg.de]                      |
# _ _\     /_ _  _________________________________________________________/ \_
#    /_   _\
#  .'  \ /  `.   (1) Sonneberg Observatory, Sternwartestr. 32, Sonneberg
#    /  :  \     (2) Max Planck Institute for Solar System Research,
#       '            Justus-von-Liebig-Weg 3, 37077 G\"ottingen, Germany

from transitleastsquares.backends import (
    SearchBackend,
    available_backends,
    get_backend,
    register_backend,
)
from transitleastsquares.catalog import catalog_info
from transitleastsquares.core import fold
from transitleastsquares.grid import duration_grid, period_grid
from transitleastsquares.helpers import cleaned_array, resample, transit_mask
from transitleastsquares.main import transitleastsquares
from transitleastsquares.stats import FAP
from transitleastsquares.version import __version__

__all__ = [
    "__version__",
    "transitleastsquares",
    "cleaned_array",
    "resample",
    "transit_mask",
    "duration_grid",
    "period_grid",
    "FAP",
    "catalog_info",
    "fold",
    "SearchBackend",
    "available_backends",
    "get_backend",
    "register_backend",
]

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

from __future__ import division, print_function
from .main import transitleastsquares
from .helpers import cleaned_array, resample, transit_mask
from .stats import FAP
from .catalog import catalog_info
from .core import fold
from .template_generator.transit_template_generator import TransitTemplateGenerator
from .template_generator.default_transit_template_generator import DefaultTransitTemplateGenerator
from .results import transitleastsquaresresults
from . import version

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
from transitleastsquares.main import transitleastsquares
from transitleastsquares.helpers import cleaned_array, resample, transit_mask
from transitleastsquares.grid import period_grid
from transitleastsquares.stats import FAP
from transitleastsquares.catalog import catalog_info
from transitleastsquares.core import fold
from transitleastsquares.template_generator.transit_template_generator import TransitTemplateGenerator
from transitleastsquares.template_generator.default_transit_template_generator import DefaultTransitTemplateGenerator
from transitleastsquares.results import transitleastsquaresresults
from transitleastsquares import version

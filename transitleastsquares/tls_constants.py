"""Constants and defaults of Transit Least Squares (TLS).

Users may override module attributes at runtime (``tls_constants.X = ...``);
all code reads them at call time.
"""

from os import path

from transitleastsquares import version as tls_version

resources_dir = path.dirname(__file__)
TLS_VERSION = (
    f"Transit Least Squares TLS {tls_version.TLS_VERSIONING} ({tls_version.TLS_DATE})"
)

# In the default, print status information during search
VERBOSE = True

# Astrophysical constants
G = 6.673e-11  # gravitational constant [m^3 / kg / s^2]
R_sun = 695508000  # radius of the Sun [m]
R_earth = 6371000  # radius of the Earth [m]
R_jup = 69911000  # radius of Jupiter [m]
M_sun = 1.989 * 10**30  # mass of the Sun [kg]
SECONDS_PER_DAY = 86400

# Default values as described in the paper
TRANSIT_DEPTH_MIN = 10 * 10**-6  # 10 ppm
NUMERICAL_STABILITY_CUTOFF = 0.01 * 10**-6  # to identify nominal flux in model

# Period grid
R_STAR = 1.0
M_STAR = 1.0
OVERSAMPLING_FACTOR = 3
N_TRANSITS_MIN = 2

# Duration grid
M_STAR_MIN = 0.1
M_STAR_MAX = 1.0
R_STAR_MIN = 0.13
R_STAR_MAX = 3.5
DURATION_GRID_STEP = 1.1

# Transit template: quadratic limb darkening for a G2V star in the Kepler bandpass
# http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BA/552/A16
DEFAULT_U = [0.4804, 0.1867]
DEFAULT_LIMB_DARK = "quadratic"
DEFAULT_ECC = 0  # eccentricity
DEFAULT_W = 90  # longitude of periastron (in degrees)
DEFAULT_PERIOD = 12.9
DEFAULT_RP = 0.03
DEFAULT_A = 23.1
DEFAULT_INC = 89.21

# Grazing template
GRAZING_B = 0.99

# Box-shaped (steep trapezoid) template
BOX_PERIOD = 29
BOX_RP = 0.1
BOX_A = 26.9
BOX_B = 0
BOX_INC = 90
BOX_U = [0]
BOX_LIMB_DARK = "linear"

# Transit model for the templates: "tls" (own implementation, transit_model.py)
# or "batman" (needs batman-package; used by TLS <= 1.33, kept for comparisons)
TRANSIT_MODEL = "tls"

# Unique depth of trial signals (at various durations). These are rescaled in
# depth so that their integral matches the mean flux in the window in question.
# In principle, "signal_depth" is an arbitrary value >0 and <1
SIGNAL_DEPTH = 0.5

# Maximum fractional transit duration ever observed is 0.117
# for Kepler-1368 b (as of Oct 2018), so we set upper_limit=0.12
# Long fractional transit durations are computationally expensive
# following a quadratic relation. If applicable, use a different value.
# Longer transits can still be found, but at decreasing sensitivity
FRACTIONAL_TRANSIT_DURATION_MAX = 0.12

# Oversampling ==> Downsampling of the reference transit:
# "Do not fit an unbinned model to binned data."
# Reference: Kipping, D., "Binning is sinning: morphological light-curve
#            distortions due to finite integration time"
#            MNRAS, Volume 408, Issue 3, pp. 1758-1769
#            http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1004.3741
# This is not time-critical as it has to be done only once
# To work for all periods correctly, it has to be re-done at each period
# This feature is currently not implemented
SUPERSAMPLE_SIZE = 10000
OVERSAMPLE_MODEL_LIGHT_CURVE = 5

# Order in which the periods are searched: "shuffled", "descending", "ascending"
# Shuffled has the advantage of the best estimate for the remaining time.
# The order has no influence on the results.
PERIODS_SEARCH_ORDER = "shuffled"

# When converting power_raw to power, a median of a certain window size is subtracted.
# For periodograms of smaller width, no smoothing is applied. The kernel size is
# calculated as kernel = oversampling_factor * SDE_MEDIAN_KERNEL_SIZE
# This value has proven to yield numerically stable results.
SDE_MEDIAN_KERNEL_SIZE = 30

# T0 margins, in units of the transit duration (0.01 = 1 %).
#
# T0_FIT_MARGIN: resolution of the final T0 fit after the search. The phase
#   space is scanned in steps of T0_FIT_MARGIN * duration. 0 = every cadence.
#   Useful in large datasets. 100k points: Extra runtime of order 2 minutes.
#
# The same margin is ALSO used during the period search (unless the parameter
# T0_search_margin is given): for trial templates wider than 1/margin cadences
# (100 cadences at 1 %), only every (width * margin)-th phase shift is tested.
# This affects the SDE (slightly) and the speed (strongly, for long templates /
# dense data). Note: older TLS documentation stated that T0_FIT_MARGIN affects
# only the final T0 fit; this was never the case.
T0_FIT_MARGIN = 0.01

# The secondary fit for T0 can take negligible or significant time, depending on the
# number of data points and on T0_FIT_MARGIN. Set an empirical threshold to avoid
# displaying a progress bar when the estimated runtime is low (<~1 sec)
# To display the progress bar in more cases, use a lower number
PROGRESSBAR_THRESHOLD = 5000

# Some cases yield empty period grids. Example: R_star=5, M_star=1
# Then: Warn and return the default grid
MINIMUM_PERIOD_GRID_SIZE = 100

# Search backend used by default (see transitleastsquares.backends):
# "fused" (fast, default) or "numba" (reference implementation of TLS 1.33).
# Can be overridden per call (power(backend=...)) or by the environment
# variable TLS_BACKEND.
DEFAULT_BACKEND = "fused"

# Warn the user if unknown **kwargs are given as parameters
VALID_PARAMETERS = [
    "R_star",
    "R_star_min",
    "R_star_max",
    "M_star",
    "M_star_min",
    "M_star_max",
    "period_min",
    "period_max",
    "n_transits_min",
    "per",
    "rp",
    "a",
    "inc",
    "b",
    "ecc",
    "w",
    "u",
    "limb_dark",
    "duration_grid_step",
    "transit_depth_min",
    "oversampling_factor",
    "T0_fit_margin",
    "T0_search_margin",
    "use_threads",
    "show_progress_bar",
    "transit_template",
    "verbose",
    "backend",
]

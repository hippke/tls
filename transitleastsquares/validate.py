"""Validation of inputs (data) and of the search parameters of power()."""

import multiprocessing
import warnings

import numpy

from transitleastsquares import tls_constants
from transitleastsquares.helpers import cleaned_array, impact_to_inclination


def validate_inputs(t, y, dy):
    """Clean (t, y, dy) and check their consistency. Returns float arrays."""
    if dy is None:
        t, y = cleaned_array(t, y)
    else:
        t, y, dy = cleaned_array(t, y, dy)
        # Normalize dy to act as weights in least squares calculation
        dy = dy / numpy.mean(dy)

    if numpy.size(y) < 3 or numpy.size(t) < 3:
        raise ValueError("Too few values in data set")
    if max(t) - min(t) <= 0:
        raise ValueError("Time duration must positive")
    if numpy.mean(y) > 1.01 or numpy.mean(y) < 0.99:
        warnings.warn(
            "Warning: The mean flux should be normalized to 1, but it was found "
            f"to be {numpy.mean(y)}"
        )
    if min(y) < 0:
        raise ValueError("Flux values must be positive")
    if max(y) >= float("inf"):
        raise ValueError("Flux values must be finite")

    # If no dy is given, create it with the standard deviation of the flux
    if dy is None:
        dy = numpy.full(len(y), numpy.std(y))
    if numpy.size(t) != numpy.size(y) or numpy.size(t) != numpy.size(dy):
        raise ValueError("Arrays (t, y, dy) must be of the same dimensions")
    if t.ndim != 1:  # Size identity ensures dimensional identity
        raise ValueError("Inputs (t, y, dy) must be 1-dimensional")
    return t, y, dy


def _check_positive_finite(name, value):
    if value <= 0 or value >= float("inf"):
        raise ValueError(f"{name} must be positive")


def validate_args(self, kwargs):
    """Validate **kwargs of power(), set defaults where missing, and store
    every parameter as an attribute of the model object `self`."""

    self.verbose = kwargs.get(
        "verbose", getattr(self, "_verbose_init", tls_constants.VERBOSE)
    )

    # Warn user if unknown parameters
    for key in kwargs:
        if key not in tls_constants.VALID_PARAMETERS:
            warnings.warn(f"Ignoring unknown parameter: {key}")

    self.show_progress_bar = kwargs.get("show_progress_bar", True)
    self.transit_depth_min = kwargs.get(
        "transit_depth_min", tls_constants.TRANSIT_DEPTH_MIN
    )
    self.R_star = kwargs.get("R_star", tls_constants.R_STAR)
    self.M_star = kwargs.get("M_star", tls_constants.M_STAR)
    self.oversampling_factor = kwargs.get(
        "oversampling_factor", tls_constants.OVERSAMPLING_FACTOR
    )
    self.period_max = kwargs.get("period_max", float("inf"))
    self.period_min = kwargs.get("period_min", 0)
    self.n_transits_min = kwargs.get("n_transits_min", tls_constants.N_TRANSITS_MIN)

    self.R_star_min = kwargs.get("R_star_min", tls_constants.R_STAR_MIN)
    self.R_star_max = kwargs.get("R_star_max", tls_constants.R_STAR_MAX)
    self.M_star_min = kwargs.get("M_star_min", tls_constants.M_STAR_MIN)
    self.M_star_max = kwargs.get("M_star_max", tls_constants.M_STAR_MAX)
    self.duration_grid_step = kwargs.get(
        "duration_grid_step", tls_constants.DURATION_GRID_STEP
    )

    self.use_threads = kwargs.get("use_threads", multiprocessing.cpu_count())
    self.backend = kwargs.get("backend", None)

    self.per = kwargs.get("per", tls_constants.DEFAULT_PERIOD)
    self.rp = kwargs.get("rp", tls_constants.DEFAULT_RP)
    self.a = kwargs.get("a", tls_constants.DEFAULT_A)

    self.T0_fit_margin = kwargs.get("T0_fit_margin", tls_constants.T0_FIT_MARGIN)
    self.T0_search_margin = kwargs.get("T0_search_margin", None)

    # If an impact parameter is given, it overrules the supplied inclination
    if "b" in kwargs:
        self.b = kwargs.get("b")
        self.inc = impact_to_inclination(b=self.b, semimajor_axis=self.a)
    else:
        self.inc = kwargs.get("inc", tls_constants.DEFAULT_INC)

    self.ecc = kwargs.get("ecc", tls_constants.DEFAULT_ECC)
    self.w = kwargs.get("w", tls_constants.DEFAULT_W)
    self.u = kwargs.get("u", tls_constants.DEFAULT_U)
    self.limb_dark = kwargs.get("limb_dark", tls_constants.DEFAULT_LIMB_DARK)

    self.transit_template = kwargs.get("transit_template", "default")
    if self.transit_template == "default":
        # User-supplied shape parameters take precedence over the default template
        if "per" not in kwargs:
            self.per = tls_constants.DEFAULT_PERIOD
        if "rp" not in kwargs:
            self.rp = tls_constants.DEFAULT_RP
        if "a" not in kwargs:
            self.a = tls_constants.DEFAULT_A
        if "inc" not in kwargs and "b" not in kwargs:
            self.inc = tls_constants.DEFAULT_INC

    elif self.transit_template == "grazing":
        self.b = tls_constants.GRAZING_B
        self.inc = impact_to_inclination(b=self.b, semimajor_axis=self.a)

    elif self.transit_template == "box":
        self.per = tls_constants.BOX_PERIOD
        self.rp = tls_constants.BOX_RP
        self.a = tls_constants.BOX_A
        self.b = tls_constants.BOX_B
        self.inc = tls_constants.BOX_INC
        self.u = tls_constants.BOX_U
        self.limb_dark = tls_constants.BOX_LIMB_DARK

    else:
        raise ValueError(
            'Unknown transit_template. Known values: "default", "grazing", "box"'
        )

    # Validations to avoid (garbage in ==> garbage out)

    # Stellar radius: 0 < R_star_min <= R_star <= R_star_max < inf
    _check_positive_finite("R_star", self.R_star)
    if self.R_star_min > self.R_star:
        raise ValueError("R_star_min <= R_star is required")
    _check_positive_finite("R_star_min", self.R_star_min)
    if self.R_star_max < self.R_star:
        raise ValueError("R_star_max >= R_star is required")
    _check_positive_finite("R_star_max", self.R_star_max)

    # Stellar mass: 0 < M_star_min <= M_star <= M_star_max < inf
    _check_positive_finite("M_star", self.M_star)
    if self.M_star_min > self.M_star:
        raise ValueError("M_star_min <= M_star is required")
    _check_positive_finite("M_star_min", self.M_star_min)
    if self.M_star_max < self.M_star:
        raise ValueError("M_star_max >= M_star required")
    _check_positive_finite("M_star_max", self.M_star_max)

    # Period grid
    if self.period_min < 0:
        raise ValueError("period_min >= 0 required")
    if self.period_min >= self.period_max:
        raise ValueError("period_min < period_max required")
    if not isinstance(self.n_transits_min, int):
        raise ValueError("n_transits_min must be an integer value")
    if self.n_transits_min < 1:
        raise ValueError("n_transits_min must be an integer value >= 1")

    if not isinstance(self.use_threads, int) or self.use_threads < 1:
        raise ValueError("use_threads must be an integer value >= 1")

    # 0 <= T0 margins <= 0.1 (sensible limit: 10% of transit duration)
    self.T0_fit_margin = _clamp_margin(self.T0_fit_margin)
    if self.T0_search_margin is None:
        self.T0_search_margin = self.T0_fit_margin
    else:
        self.T0_search_margin = _clamp_margin(self.T0_search_margin)
    return self, kwargs


def _clamp_margin(value):
    return min(max(value, 0), 0.1)

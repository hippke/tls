"""Period and duration grids."""

import warnings

import numba
import numpy
from numpy import pi, sqrt

from transitleastsquares import tls_constants


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def T14(
    R_s, M_s, P, upper_limit=tls_constants.FRACTIONAL_TRANSIT_DURATION_MAX, small=False
):
    """Input:  Stellar radius and mass; planetary period
            Units: Solar radius and mass; days
    Output: Maximum planetary transit duration T_14max
            Unit: Fraction of period P"""

    P = P * tls_constants.SECONDS_PER_DAY
    R_s = tls_constants.R_sun * R_s
    M_s = tls_constants.M_sun * M_s

    if small:  # small planet assumption
        T14max = R_s * ((4 * P) / (pi * tls_constants.G * M_s)) ** (1 / 3)
    else:  # planet size 2 R_jup
        T14max = (R_s + 2 * tls_constants.R_jup) * (
            (4 * P) / (pi * tls_constants.G * M_s)
        ) ** (1 / 3)

    result = T14max / P
    if result > upper_limit:
        result = upper_limit
    return result


def _density(M, R):
    return M / (R * R * R)


# Stellar densities (solar units) of the default corners: the smallest star
# (R_STAR_MIN, M_STAR_MIN) and the largest one (R_STAR_MAX, M_STAR_MAX).
RHO_DEFAULT_MAX = _density(tls_constants.M_STAR_MIN, tls_constants.R_STAR_MIN)
RHO_DEFAULT_MIN = _density(tls_constants.M_STAR_MAX, tls_constants.R_STAR_MAX)


def duration_limit_masses(R_star_min, R_star_max, M_star_min, M_star_max):
    """Masses to pair with R_star_min (shortest) and R_star_max (longest duration).

    T14 ~ R * M^(-1/3) ~ (P / rho)^(1/3). The interval extremes are (R_min, M_max)
    for the shortest and (R_max, M_min) for the longest duration. TLS <= 1.33
    paired (R_min, M_min) and (R_max, M_max) instead (BUGS.md F1), which is too
    narrow for narrow user priors. But for wide intervals the extreme corners are
    unphysical (R 0.13 with M 1.0 is 455 rho_sun). So the extreme density is
    clipped to the more extreme of the old corner and the default corner
    (RHO_DEFAULT_MAX / _MIN). Hence: default limits give the TLS 1.33 grid exactly,
    narrow priors get the full interval range, and no range is ever narrower than
    in TLS 1.33.

    Returns (m_short, m_long).
    """
    # shortest duration: highest density
    rho_cap = max(_density(M_star_min, R_star_min), RHO_DEFAULT_MAX)
    if _density(M_star_max, R_star_min) <= rho_cap:
        m_short = M_star_max
    elif _density(M_star_min, R_star_min) >= RHO_DEFAULT_MAX:
        m_short = M_star_min  # old corner (exact)
    else:
        m_short = RHO_DEFAULT_MAX * R_star_min**3
    # longest duration: lowest density
    rho_floor = min(_density(M_star_max, R_star_max), RHO_DEFAULT_MIN)
    if _density(M_star_min, R_star_max) >= rho_floor:
        m_long = M_star_min
    elif _density(M_star_max, R_star_max) <= RHO_DEFAULT_MIN:
        m_long = M_star_max  # old corner (exact)
    else:
        m_long = RHO_DEFAULT_MIN * R_star_max**3
    return m_short, m_long


def duration_grid(
    periods,
    shortest,
    log_step=tls_constants.DURATION_GRID_STEP,
    R_star_min=tls_constants.R_STAR_MIN,
    R_star_max=tls_constants.R_STAR_MAX,
    M_star_min=tls_constants.M_STAR_MIN,
    M_star_max=tls_constants.M_STAR_MAX,
):
    """Logarithmic grid of trial durations (fractions of the period).

    ``shortest`` is accepted for backwards compatibility and unused.
    """
    m_short, m_long = duration_limit_masses(
        R_star_min, R_star_max, M_star_min, M_star_max
    )
    duration_max = T14(
        R_s=R_star_max, M_s=m_long, P=numpy.min(periods), small=False
    )  # large planet for long transit duration
    duration_min = T14(
        R_s=R_star_min, M_s=m_short, P=numpy.max(periods), small=True
    )  # small planet for short transit duration

    durations = [duration_min]
    current_depth = duration_min
    while current_depth * log_step < duration_max:
        current_depth = current_depth * log_step
        durations.append(current_depth)
    durations.append(duration_max)  # Append endpoint. Not perfectly spaced.
    return durations


def _clamp(name, value, lower, upper):
    if value < lower:
        warnings.warn(
            f"Warning: {name} was set to {lower} for period_grid "
            f"(was unphysical: {value})"
        )
        return lower
    if value > upper:
        warnings.warn(
            f"Warning: {name} was set to {upper} for period_grid "
            f"(was unphysical: {value})"
        )
        return upper
    return value


def period_grid(
    R_star,
    M_star,
    time_span,
    period_min=0,
    period_max=float("inf"),
    oversampling_factor=tls_constants.OVERSAMPLING_FACTOR,
    n_transits_min=tls_constants.N_TRANSITS_MIN,
    _is_fallback=False,
):
    """Returns array of optimal sampling periods for transit search in light curves
    Following Ofir (2014, A&A, 561, A138)"""

    R_star = _clamp("R_star", R_star, 0.01, 10000)
    M_star = _clamp("M_star", M_star, 0.01, 1000)

    R_star = R_star * tls_constants.R_sun
    M_star = M_star * tls_constants.M_sun
    time_span = time_span * tls_constants.SECONDS_PER_DAY  # seconds

    # boundary conditions
    f_min = n_transits_min / time_span
    f_max = 1.0 / (2 * pi) * sqrt(tls_constants.G * M_star / (3 * R_star) ** 3)

    # optimal frequency sampling, Equations (5), (6), (7)
    A = (
        (2 * pi) ** (2.0 / 3)
        / pi
        * R_star
        / (tls_constants.G * M_star) ** (1.0 / 3)
        / (time_span * oversampling_factor)
    )
    C = f_min ** (1.0 / 3) - A / 3.0
    N_opt = (f_max ** (1.0 / 3) - f_min ** (1.0 / 3) + A / 3) * 3 / A

    X = numpy.arange(N_opt) + 1
    f_x = (A / 3 * X + C) ** 3
    P_x = 1 / f_x

    # Cut to given (optional) selection of periods
    periods = P_x / tls_constants.SECONDS_PER_DAY
    selected_index = numpy.where(
        numpy.logical_and(periods > period_min, periods <= period_max)
    )
    number_of_periods = numpy.size(periods[selected_index])

    if number_of_periods > 10**6:
        warnings.warn(
            f"period_grid generates a very large grid ({number_of_periods}). "
            "Recommend to check physical plausibility for stellar mass, radius, "
            "and time series duration."
        )

    if number_of_periods < tls_constants.MINIMUM_PERIOD_GRID_SIZE and _is_fallback:
        # The fallback grid (R_star=M_star=1) is still small, e.g. because the
        # user requested a narrow [period_min, period_max]. Return it as is
        # rather than recursing forever or ignoring the user's period range.
        if number_of_periods == 0:
            raise ValueError(
                "Empty period grid. Check period_min, period_max and time span."
            )
        return periods[selected_index]

    if number_of_periods < tls_constants.MINIMUM_PERIOD_GRID_SIZE:
        if time_span < 5 * tls_constants.SECONDS_PER_DAY:
            time_span = 5 * tls_constants.SECONDS_PER_DAY
        warnings.warn(
            "period_grid defaults to R_star=1 and M_star=1 as given density yielded "
            "grid with too few values"
        )
        return period_grid(
            R_star=1,
            M_star=1,
            time_span=time_span / tls_constants.SECONDS_PER_DAY,
            period_min=period_min,
            period_max=period_max,
            oversampling_factor=oversampling_factor,
            n_transits_min=n_transits_min,
            _is_fallback=True,
        )
    return periods[selected_index]  # periods in [days]

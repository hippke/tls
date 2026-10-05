"""Reference backend "numba": the original TLS algorithm (core.search_period)."""

from transitleastsquares.backends.pool import ProcessPoolBackend
from transitleastsquares.core import search_period


class NumbaBackend(ProcessPoolBackend):
    name = "numba"

    def evaluate(self, problem, period):
        return search_period(
            period,
            problem.t,
            problem.y,
            problem.dy,
            problem.transit_depth_min,
            problem.R_star_min,
            problem.R_star_max,
            problem.M_star_min,
            problem.M_star_max,
            problem.lc_arr,
            problem.lc_cache_overview,
            problem.T0_search_margin,
        )

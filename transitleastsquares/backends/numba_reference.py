"""Reference backend: the original TLS algorithm (numba kernels in core.py,
one task per period on a multiprocessing pool)."""

import multiprocessing
from functools import partial

from transitleastsquares.backends import SearchBackend, SearchResult
from transitleastsquares.core import search_period


class NumbaBackend(SearchBackend):
    name = "numba"

    def search(self, problem, periods, use_threads=1, progress=None):
        worker = partial(
            search_period,
            t=problem.t,
            y=problem.y,
            dy=problem.dy,
            transit_depth_min=problem.transit_depth_min,
            R_star_min=problem.R_star_min,
            R_star_max=problem.R_star_max,
            M_star_min=problem.M_star_min,
            M_star_max=problem.M_star_max,
            lc_arr=problem.lc_arr,
            lc_cache_overview=problem.lc_cache_overview,
            T0_fit_margin=problem.T0_search_margin,
        )
        out_periods, out_chi2, out_rows, out_depths = [], [], [], []

        def collect(data):
            out_periods.append(data[0])
            out_chi2.append(data[1])
            out_rows.append(data[2])
            out_depths.append(data[3])
            if progress is not None:
                progress(1)

        if use_threads > 1:
            pool = multiprocessing.Pool(processes=use_threads)
            try:
                for data in pool.imap_unordered(worker, periods):
                    collect(data)
            finally:
                pool.close()
                pool.join()
        else:
            for period in periods:
                collect(worker(period))

        return SearchResult.from_unsorted(out_periods, out_chi2, out_rows, out_depths)

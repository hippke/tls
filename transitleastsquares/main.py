from __future__ import division, print_function
import multiprocessing
import numpy
import sys
import warnings
from functools import partial
from tqdm import tqdm

# TLS parts
from .template_generator.tailed_transit_template_generator import TailedTransitTemplateGenerator
from .template_generator.default_transit_template_generator import DefaultTransitTemplateGenerator
from . import tls_constants
from .core import (
    search_period,
)
from .validate import validate_inputs, validate_args


class transitleastsquares(object):
    """Compute the transit least squares of limb-darkened transit models"""

    def __init__(self, t, y, dy=None):
        self.t, self.y, self.dy = validate_inputs(t, y, dy)
        default_transit_template_generator = DefaultTransitTemplateGenerator()
        self.transit_template_generators = {"default": default_transit_template_generator,
                               "grazing": default_transit_template_generator,
                               "box": default_transit_template_generator,
                               "tailed": TailedTransitTemplateGenerator()}

    def power(self, **kwargs):
        """Compute the periodogram for a set of user-defined parameters"""

        print(tls_constants.TLS_VERSION)
        self, kwargs = validate_args(self, kwargs)
        transit_template_generator = self.transit_template_generators[self.transit_template]
        periods = self.period_grid if self.period_grid is not None else transit_template_generator.period_grid(
            R_star=self.R_star,
            M_star=self.M_star,
            time_span=numpy.max(self.t) - numpy.min(self.t),
            period_min=self.period_min,
            period_max=self.period_max,
            oversampling_factor=self.oversampling_factor,
            n_transits_min=self.n_transits_min,
        )

        durations = transit_template_generator.duration_grid(
            periods, shortest=1 / len(self.t), log_step=self.duration_grid_step
        )

        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        lc_cache_overview, lc_arr = transit_template_generator.get_cache(
            period_grid=periods,
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
        )

        print(
            "Searching "
            + str(len(self.y))
            + " data points, "
            + str(len(periods))
            + " periods from "
            + str(round(min(periods), 3))
            + " to "
            + str(round(max(periods), 3))
            + " days"
        )

        # Python 2 multiprocessing with "partial" doesn't work
        # For now, only single-threading in Python 2 is supported
        if sys.version_info[0] < 3:
            self.use_threads = 1
            warnings.warn("This TLS version supports no multithreading on Python 2")

        if self.use_threads == multiprocessing.cpu_count():
            print("Using all " + str(self.use_threads) + " CPU threads")
        else:
            print(
                "Using "
                + str(self.use_threads)
                + " of "
                + str(multiprocessing.cpu_count())
                + " CPU threads"
            )

        if self.show_progress_bar:
            bar_format = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} periods | {elapsed}<{remaining}"
            pbar = tqdm(total=numpy.size(periods), smoothing=0.3, bar_format=bar_format)

        if tls_constants.PERIODS_SEARCH_ORDER == "ascending":
            periods = reversed(periods)
        elif tls_constants.PERIODS_SEARCH_ORDER == "descending":
            pass  # it already is
        elif tls_constants.PERIODS_SEARCH_ORDER == "shuffled":
            periods = numpy.random.permutation(periods)
        else:
            raise ValueError("Unknown PERIODS_SEARCH_ORDER")

        # Result lists now (faster), convert to numpy array later
        test_statistic_periods = []
        test_statistic_residuals = []
        test_statistic_rows = []
        test_statistic_depths = []

        if self.use_threads > 1:  # Run multi-core search
            pool = multiprocessing.Pool(processes=self.use_threads)
            params = partial(
                search_period,
                transit_template_generator,
                t=self.t,
                y=self.y,
                dy=self.dy,
                transit_depth_min=self.transit_depth_min,
                R_star_min=self.R_star_min,
                R_star_max=self.R_star_max,
                M_star_min=self.M_star_min,
                M_star_max=self.M_star_max,
                lc_arr=lc_arr,
                lc_cache_overview=lc_cache_overview,
                T0_fit_margin=self.T0_fit_margin,
            )
            for data in pool.imap_unordered(params, periods):
                test_statistic_periods.append(data[0])
                test_statistic_residuals.append(data[1])
                test_statistic_rows.append(data[2])
                test_statistic_depths.append(data[3])
                if self.show_progress_bar:
                    pbar.update(1)
            pool.close()
        else:
            for period in periods:
                data = search_period(
                    transit_template_generator,
                    period=period,
                    t=self.t,
                    y=self.y,
                    dy=self.dy,
                    transit_depth_min=self.transit_depth_min,
                    R_star_min=self.R_star_min,
                    R_star_max=self.R_star_max,
                    M_star_min=self.M_star_min,
                    M_star_max=self.M_star_max,
                    lc_arr=lc_arr,
                    lc_cache_overview=lc_cache_overview,
                    T0_fit_margin=self.T0_fit_margin,
                )
                test_statistic_periods.append(data[0])
                test_statistic_residuals.append(data[1])
                test_statistic_rows.append(data[2])
                test_statistic_depths.append(data[3])
                if self.show_progress_bar:
                    pbar.update(1)

        if self.show_progress_bar:
            pbar.close()

        # imap_unordered delivers results in unsorted order ==> sort
        test_statistic_periods = numpy.array(test_statistic_periods)
        sort_index = numpy.argsort(test_statistic_periods)
        test_statistic_periods = test_statistic_periods[sort_index]
        test_statistic_residuals = numpy.array(test_statistic_residuals)[sort_index]
        test_statistic_rows = numpy.array(test_statistic_rows)[sort_index]
        test_statistic_depths = numpy.array(test_statistic_depths)[sort_index]

        idx_best = numpy.argmin(test_statistic_residuals)
        best_row = test_statistic_rows[idx_best]
        duration = lc_cache_overview["duration"][best_row]
        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.t))

        if max(test_statistic_residuals) == min(test_statistic_residuals):
            no_transits_were_fit = True
            warnings.warn('No transit were fit. Try smaller "transit_depth_min"')
        else:
            no_transits_were_fit = False

        # Power spectra variants
        chi2 = test_statistic_residuals
        degrees_of_freedom = 4
        chi2red = test_statistic_residuals / (len(self.t) - degrees_of_freedom)
        chi2_min = numpy.min(chi2)
        chi2red_min = numpy.min(chi2red)

        return transit_template_generator.calculate_results(no_transits_were_fit, chi2, chi2red, chi2_min,
                                                            chi2red_min, test_statistic_periods, test_statistic_depths,
                                                            self, lc_arr, best_row, periods,
                                                            durations, duration, maxwidth_in_samples)


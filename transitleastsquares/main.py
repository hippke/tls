"""The transitleastsquares model class: data in, power() -> results."""

import multiprocessing
import warnings

import numpy
from tqdm import tqdm

from transitleastsquares import tls_constants
from transitleastsquares.backends import SearchProblem, get_backend
from transitleastsquares.core import fold
from transitleastsquares.grid import duration_grid, duration_limit_masses, period_grid
from transitleastsquares.helpers import transit_mask
from transitleastsquares.results import transitleastsquaresresults
from transitleastsquares.stats import (
    FAP,
    all_transit_times,
    calculate_fill_factor,
    calculate_stretch,
    calculate_transit_duration_in_days,
    count_stats,
    intransit_stats,
    model_lightcurve,
    period_uncertainty,
    rp_rs_from_depth,
    snr_stats,
    spectra,
)
from transitleastsquares.transit import fractional_transit, get_cache
from transitleastsquares.validate import validate_args, validate_inputs
from transitleastsquares.warmup import ensure_compiled

DEGREES_OF_FREEDOM = 4  # period, T0, duration, depth


class transitleastsquares:
    """Compute the transit least squares of limb-darkened transit models.

    model = transitleastsquares(t, y, dy=None, verbose=True)
    results = model.power(**parameters)
    """

    def __init__(self, t, y, dy=None, verbose=True):
        self.t, self.y, self.dy = validate_inputs(t, y, dy)
        self.verbose = verbose
        self._verbose_init = verbose  # default for power(verbose=...)

    # ------------------------------------------------------------------ steps
    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _template_params(self):
        return dict(
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
        )

    def _grids(self):
        periods = period_grid(
            R_star=self.R_star,
            M_star=self.M_star,
            time_span=numpy.max(self.t) - numpy.min(self.t),
            period_min=self.period_min,
            period_max=self.period_max,
            oversampling_factor=self.oversampling_factor,
            n_transits_min=self.n_transits_min,
        )
        durations = duration_grid(
            periods,
            shortest=1 / len(self.t),
            log_step=self.duration_grid_step,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=self.M_star_min,
            M_star_max=self.M_star_max,
        )
        return periods, durations

    def _templates(self, durations):
        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        lc_cache_overview, lc_arr = get_cache(
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            verbose=self.verbose,
            **self._template_params(),
        )
        return lc_cache_overview, lc_arr

    def _search_order(self, periods):
        order = tls_constants.PERIODS_SEARCH_ORDER
        if order == "ascending":
            return periods[::-1]
        if order == "descending":
            return periods  # it already is
        if order == "shuffled":
            # Local generator: do not consume the user's global numpy random state
            return numpy.random.default_rng().permutation(periods)
        raise ValueError("Unknown PERIODS_SEARCH_ORDER")

    def _problem(self, t, y, dy, lc_cache_overview, lc_arr):
        # masses paired with R_star_min / R_star_max (BUGS.md F1)
        m_short, m_long = duration_limit_masses(
            self.R_star_min, self.R_star_max, self.M_star_min, self.M_star_max
        )
        return SearchProblem(
            t=t,
            y=y,
            dy=dy,
            lc_arr=lc_arr,
            lc_cache_overview=lc_cache_overview,
            transit_depth_min=self.transit_depth_min,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=m_short,
            M_star_max=m_long,
            T0_search_margin=self.T0_search_margin,
        )

    def _search_plan(self, backend, periods, lc_cache_overview, lc_arr):
        """Search tasks [(problem, periods, chi2 offset, row map)]: the
        unbinned light curve, plus binned copies for the periods whose
        shortest trial duration allows it (pre-binning, binning.py)."""
        from transitleastsquares import binning

        periods = numpy.asarray(periods)
        base = self._problem(self.t, self.y, self.dy, lc_cache_overview, lc_arr)
        # pre-binning is an approximation: only for approximate backends
        fraction = 0.0 if getattr(backend, "exact", True) else binning.bin_fraction()
        cad = binning.cadence(self.t)
        k = binning.bin_factors(
            periods, cad, base.R_star_min, base.M_star_min, fraction
        )
        tasks = []
        for kk in numpy.unique(k):
            sel = periods[k == kk]
            if kk == 1:
                tasks.append((base, sel, 0.0, None))
                continue
            tb, yb, dyb, offset = binning.bin_lightcurve(
                self.t, self.y, self.dy, int(kk), cad
            )
            maxwidth = int(numpy.max(lc_cache_overview["duration"]) * len(yb))
            maxwidth += maxwidth % 2
            ov_b, lc_b = get_cache(
                durations=lc_cache_overview["duration"],
                maxwidth_in_samples=maxwidth,
                verbose=False,
                **self._template_params(),
            )
            # rows of the binned cache -> rows of the unbinned cache
            index = {d: i for i, d in enumerate(lc_cache_overview["duration"])}
            row_map = numpy.array([index[d] for d in ov_b["duration"]])
            problem = self._problem(tb, yb, dyb, ov_b, lc_b)
            tasks.append((problem, sel, offset, row_map))
            self._log(
                f"  {len(sel)} periods on {len(yb)} points (bins of {kk} cadences)"
            )
        return tasks

    def _search(self, backend, periods, lc_cache_overview, lc_arr):
        from transitleastsquares.backends import SearchResult

        tasks = self._search_plan(backend, periods, lc_cache_overview, lc_arr)
        pbar = None
        if self.show_progress_bar:
            bar_format = (
                "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} periods "
                "| {elapsed}<{remaining}"
            )
            pbar = tqdm(total=numpy.size(periods), smoothing=0.3, bar_format=bar_format)
        try:
            parts = []
            for problem, sel, offset, row_map in tasks:
                found = backend.search(
                    problem,
                    self._search_order(sel),
                    use_threads=self.use_threads,
                    progress=pbar.update if pbar is not None else None,
                )
                rows = found.rows if row_map is None else row_map[found.rows]
                parts.append((found.periods, found.chi2 + offset, rows, found.depths))
        finally:
            if pbar is not None:
                pbar.close()
        if len(parts) == 1:
            p, c, r, d = parts[0]
        else:
            p, c, r, d = (numpy.concatenate(x) for x in zip(*parts))
        return SearchResult.from_unsorted(p, c, r, d)

    # ------------------------------------------------------------------ power
    def power(self, **kwargs):
        """Compute the periodogram for a set of user-defined parameters"""
        self, kwargs = validate_args(self, kwargs)
        backend = get_backend(self.backend)
        # numba: compile on the first run (with a message), else load the
        # cached kernels once in this process (inherited by forked workers)
        ensure_compiled(self.backend, verbose=self.verbose)
        self._log(tls_constants.TLS_VERSION)

        periods, durations = self._grids()
        lc_cache_overview, lc_arr = self._templates(durations)

        self._log(
            f"Searching {len(self.y)} data points, {len(periods)} periods from "
            f"{round(numpy.min(periods), 3)} to {round(numpy.max(periods), 3)} days"
        )
        if self.use_threads == multiprocessing.cpu_count():
            self._log(f"Using all {self.use_threads} CPU threads")
        else:
            self._log(
                f"Using {self.use_threads} of {multiprocessing.cpu_count()} CPU threads"
            )

        found = self._search(backend, periods, lc_cache_overview, lc_arr)
        return self._results(found, durations, lc_cache_overview, lc_arr, backend)

    # ---------------------------------------------------------------- results
    def _results(self, found, durations, lc_cache_overview, lc_arr, backend):
        chi2 = found.chi2
        chi2red = chi2 / (len(self.t) - DEGREES_OF_FREEDOM)
        r = dict(
            chi2_min=numpy.min(chi2),
            chi2red_min=numpy.min(chi2red),
            periods=found.periods,
            chi2=chi2,
            chi2red=chi2red,
        )

        # A period without any fit keeps best_depth == 0
        no_transits_were_fit = numpy.all(found.depths == 0) or (
            numpy.max(chi2) == numpy.min(chi2)
        )
        if no_transits_were_fit:
            warnings.warn('No transit were fit. Try smaller "transit_depth_min"')
            r.update(_NO_FIT_RESULTS)
            r.update(power=numpy.zeros(len(chi2)), power_raw=numpy.zeros(len(chi2)))
        else:
            r.update(
                self._detection_statistics(
                    found, durations, lc_cache_overview, lc_arr, backend
                )
            )
        r["period_uncertainty"] = period_uncertainty(found.periods, r["power"])
        r["FAP"] = FAP(r["SDE"])
        return transitleastsquaresresults(**r)

    def _detection_statistics(
        self, found, durations, lc_cache_overview, lc_arr, backend
    ):
        t, y, dy = self.t, self.y, self.dy
        SR, power_raw, power, SDE_raw, SDE = spectra(
            found.chi2, self.oversampling_factor
        )
        index_highest_power = numpy.argmax(power)
        period = found.periods[index_highest_power]
        depth = found.depths[index_highest_power]
        # Template row (duration) of the same period as period and depth
        best_row = found.rows[index_highest_power]
        duration = lc_cache_overview["duration"][best_row]

        T0 = backend.fit_T0(
            signal=lc_arr[best_row],
            depth=depth,
            t=t,
            y=y,
            dy=dy,
            period=period,
            T0_fit_margin=self.T0_fit_margin,
            show_progress_bar=self.show_progress_bar,
            verbose=self.verbose,
        )
        transit_times = all_transit_times(T0, t, period)
        transit_duration_in_days = calculate_transit_duration_in_days(
            t, period, transit_times, duration
        )

        # Phase-folded data, mid-transit at phase 0.5
        phases = fold(t, period, T0=T0 + period / 2)
        sort_index = numpy.argsort(phases)

        # Model phase, shifted by half a cadence so that mid-transit is at phase=0.5
        model_folded_phase = numpy.linspace(
            0 + 1 / numpy.size(t) / 2, 1 + 1 / numpy.size(t) / 2, numpy.size(t)
        )
        # Folded model / model curve. Data phase 0.5 is not always at the midpoint
        # (not at cadence: len(y)/2), so the model is stretched accordingly.
        # Note: unlike the cache, maxwidth is *not* rounded up to an even number here
        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(t))
        fill_half = 1 - ((1 - calculate_fill_factor(t)) * 0.5)
        stretch = calculate_stretch(t, period, transit_times)
        internal_samples = (
            int(len(y) / len(transit_times))
        ) * tls_constants.OVERSAMPLE_MODEL_LIGHT_CURVE
        template = self._template_params()
        model_folded_model = fractional_transit(
            duration=duration * maxwidth_in_samples * fill_half,
            maxwidth=maxwidth_in_samples / stretch,
            depth=1 - depth,
            samples=int(len(t)),
            **template,
        )
        model_transit_single = fractional_transit(
            duration=(duration * maxwidth_in_samples),
            maxwidth=maxwidth_in_samples / stretch,
            depth=1 - depth,
            samples=internal_samples,
            **template,
        )
        model_lightcurve_model, model_lightcurve_time = model_lightcurve(
            transit_times, period, t, model_transit_single
        )

        (
            depth_mean_odd,
            depth_mean_even,
            depth_mean_odd_std,
            depth_mean_even_std,
            all_flux_intransit_odd,
            all_flux_intransit_even,
            per_transit_count,
            transit_depths,
            transit_depths_uncertainties,
        ) = intransit_stats(t, y, transit_times, transit_duration_in_days)
        all_flux_intransit = numpy.concatenate(
            [all_flux_intransit_odd, all_flux_intransit_even]
        )
        snr_per_transit, snr_pink_per_transit = snr_stats(
            t=t,
            y=y,
            period=period,
            duration=transit_duration_in_days,
            T0=T0,
            transit_times=transit_times,
            transit_duration_in_days=transit_duration_in_days,
            per_transit_count=per_transit_count,
        )
        intransit = transit_mask(t, period, 2 * transit_duration_in_days, T0)
        flux_ootr = y[~intransit]
        depth_mean = numpy.mean(all_flux_intransit)
        depth_mean_std = numpy.std(all_flux_intransit) / numpy.sum(
            per_transit_count
        ) ** (0.5)
        snr = ((1 - depth_mean) / numpy.std(flux_ootr)) * len(all_flux_intransit) ** (
            0.5
        )

        in_transit_count, after_transit_count, before_transit_count = count_stats(
            t, y, transit_times, transit_duration_in_days
        )

        # Odd even mismatch in standard deviations
        odd_even_mismatch = abs(depth_mean_odd - depth_mean_even) / (
            depth_mean_odd_std + depth_mean_even_std
        )

        transit_count = len(transit_times)
        empty_transit_count = numpy.count_nonzero(per_transit_count == 0)
        if empty_transit_count / transit_count >= 0.33:
            warnings.warn(
                f"{empty_transit_count} of {transit_count} transits without data. "
                "The true period may be twice the given period."
            )

        return dict(
            SDE=SDE,
            SDE_raw=SDE_raw,
            period=period,
            T0=T0,
            duration=transit_duration_in_days,
            depth=depth,
            depth_mean=(depth_mean, depth_mean_std),
            depth_mean_even=(depth_mean_even, depth_mean_even_std),
            depth_mean_odd=(depth_mean_odd, depth_mean_odd_std),
            transit_depths=transit_depths,
            transit_depths_uncertainties=transit_depths_uncertainties,
            rp_rs=rp_rs_from_depth(depth=1 - depth, law=self.limb_dark, params=self.u),
            snr=snr,
            snr_per_transit=snr_per_transit,
            snr_pink_per_transit=snr_pink_per_transit,
            odd_even_mismatch=odd_even_mismatch,
            transit_times=transit_times,
            per_transit_count=per_transit_count,
            transit_count=transit_count,
            distinct_transit_count=transit_count - empty_transit_count,
            empty_transit_count=empty_transit_count,
            in_transit_count=in_transit_count,
            after_transit_count=after_transit_count,
            before_transit_count=before_transit_count,
            power=power,
            power_raw=power_raw,
            SR=SR,
            model_lightcurve_time=model_lightcurve_time,
            model_lightcurve_model=model_lightcurve_model,
            model_folded_phase=model_folded_phase,
            folded_y=y[sort_index],
            folded_dy=dy[sort_index],
            folded_phase=phases[sort_index],
            model_folded_model=model_folded_model,
        )


# Results when no transit was fit at all (flat spectrum)
_NO_FIT_RESULTS = dict(
    SDE=0,
    SDE_raw=0,
    period=numpy.nan,
    T0=0,
    duration=numpy.nan,
    depth=1,
    depth_mean=(numpy.nan, numpy.nan),
    depth_mean_even=(numpy.nan, numpy.nan),
    depth_mean_odd=(numpy.nan, numpy.nan),
    transit_depths=numpy.nan,
    transit_depths_uncertainties=numpy.nan,
    rp_rs=numpy.nan,
    snr=numpy.nan,
    snr_per_transit=numpy.nan,
    snr_pink_per_transit=numpy.nan,
    odd_even_mismatch=numpy.nan,
    transit_times=numpy.nan,
    per_transit_count=numpy.nan,
    transit_count=numpy.nan,
    distinct_transit_count=numpy.nan,
    empty_transit_count=numpy.nan,
    in_transit_count=numpy.nan,
    after_transit_count=numpy.nan,
    before_transit_count=numpy.nan,
    SR=0,
    model_lightcurve_time=numpy.nan,
    model_lightcurve_model=numpy.nan,
    model_folded_phase=numpy.nan,
    folded_y=numpy.nan,
    folded_dy=numpy.nan,
    folded_phase=numpy.nan,
    model_folded_model=numpy.nan,
)
